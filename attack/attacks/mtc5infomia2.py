import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager


class Mtc5informia2Attack(AbstractAttack):
    """
    Independent InfoRMIA x MTC5 hybrid attack for dllm models.

    Supported modes:
    - ref_only: uniform token sampling + InfoRMIA reference-ratio score
    - weight_only: reference-loss weighted sampling + target-only centered score
    - hybrid: reference-loss weighted sampling + InfoRMIA reference-ratio score
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))
        self.mode = str(config.get("mode", "hybrid")).lower()
        self.aggregation = str(config.get("aggregation", "mean")).lower()
        self.min_k_ratio = float(config.get("min_k_ratio", 0.2))
        self.min_token_count = int(config.get("min_token_count", 1))
        self.masked_token_batch_size = int(config.get("masked_token_batch_size", 16))
        self.dllm_mc_num = int(config.get("dllm_mc_num", 2))
        self.dllm_mask_ratio = float(config.get("dllm_mask_ratio", 0.3))
        self.sampled_token_ratio = float(config.get("sampled_token_ratio", 0.25))
        self.min_sampled_tokens = int(config.get("min_sampled_tokens", 8))
        self.token_weight_temperature = float(
            config.get("token_weight_temperature", 1.0)
        )
        self.token_weight_floor = float(config.get("token_weight_floor", 1e-8))
        self.offline_a = float(config.get("offline_a", 1.0))
        self.log_eps = float(config.get("log_eps", 1e-12))
        self.save_metadata = bool(config.get("save_metadata", False))
        self.save_token_scores = bool(config.get("save_token_scores", True))

        valid_modes = {"ref_only", "weight_only", "hybrid"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {sorted(valid_modes)}, got {self.mode}"
            )

        self.rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        self.metadata_dir = config.get("metadata_dir") or os.environ.get(
            "SAMA_METADATA_DIR", "./"
        )
        self.metadata_dir = os.path.join(
            self.metadata_dir,
            f"sama_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        if self.save_metadata:
            os.makedirs(self.metadata_dir, exist_ok=True)
            with open(os.path.join(self.metadata_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self.metadata_buffer: List[Dict[str, Any]] = []

        if "model_mask_id" in config and "model_shift_logits" in config:
            self.target_mask_id = config["model_mask_id"]
            self.target_shift_logits = config["model_shift_logits"]
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(
                self.model
            )

        hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        ref_model_path = config.get("reference_model_path")
        if not ref_model_path:
            raise ValueError(
                "Mtc5Informia requires 'reference_model_path' in the config."
            )
        self.ref_device = torch.device(config.get("reference_device", str(device)))
        self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
            ref_model_path, ref_model_path, self.ref_device
        )
        self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

    def run(self, dataset: Dataset) -> Dataset:
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset
        if "text" not in dataset.column_names:
            raise ValueError(
                "Mtc5Informia requires the dataset to contain a 'text' column."
            )

        membership_scores: List[float] = []
        for start_idx in tqdm(
            range(0, n_samples, self.batch_size), desc=f"{self.name}"
        ):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]
            batch_scores = self._compute_batch_scores(batch["text"], start_idx)
            membership_scores.extend(batch_scores)

        dataset = dataset.add_column(self.name, membership_scores)

        if self.save_metadata and self.metadata_buffer:
            metadata_path = os.path.join(self.metadata_dir, "full_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata_buffer, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

        self._cleanup()
        return dataset

    def _compute_batch_scores(
        self, texts: Sequence[str], batch_start_idx: int
    ) -> List[float]:
        batch_scores: List[float] = []

        for offset, text in enumerate(texts):
            sample_idx = batch_start_idx + offset
            final_score, meta = self._score_text(str(text))
            batch_scores.append(final_score)

            if self.save_metadata:
                self.metadata_buffer.append(
                    {
                        "sample_idx": sample_idx,
                        "text": str(text)[:200],
                        "final_score": final_score,
                        **meta,
                    }
                )

        return batch_scores

    def _score_text(self, text: str) -> Tuple[float, Dict[str, Any]]:
        input_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if not input_ids:
            return 0.0, {
                "valid": False,
                "reason": "empty_input",
                "mode": self.mode,
                "aggregation": self.aggregation,
            }

        input_ids = input_ids[: self.max_length]
        seq_len = len(input_ids)
        if seq_len < 1:
            return 0.0, {
                "valid": False,
                "reason": "too_short",
                "mode": self.mode,
                "aggregation": self.aggregation,
            }

        token_weights = (
            self._compute_reference_token_weights(input_ids)
            if self.mode in {"weight_only", "hybrid"}
            else None
        )
        mask_plans = self._build_mask_plans(seq_len, token_weights)

        target_info, target_meta = self._get_mc_token_predictions(
            self.model,
            input_ids,
            self.target_mask_id,
            self.target_shift_logits,
            mask_plans,
        )

        ref_info = None
        ref_meta: Dict[str, Any] = {
            "valid": True,
            "reason": "skipped",
            "raw_token_count": seq_len,
            "sampled_token_count": 0,
        }
        if self.mode in {"ref_only", "hybrid"}:
            ref_info, ref_meta = self._get_mc_token_predictions(
                self.ref_model,
                input_ids,
                self.ref_mask_id,
                self.ref_shift_logits,
                mask_plans,
            )

        if not target_meta["valid"] or not ref_meta["valid"]:
            return 0.0, {
                "valid": False,
                "reason": target_meta.get("reason") or ref_meta.get("reason"),
                "mode": self.mode,
                "aggregation": self.aggregation,
                "target_meta": target_meta,
                "ref_meta": ref_meta,
            }

        token_sum = torch.zeros(seq_len, dtype=torch.float32, device=self.device)
        token_count = torch.zeros(seq_len, dtype=torch.float32, device=self.device)

        for pass_idx in range(self.dllm_mc_num):
            target_pass = target_info["passes"][pass_idx]
            positions = target_pass["positions"]
            labels = target_pass["labels"]
            target_probs = target_pass["probs"]

            if self.mode == "weight_only":
                pass_scores = self._compute_target_only_scores(target_probs, labels)
            else:
                ref_probs = ref_info["passes"][pass_idx]["probs"].to(
                    target_probs.device
                )
                pass_scores = self._compute_informia_scores(
                    target_probs=target_probs,
                    ref_probs=ref_probs,
                    labels=labels,
                )

            token_sum[positions] += pass_scores
            token_count[positions] += 1.0

        covered_mask = token_count > 0
        covered_count = int(covered_mask.sum().item())
        if covered_count == 0:
            return 0.0, {
                "valid": False,
                "reason": "no_tokens_sampled",
                "mode": self.mode,
                "aggregation": self.aggregation,
                "target_meta": target_meta,
                "ref_meta": ref_meta,
            }

        token_scores_full = torch.full(
            (seq_len,), float("nan"), dtype=torch.float32, device=self.device
        )
        token_scores_full[covered_mask] = (
            token_sum[covered_mask] / token_count[covered_mask]
        )
        covered_scores = token_scores_full[covered_mask]
        final_score = self._aggregate_token_scores(covered_scores)
        token_strings = self.tokenizer.convert_ids_to_tokens(input_ids)

        meta: Dict[str, Any] = {
            "valid": True,
            "mode": self.mode,
            "aggregation": self.aggregation,
            "token_count": seq_len,
            "covered_token_count": covered_count,
            "coverage_ratio": covered_count / max(1, seq_len),
            "dllm_mc_num": self.dllm_mc_num,
            "dllm_mask_ratio": self.dllm_mask_ratio,
            "sampled_token_ratio": self.sampled_token_ratio,
            "min_sampled_tokens": self.min_sampled_tokens,
            "token_score_mean": float(covered_scores.mean().item()),
            "token_score_std": (
                float(covered_scores.std().item())
                if covered_scores.numel() > 1
                else 0.0
            ),
            "target_meta": target_meta,
            "ref_meta": ref_meta,
        }

        if token_weights is not None:
            meta["token_weight_mean"] = float(token_weights.mean())
            meta["token_weight_max"] = float(token_weights.max())
            meta["token_weight_min"] = float(token_weights.min())

        if self.save_token_scores:
            token_scores_list = [
                None if torch.isnan(score) else float(score.item())
                for score in token_scores_full.detach().cpu()
            ]
            meta["tokens"] = token_strings
            meta["token_ids"] = list(map(int, input_ids))
            meta["token_scores"] = token_scores_list
            if token_weights is not None:
                meta["token_weights"] = token_weights.tolist()

        return final_score, meta

    def _compute_reference_token_weights(self, input_ids: Sequence[int]) -> np.ndarray:
        ref_model_device = self._model_device(self.ref_model)
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=ref_model_device
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.bool)

        with torch.no_grad():
            masked_input = torch.full_like(input_tensor, self.ref_mask_id)
            out = self.ref_model(
                input_ids=masked_input,
                attention_mask=(attention_mask if not self.ref_shift_logits else None),
            )
            logits = out.logits if hasattr(out, "logits") else out[0]
            if self.ref_shift_logits:
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

            ce = (
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    input_tensor.view(-1),
                    reduction="none",
                )
                .view(1, -1)
                .float()
            )

        weights = ce[0] * attention_mask[0].float()
        weights = weights.clamp_min(self.token_weight_floor)
        log_weights = torch.log(weights)
        log_weights = log_weights / max(self.token_weight_temperature, 1e-8)
        normalized = F.softmax(log_weights, dim=0)
        return normalized.detach().cpu().numpy()

    def _build_mask_plans(
        self, seq_len: int, token_weights: Optional[np.ndarray]
    ) -> List[Dict[str, List[List[int]]]]:
        mask_plans: List[Dict[str, List[List[int]]]] = []
        num_extra_masks = self._num_extra_masks(seq_len)

        for _ in range(self.dllm_mc_num):
            sampled_positions = self._sample_target_positions(seq_len, token_weights)
            extra_masks: List[List[int]] = []
            for target_pos in sampled_positions:
                if num_extra_masks <= 0:
                    extra_masks.append([])
                    continue

                candidates = [idx for idx in range(seq_len) if idx != target_pos]
                if not candidates:
                    extra_masks.append([])
                    continue

                picked_count = min(num_extra_masks, len(candidates))
                if token_weights is not None:
                    candidate_weights = token_weights[candidates]
                    candidate_weights = candidate_weights / candidate_weights.sum()
                    chosen = self.rng.choice(
                        candidates,
                        size=picked_count,
                        replace=False,
                        p=candidate_weights,
                    )
                else:
                    chosen = self.rng.choice(
                        candidates, size=picked_count, replace=False
                    )
                extra_masks.append(np.asarray(chosen, dtype=np.int64).tolist())

            mask_plans.append(
                {
                    "sampled_positions": sampled_positions,
                    "extra_masks": extra_masks,
                }
            )

        return mask_plans

    def _sample_target_positions(
        self, seq_len: int, token_weights: Optional[np.ndarray]
    ) -> List[int]:
        if seq_len <= 0:
            return []

        sampled_count = max(
            self.min_sampled_tokens,
            int(np.ceil(self.sampled_token_ratio * seq_len)),
        )
        sampled_count = min(sampled_count, seq_len)

        candidates = np.arange(seq_len, dtype=np.int64)
        if token_weights is not None:
            weights = token_weights[candidates]
            weights = weights / weights.sum()
            chosen = self.rng.choice(
                candidates, size=sampled_count, replace=False, p=weights
            )
        else:
            chosen = self.rng.choice(candidates, size=sampled_count, replace=False)

        chosen = np.asarray(chosen, dtype=np.int64)
        chosen.sort()
        return chosen.tolist()

    def _get_mc_token_predictions(
        self,
        model,
        input_ids: Sequence[int],
        mask_id: int,
        shift_logits: bool,
        mask_plans: List[Dict[str, List[List[int]]]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model_device = self._model_device(model)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model_device)
        attention_mask = torch.ones_like(
            input_tensor, dtype=torch.bool, device=model_device
        )
        passes: List[Dict[str, torch.Tensor]] = []
        total_sampled = 0

        for pass_plan in mask_plans:
            sampled_positions = pass_plan["sampled_positions"]
            if not sampled_positions:
                continue

            pass_probs, positions_tensor = self._run_masked_pass(
                model=model,
                input_tensor=input_tensor,
                attention_mask=attention_mask,
                sampled_positions=sampled_positions,
                extra_masks=pass_plan["extra_masks"],
                mask_id=mask_id,
                shift_logits=shift_logits,
            )
            labels = input_tensor[positions_tensor]
            passes.append(
                {
                    "positions": positions_tensor.to(self.device),
                    "labels": labels.to(self.device),
                    "probs": pass_probs.to(self.device),
                }
            )
            total_sampled += int(positions_tensor.numel())

        if not passes:
            return {}, {
                "valid": False,
                "reason": "no_usable_positions",
                "raw_token_count": len(input_ids),
                "sampled_token_count": 0,
            }

        return {"passes": passes}, {
            "valid": True,
            "reason": "ok",
            "raw_token_count": len(input_ids),
            "sampled_token_count": total_sampled,
            "mc_passes": self.dllm_mc_num,
            "masked_token_batch_size": self.masked_token_batch_size,
        }

    def _run_masked_pass(
        self,
        model,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        sampled_positions: Sequence[int],
        extra_masks: Sequence[Sequence[int]],
        mask_id: int,
        shift_logits: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.tensor(
            sampled_positions, dtype=torch.long, device=input_tensor.device
        )
        chunk_probs: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, int(positions.numel()), self.masked_token_batch_size):
                batch_positions = positions[
                    start : start + self.masked_token_batch_size
                ]
                batch_size = int(batch_positions.numel())
                batch_input_ids = input_tensor.unsqueeze(0).repeat(batch_size, 1)
                batch_attention = attention_mask.unsqueeze(0).repeat(batch_size, 1)

                batch_input_ids[
                    torch.arange(batch_size, device=input_tensor.device),
                    batch_positions,
                ] = mask_id

                for row_idx, local_idx in enumerate(range(start, start + batch_size)):
                    extra_positions = extra_masks[local_idx]
                    if extra_positions:
                        extra_tensor = torch.tensor(
                            extra_positions,
                            device=input_tensor.device,
                            dtype=torch.long,
                        )
                        batch_input_ids[row_idx, extra_tensor] = mask_id

                out = model(
                    input_ids=batch_input_ids,
                    attention_mask=(batch_attention if not shift_logits else None),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                if shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                selected_logits = logits[
                    torch.arange(batch_size, device=input_tensor.device),
                    batch_positions,
                    :,
                ].float()
                chunk_probs.append(F.softmax(selected_logits, dim=-1))

        return torch.cat(chunk_probs, dim=0), positions

    def _compute_informia_scores(
        self, target_probs: torch.Tensor, ref_probs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        ref_probs = ref_probs.to(target_probs.device)
        population_probs = self._estimate_population_distribution(ref_probs)
        log_ratio = torch.log(target_probs.clamp_min(self.log_eps)) - torch.log(
            population_probs.clamp_min(self.log_eps)
        )
        return log_ratio.gather(1, labels.unsqueeze(1)).squeeze(1) - (
            population_probs * log_ratio
        ).sum(dim=1)

    def _compute_target_only_scores(
        self, target_probs: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        log_target = torch.log(target_probs.clamp_min(self.log_eps))
        return log_target.gather(1, labels.unsqueeze(1)).squeeze(1) - (
            target_probs * log_target
        ).sum(dim=1)

    def _num_extra_masks(self, seq_len: int) -> int:
        if seq_len <= 1:
            return 0
        total_masks = self._num_masked_per_row(seq_len)
        return max(0, min(seq_len - 1, total_masks - 1))

    def _num_masked_per_row(self, seq_len: int) -> int:
        mask_ratio = min(max(self.dllm_mask_ratio, 0.0), 1.0)
        total_masks = max(1, int(round(mask_ratio * seq_len)))
        return min(total_masks, seq_len)

    def _estimate_population_distribution(
        self, ref_probs: torch.Tensor
    ) -> torch.Tensor:
        if self.offline_a >= 1.0:
            return ref_probs

        vocab_size = ref_probs.size(-1)
        uniform = torch.full_like(ref_probs, 1.0 / float(vocab_size))
        alpha = max(0.0, min(1.0, (1.0 + self.offline_a) / 2.0))
        return alpha * ref_probs + (1.0 - alpha) * uniform

    def _aggregate_token_scores(self, token_scores: torch.Tensor) -> float:
        if token_scores.numel() == 0:
            return 0.0

        if self.aggregation == "min_k":
            k = max(
                self.min_token_count,
                int(np.ceil(self.min_k_ratio * token_scores.numel())),
            )
            k = min(k, int(token_scores.numel()))
            values = torch.topk(token_scores, k=k, largest=True).values
            return float(values.mean().item())

        return float(token_scores.mean().item())

    def _model_device(self, model) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _cleanup(self):
        if hasattr(self, "ref_model"):
            self.ref_model.to("cpu")
            del self.ref_model
            del self.ref_tokenizer
        torch.cuda.empty_cache()
