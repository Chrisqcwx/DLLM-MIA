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


class Mtc5informiaAttack(AbstractAttack):
    """
    Independent InfoRMIA x MTC5 hybrid attack for dllm models.

    Supported modes:
    - ref_only: uniform extra masking + InfoRMIA reference-ratio score
    - weight_only: reference-loss weighted masking + target-only centered score
    - hybrid: reference-loss weighted masking + InfoRMIA reference-ratio score
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
            raise ValueError(f"mode must be one of {sorted(valid_modes)}, got {self.mode}")

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
            raise ValueError("Mtc5Informia requires 'reference_model_path' in the config.")
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
        if len(input_ids) < 1:
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
        mask_plans = self._build_mask_plans(len(input_ids), token_weights)

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
            "raw_token_count": len(input_ids),
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

        labels = target_info["labels"]
        token_strings = target_info["token_strings"]
        token_scores_passes: List[torch.Tensor] = []

        for pass_idx in range(self.dllm_mc_num):
            target_probs = target_info["probs_per_pass"][pass_idx]
            if self.mode == "weight_only":
                pass_scores = self._compute_target_only_scores(target_probs, labels)
            else:
                ref_probs = ref_info["probs_per_pass"][pass_idx].to(target_probs.device)
                pass_scores = self._compute_informia_scores(
                    target_probs=target_probs,
                    ref_probs=ref_probs,
                    labels=labels,
                )
            token_scores_passes.append(pass_scores)

        token_scores = torch.stack(token_scores_passes, dim=0).mean(dim=0)
        final_score = self._aggregate_token_scores(token_scores)

        meta: Dict[str, Any] = {
            "valid": True,
            "mode": self.mode,
            "aggregation": self.aggregation,
            "token_count": int(token_scores.numel()),
            "dllm_mc_num": self.dllm_mc_num,
            "dllm_mask_ratio": self.dllm_mask_ratio,
            "token_score_mean": float(token_scores.mean().item()),
            "token_score_std": float(token_scores.std().item())
            if token_scores.numel() > 1
            else 0.0,
            "target_meta": target_meta,
            "ref_meta": ref_meta,
        }

        if token_weights is not None:
            meta["token_weight_mean"] = float(token_weights.mean())
            meta["token_weight_max"] = float(token_weights.max())
            meta["token_weight_min"] = float(token_weights.min())

        if self.save_token_scores:
            meta["tokens"] = token_strings
            meta["token_ids"] = labels.detach().cpu().tolist()
            meta["token_scores"] = token_scores.detach().cpu().tolist()
            if token_weights is not None:
                meta["token_weights"] = token_weights.tolist()

        return final_score, meta

    def _compute_reference_token_weights(self, input_ids: Sequence[int]) -> np.ndarray:
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=self.ref_model.device
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
    ) -> List[List[List[int]]]:
        mask_plans: List[List[List[int]]] = []
        num_extra_masks = self._num_extra_masks(seq_len)

        for _ in range(self.dllm_mc_num):
            pass_plan: List[List[int]] = []
            for target_pos in range(seq_len):
                if num_extra_masks == 0:
                    pass_plan.append([])
                    continue

                candidates = [idx for idx in range(seq_len) if idx != target_pos]
                picked_count = min(num_extra_masks, len(candidates))
                if picked_count <= 0:
                    pass_plan.append([])
                    continue

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
                chosen = np.asarray(chosen, dtype=np.int64).tolist()
                pass_plan.append(chosen)
            mask_plans.append(pass_plan)

        return mask_plans

    def _get_mc_token_predictions(
        self,
        model,
        input_ids: Sequence[int],
        mask_id: int,
        shift_logits: bool,
        mask_plans: List[List[List[int]]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.bool, device=model.device)
        positions = torch.arange(len(input_ids), device=model.device, dtype=torch.long)
        probs_per_pass: List[torch.Tensor] = []

        for pass_plan in mask_plans:
            pass_probs = self._run_masked_pass(
                model=model,
                input_tensor=input_tensor,
                attention_mask=attention_mask,
                positions=positions,
                mask_id=mask_id,
                shift_logits=shift_logits,
                pass_plan=pass_plan,
            )
            probs_per_pass.append(pass_probs)

        token_strings = self.tokenizer.convert_ids_to_tokens(input_ids)
        return {
            "probs_per_pass": probs_per_pass,
            "labels": input_tensor,
            "token_strings": token_strings,
        }, {
            "valid": True,
            "reason": "ok",
            "raw_token_count": len(input_ids),
            "scored_token_count": len(input_ids),
            "mc_passes": self.dllm_mc_num,
            "num_masked_per_row": self._num_masked_per_row(len(input_ids)),
            "masked_token_batch_size": self.masked_token_batch_size,
        }

    def _run_masked_pass(
        self,
        model,
        input_tensor: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        mask_id: int,
        shift_logits: bool,
        pass_plan: List[List[int]],
    ) -> torch.Tensor:
        seq_len = int(input_tensor.numel())
        chunk_probs: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, seq_len, self.masked_token_batch_size):
                batch_positions = positions[start : start + self.masked_token_batch_size]
                batch_size = int(batch_positions.numel())
                batch_input_ids = input_tensor.unsqueeze(0).repeat(batch_size, 1)
                batch_attention = attention_mask.unsqueeze(0).repeat(batch_size, 1)

                for row_idx, target_pos in enumerate(batch_positions.tolist()):
                    batch_input_ids[row_idx, target_pos] = mask_id
                    extra_positions = pass_plan[target_pos]
                    if extra_positions:
                        extra_positions = torch.tensor(
                            extra_positions, device=model.device, dtype=torch.long
                        )
                        batch_input_ids[row_idx, extra_positions] = mask_id

                out = model(
                    input_ids=batch_input_ids,
                    attention_mask=(batch_attention if not shift_logits else None),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                if shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                selected_logits = logits[
                    torch.arange(batch_size, device=model.device), batch_positions, :
                ].float()
                chunk_probs.append(F.softmax(selected_logits, dim=-1))

        return torch.cat(chunk_probs, dim=0)

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

    def _estimate_population_distribution(self, ref_probs: torch.Tensor) -> torch.Tensor:
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

    def _cleanup(self):
        if hasattr(self, "ref_model"):
            self.ref_model.to("cpu")
            del self.ref_model
            del self.ref_tokenizer
        torch.cuda.empty_cache()
