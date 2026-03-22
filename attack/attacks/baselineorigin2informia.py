import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager


class Baselineorigin2informiaAttack(AbstractAttack):
    """
    dllm-friendly InfoRMIA reproduction.

    This implementation follows the token-level InfoRMIA backbone from the paper:
        score_t = log Ratio_x - E_z[p(z)] log Ratio_z
        Ratio = p(. | target) / p(.)

    In this repo, p(.) is approximated by the reference model token distribution.
    Sequence-level membership is obtained by aggregating token-level scores.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))
        self.aggregation = str(config.get("aggregation", "mean")).lower()
        self.min_k_ratio = float(config.get("min_k_ratio", 0.2))
        self.min_token_count = int(config.get("min_token_count", 1))
        self.masked_token_batch_size = int(config.get("masked_token_batch_size", 16))
        self.dllm_mc_num = int(config.get("dllm_mc_num", 8))
        self.dllm_mask_ratio = float(config.get("dllm_mask_ratio", 0.5))
        self.offline_a = float(config.get("offline_a", 1.0))
        self.log_eps = float(config.get("log_eps", 1e-12))
        self.save_metadata = bool(config.get("save_metadata", False))
        self.save_token_scores = bool(config.get("save_token_scores", True))

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
            raise ValueError("InfoRMIA requires 'reference_model_path' in the config.")
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
            raise ValueError("InfoRMIA requires the dataset to contain a 'text' column.")

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
        target_info, target_meta = self._get_mc_token_predictions(
            self.model,
            self.tokenizer,
            self.target_mask_id,
            self.target_shift_logits,
            text,
        )
        ref_info, ref_meta = self._get_mc_token_predictions(
            self.ref_model,
            self.ref_tokenizer,
            self.ref_mask_id,
            self.ref_shift_logits,
            text,
        )

        all_valid = target_meta["valid"] and ref_meta["valid"]
        if not all_valid:
            return 0.0, {
                "valid": False,
                "aggregation": self.aggregation,
                "target_meta": target_meta,
                "ref_meta": ref_meta,
                "reason": target_meta.get("reason") or ref_meta.get("reason"),
            }

        labels = target_info["labels"]
        token_strings = target_info["token_strings"]
        token_scores_passes: List[torch.Tensor] = []
        for pass_idx in range(self.dllm_mc_num):
            target_probs = target_info["probs_per_pass"][pass_idx]
            ref_probs = ref_info["probs_per_pass"][pass_idx].to(target_probs.device)
            population_probs = self._estimate_population_distribution(ref_probs)
            log_ratio = torch.log(target_probs.clamp_min(self.log_eps)) - torch.log(
                population_probs.clamp_min(self.log_eps)
            )
            pass_scores = log_ratio.gather(1, labels.unsqueeze(1)).squeeze(1) - (
                population_probs * log_ratio
            ).sum(dim=1)
            token_scores_passes.append(pass_scores)

        token_scores = torch.stack(token_scores_passes, dim=0).mean(dim=0)

        final_score = self._aggregate_token_scores(token_scores)
        token_scores_list = token_scores.detach().cpu().tolist()

        meta: Dict[str, Any] = {
            "valid": True,
            "aggregation": self.aggregation,
            "token_count": int(token_scores.numel()),
            "dllm_mc_num": self.dllm_mc_num,
            "dllm_mask_ratio": self.dllm_mask_ratio,
            "target_meta": target_meta,
            "ref_meta": ref_meta,
            "token_score_mean": float(token_scores.mean().item()),
            "token_score_std": float(token_scores.std().item())
            if token_scores.numel() > 1
            else 0.0,
        }
        if self.save_token_scores:
            meta["tokens"] = token_strings
            meta["token_ids"] = labels.detach().cpu().tolist()
            meta["token_scores"] = token_scores_list

        return final_score, meta

    def _get_mc_token_predictions(
        self,
        model,
        tokenizer,
        mask_id: int,
        shift_logits: bool,
        text: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_ids = tokenizer.encode(str(text), add_special_tokens=False)
        if not input_ids:
            return {}, {
                "valid": False,
                "reason": "empty_input",
                "raw_token_count": 0,
            }

        input_ids = input_ids[: self.max_length]
        if len(input_ids) < 1:
            return {}, {
                "valid": False,
                "reason": "too_short",
                "raw_token_count": len(input_ids),
            }

        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.bool, device=model.device)

        seq_len = int(input_tensor.numel())
        positions = torch.arange(seq_len, device=model.device, dtype=torch.long)
        probs_per_pass: List[torch.Tensor] = []

        for _ in range(self.dllm_mc_num):
            pass_probs = self._run_masked_pass(
                model=model,
                input_tensor=input_tensor,
                attention_mask=attention_mask,
                positions=positions,
                mask_id=mask_id,
                shift_logits=shift_logits,
            )
            probs_per_pass.append(pass_probs)

        if not probs_per_pass:
            return {}, {
                "valid": False,
                "reason": "no_usable_positions",
                "raw_token_count": len(input_ids),
            }

        token_strings = tokenizer.convert_ids_to_tokens(input_ids)

        return {
            "probs_per_pass": probs_per_pass,
            "labels": input_tensor,
            "token_strings": token_strings,
        }, {
            "valid": True,
            "reason": "ok",
            "raw_token_count": len(input_ids),
            "scored_token_count": seq_len,
            "mc_passes": self.dllm_mc_num,
            "num_masked_per_row": self._num_masked_per_row(seq_len),
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
    ) -> torch.Tensor:
        seq_len = int(input_tensor.numel())
        chunk_probs: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, seq_len, self.masked_token_batch_size):
                batch_positions = positions[start : start + self.masked_token_batch_size]
                batch_size = int(batch_positions.numel())
                batch_input_ids = input_tensor.unsqueeze(0).repeat(batch_size, 1)
                batch_attention = attention_mask.unsqueeze(0).repeat(batch_size, 1)
                batch_input_ids[
                    torch.arange(batch_size, device=model.device), batch_positions
                ] = mask_id

                num_extra_masks = self._num_extra_masks(seq_len)
                if num_extra_masks > 0:
                    for row_idx, target_pos in enumerate(batch_positions.tolist()):
                        candidate_positions = [idx for idx in range(seq_len) if idx != target_pos]
                        if not candidate_positions:
                            continue
                        picked_count = min(num_extra_masks, len(candidate_positions))
                        extra_positions = self.rng.choice(
                            candidate_positions, size=picked_count, replace=False
                        )
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
            k = max(self.min_token_count, int(np.ceil(self.min_k_ratio * token_scores.numel())))
            k = min(k, int(token_scores.numel()))
            # Higher InfoRMIA scores indicate stronger membership, so we average the
            # most member-like tokens as the sign-adjusted min-k analogue.
            values = torch.topk(token_scores, k=k, largest=True).values
            return float(values.mean().item())

        return float(token_scores.mean().item())

    def _cleanup(self):
        if hasattr(self, "ref_model"):
            self.ref_model.to("cpu")
            del self.ref_model
            del self.ref_tokenizer
        torch.cuda.empty_cache()
