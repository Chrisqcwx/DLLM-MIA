import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params


class Baselineorigin2conrecallAttack(AbstractAttack):
    """
    dllm-only CON-RECALL attack.

    Score:
        (LL(x | P_nonmember) - gamma * LL(x | P_member)) / LL(x)

    For diffusion-style models, each conditional log-likelihood is estimated by
    repeated partial masking over target tokens and averaging token scores across
    Monte Carlo passes.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))
        self.gamma = float(config.get("gamma", 1.0))
        self.n_shots = int(config.get("n_shots", 7))
        self.dllm_mc_num = int(config.get("dllm_mc_num", 8))
        self.dllm_mask_ratio = float(config.get("dllm_mask_ratio", 0.5))
        self.prefix_separator = str(config.get("prefix_separator", "\n\n"))
        self.prefix_max_chars_per_shot = config.get("prefix_max_chars_per_shot")
        self.save_metadata = bool(config.get("save_metadata", False))

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

        self.dataset_texts: List[str] = []
        self.dataset_labels: List[int] = []
        self.member_indices: List[int] = []
        self.nonmember_indices: List[int] = []

    def run(self, dataset: Dataset) -> Dataset:
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset

        self.build_prefix_pools(dataset)

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

    def build_prefix_pools(self, dataset: Dataset) -> None:
        if "text" not in dataset.column_names or "label" not in dataset.column_names:
            raise ValueError("CON-RECALL requires dataset columns 'text' and 'label'.")

        self.dataset_texts = [str(text) for text in dataset["text"]]
        self.dataset_labels = [int(label) for label in dataset["label"]]
        self.member_indices = [
            idx for idx, label in enumerate(self.dataset_labels) if label == 1
        ]
        self.nonmember_indices = [
            idx for idx, label in enumerate(self.dataset_labels) if label == 0
        ]

        if not self.member_indices or not self.nonmember_indices:
            raise ValueError(
                "CON-RECALL needs both member(label=1) and non-member(label=0) samples."
            )

    def _compute_batch_scores(
        self, texts: Sequence[str], batch_start_idx: int
    ) -> List[float]:
        batch_scores: List[float] = []

        for offset, target_text in enumerate(texts):
            sample_idx = batch_start_idx + offset
            sample_label = self.dataset_labels[sample_idx]

            prefix_info = self.select_prefixes_for_sample(sample_idx)
            member_prefixes = prefix_info["member_texts"]
            nonmember_prefixes = prefix_info["nonmember_texts"]

            base_condition = self.compose_prefixed_text([], target_text)
            member_condition = self.compose_prefixed_text(member_prefixes, target_text)
            nonmember_condition = self.compose_prefixed_text(
                nonmember_prefixes, target_text
            )

            base_ll, base_meta = self.estimate_dllm_loglikelihood(base_condition)
            member_ll, member_meta = self.estimate_dllm_loglikelihood(member_condition)
            nonmember_ll, nonmember_meta = self.estimate_dllm_loglikelihood(
                nonmember_condition
            )

            all_valid = (
                base_meta["valid"] and member_meta["valid"] and nonmember_meta["valid"]
            )
            final_score = (
                self.compute_conrecall_score(base_ll, member_ll, nonmember_ll)
                if all_valid
                else 0.0
            )
            batch_scores.append(final_score)

            if self.save_metadata:
                self.metadata_buffer.append(
                    {
                        "sample_idx": sample_idx,
                        "label": sample_label,
                        "text": str(target_text)[:200],
                        "final_score": final_score,
                        "valid": all_valid,
                        "gamma": self.gamma,
                        "member_prefix_indices": prefix_info["member_indices"],
                        "nonmember_prefix_indices": prefix_info["nonmember_indices"],
                        "num_member_prefixes": len(member_prefixes),
                        "num_nonmember_prefixes": len(nonmember_prefixes),
                        "base_ll": base_ll,
                        "member_ll": member_ll,
                        "nonmember_ll": nonmember_ll,
                        "base_meta": base_meta,
                        "member_meta": member_meta,
                        "nonmember_meta": nonmember_meta,
                    }
                )

        return batch_scores

    def select_prefixes_for_sample(self, sample_idx: int) -> Dict[str, Any]:
        member_candidates = [idx for idx in self.member_indices if idx != sample_idx]
        nonmember_candidates = [
            idx for idx in self.nonmember_indices if idx != sample_idx
        ]

        member_rng = np.random.default_rng(self.seed + sample_idx * 2 + 1)
        nonmember_rng = np.random.default_rng(self.seed + sample_idx * 2 + 2)

        member_pick_count = min(self.n_shots, len(member_candidates))
        nonmember_pick_count = min(self.n_shots, len(nonmember_candidates))

        member_indices = (
            member_rng.choice(
                member_candidates, size=member_pick_count, replace=False
            ).tolist()
            if member_pick_count > 0
            else []
        )
        nonmember_indices = (
            nonmember_rng.choice(
                nonmember_candidates, size=nonmember_pick_count, replace=False
            ).tolist()
            if nonmember_pick_count > 0
            else []
        )

        return {
            "member_indices": member_indices,
            "nonmember_indices": nonmember_indices,
            "member_texts": [self.dataset_texts[idx] for idx in member_indices],
            "nonmember_texts": [self.dataset_texts[idx] for idx in nonmember_indices],
        }

    def compose_prefixed_text(
        self, prefix_texts: Sequence[str], target_text: str
    ) -> Dict[str, Any]:
        processed_prefixes = []
        for text in prefix_texts:
            if self.prefix_max_chars_per_shot is not None:
                processed_prefixes.append(str(text)[: int(self.prefix_max_chars_per_shot)])
            else:
                processed_prefixes.append(str(text))

        prefix_text = self.prefix_separator.join(
            [text for text in processed_prefixes if text]
        )

        target_ids = self.tokenizer.encode(str(target_text), add_special_tokens=False)
        separator_ids = (
            self.tokenizer.encode(self.prefix_separator, add_special_tokens=False)
            if prefix_text
            else []
        )
        prefix_ids = (
            self.tokenizer.encode(prefix_text, add_special_tokens=False)
            if prefix_text
            else []
        )

        if len(target_ids) >= self.max_length:
            target_ids = target_ids[: self.max_length]
            prefix_block_ids: List[int] = []
        else:
            prefix_budget = self.max_length - len(target_ids)
            if prefix_text and prefix_budget > len(separator_ids):
                max_prefix_ids = prefix_budget - len(separator_ids)
                kept_prefix_ids = (
                    prefix_ids[-max_prefix_ids:]
                    if max_prefix_ids < len(prefix_ids)
                    else prefix_ids
                )
                prefix_block_ids = kept_prefix_ids + separator_ids
            else:
                prefix_block_ids = []

        input_ids = prefix_block_ids + target_ids
        target_start = len(prefix_block_ids)
        target_end = len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.bool),
            "target_start": target_start,
            "target_end": target_end,
            "target_token_count": len(target_ids),
            "used_prefix_count": len(processed_prefixes),
            "prefix_preview": prefix_text[:200],
        }

    def estimate_dllm_loglikelihood(self, condition_inputs: Dict[str, Any]):
        input_ids = condition_inputs["input_ids"]
        attention_mask = condition_inputs["attention_mask"]
        target_start = int(condition_inputs["target_start"])
        target_end = int(condition_inputs["target_end"])

        if input_ids.numel() == 0 or target_end <= target_start:
            return 0.0, {
                "valid": False,
                "reason": "empty_target_span",
                "target_token_count": 0,
                "covered_token_count": 0,
                "coverage_ratio": 0.0,
                "mc_passes": self.dllm_mc_num,
            }

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        target_positions = torch.arange(target_start, target_end, device=self.device)
        valid_positions = target_positions[attention_mask[target_positions]]

        if valid_positions.numel() == 0:
            return 0.0, {
                "valid": False,
                "reason": "no_valid_target_tokens",
                "target_token_count": int(target_positions.numel()),
                "covered_token_count": 0,
                "coverage_ratio": 0.0,
                "mc_passes": self.dllm_mc_num,
            }

        num_target_tokens = int(valid_positions.numel())
        token_loss_sum = torch.zeros(
            num_target_tokens, dtype=torch.float32, device=self.device
        )
        token_count = torch.zeros(
            num_target_tokens, dtype=torch.float32, device=self.device
        )

        mask_ratio = min(max(self.dllm_mask_ratio, 0.0), 1.0)
        num_to_mask = max(1, int(round(mask_ratio * num_target_tokens)))
        num_to_mask = min(num_to_mask, num_target_tokens)

        with torch.no_grad():
            for _ in range(self.dllm_mc_num):
                local_indices = torch.randperm(num_target_tokens, device=self.device)[
                    :num_to_mask
                ]
                masked_positions = valid_positions[local_indices]

                masked_ids = input_ids.clone()
                masked_ids[masked_positions] = self.target_mask_id

                out = self.model(
                    input_ids=masked_ids.unsqueeze(0),
                    attention_mask=(
                        attention_mask.unsqueeze(0)
                        if not self.target_shift_logits
                        else None
                    ),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                if self.target_shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                masked_logits = logits[0, masked_positions, :]
                masked_labels = input_ids[masked_positions]
                nll = F.cross_entropy(
                    masked_logits, masked_labels, reduction="none"
                ).float()

                token_loss_sum[local_indices] += nll
                token_count[local_indices] += 1.0

        covered_mask = token_count > 0
        covered_token_count = int(covered_mask.sum().item())

        if covered_token_count == 0:
            return 0.0, {
                "valid": False,
                "reason": "no_tokens_covered",
                "target_token_count": num_target_tokens,
                "covered_token_count": 0,
                "coverage_ratio": 0.0,
                "mc_passes": self.dllm_mc_num,
            }

        avg_nll = token_loss_sum[covered_mask] / token_count[covered_mask]
        ll = -avg_nll.mean().item()

        return ll, {
            "valid": True,
            "reason": "ok",
            "target_token_count": num_target_tokens,
            "covered_token_count": covered_token_count,
            "coverage_ratio": covered_token_count / max(1, num_target_tokens),
            "mc_passes": self.dllm_mc_num,
            "num_masked_per_pass": num_to_mask,
            "avg_nll_mean": float(avg_nll.mean().item()),
            "avg_nll_std": float(avg_nll.std().item()) if covered_token_count > 1 else 0.0,
            "used_prefix_count": int(condition_inputs["used_prefix_count"]),
        }

    def compute_conrecall_score(
        self, base_ll: float, member_ll: float, nonmember_ll: float
    ) -> float:
        if abs(base_ll) < 1e-8:
            base_ll = -1e-8 if base_ll < 0 else 1e-8
        return float((nonmember_ll - self.gamma * member_ll) / base_ll)

    def _cleanup(self):
        torch.cuda.empty_cache()
