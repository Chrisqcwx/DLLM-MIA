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


class Baselineorigin2dfmiaAttack(AbstractAttack):
    """
    dllm-oriented DF-MIA approximation for fine-tuned language models.

    This implementation follows the paper's high-level two-stage intuition:
    1. Use a reference-free score to identify likely non-members inside the
       evaluation set.
    2. Build a pseudo non-member calibration pool from those samples and their
       perturbations, then calibrate a loss-gap score against that pool.

    For dllm models, token losses are estimated by repeated partial masking and
    averaging over covered target tokens.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))
        self.dllm_mc_num = int(config.get("dllm_mc_num", 8))
        self.dllm_mask_ratio = float(config.get("dllm_mask_ratio", 0.5))

        self.pseudo_nonmember_ratio = float(config.get("pseudo_nonmember_ratio", 0.5))
        self.min_pseudo_nonmembers = int(config.get("min_pseudo_nonmembers", 64))
        self.max_pseudo_nonmembers = int(config.get("max_pseudo_nonmembers", 512))
        self.calibration_samples = int(config.get("calibration_samples", 256))
        self.num_perturbations = int(config.get("num_perturbations", 1))
        self.perturbation_ratio = float(config.get("perturbation_ratio", 0.15))
        self.score_stat = str(config.get("score_stat", "median_mad")).lower()
        self.stage1_score_column = config.get("stage1_score_column", "nlloss")
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

        ref_model_path = config.get("reference_model_path")
        if not ref_model_path:
            raise ValueError("DF-MIA requires 'reference_model_path' in the config.")
        self.ref_device = torch.device(config.get("reference_device", str(device)))
        self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
            ref_model_path, ref_model_path, self.ref_device
        )
        self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

        self.calibration_stats: Dict[str, float] = {}
        self.pseudo_nonmember_indices: List[int] = []

    def run(self, dataset: Dataset) -> Dataset:
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset
        if "text" not in dataset.column_names:
            raise ValueError("DF-MIA requires the dataset to contain a 'text' column.")

        texts = [str(text) for text in dataset["text"]]
        stage1_scores = self._get_stage1_scores(dataset, texts)
        self.pseudo_nonmember_indices = self._select_pseudo_nonmembers(stage1_scores)
        self.calibration_stats = self._build_calibration_stats(texts, stage1_scores)

        membership_scores: List[float] = []
        for start_idx in tqdm(
            range(0, n_samples, self.batch_size), desc=f"{self.name}"
        ):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_scores = self._compute_batch_scores(
                texts[start_idx:end_idx], stage1_scores[start_idx:end_idx], start_idx
            )
            membership_scores.extend(batch_scores)

        dataset = dataset.add_column(self.name, membership_scores)

        if self.save_metadata and self.metadata_buffer:
            metadata_path = os.path.join(self.metadata_dir, "full_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata_buffer, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

        self._cleanup()
        return dataset

    def _get_stage1_scores(self, dataset: Dataset, texts: Sequence[str]) -> List[float]:
        if self.stage1_score_column in dataset.column_names:
            return [float(v) for v in dataset[self.stage1_score_column]]

        scores: List[float] = []
        for text in tqdm(texts, desc=f"{self.name}-stage1"):
            nll, _ = self._estimate_condition_nll(
                self.model,
                self.target_mask_id,
                self.target_shift_logits,
                self.tokenizer,
                text,
            )
            scores.append(nll)
        return scores

    def _select_pseudo_nonmembers(self, stage1_scores: Sequence[float]) -> List[int]:
        n_samples = len(stage1_scores)
        desired = int(round(n_samples * self.pseudo_nonmember_ratio))
        desired = max(self.min_pseudo_nonmembers, desired)
        desired = min(self.max_pseudo_nonmembers, desired, n_samples)
        ranked_indices = np.argsort(np.asarray(stage1_scores))[::-1]
        return ranked_indices[:desired].tolist()

    def _build_calibration_stats(
        self, texts: Sequence[str], stage1_scores: Sequence[float]
    ) -> Dict[str, float]:
        if not self.pseudo_nonmember_indices:
            raise ValueError("Pseudo non-member pool is empty.")

        ranked_pool = sorted(
            self.pseudo_nonmember_indices,
            key=lambda idx: stage1_scores[idx],
            reverse=True,
        )
        pool = ranked_pool[: min(len(ranked_pool), self.calibration_samples)]

        calibration_texts: List[Tuple[int, str, str]] = []
        for idx in pool:
            original_text = texts[idx]
            calibration_texts.append((idx, original_text, "original"))
            for perturb_id in range(self.num_perturbations):
                perturbed = self._perturb_text(original_text)
                calibration_texts.append((idx, perturbed, f"perturb_{perturb_id}"))

        gaps: List[float] = []
        entries_meta: List[Dict[str, Any]] = []
        for source_idx, sample_text, source_type in tqdm(
            calibration_texts, desc=f"{self.name}-calibration"
        ):
            target_nll, target_meta = self._estimate_condition_nll(
                self.model,
                self.target_mask_id,
                self.target_shift_logits,
                self.tokenizer,
                sample_text,
            )
            ref_nll, ref_meta = self._estimate_condition_nll(
                self.ref_model,
                self.ref_mask_id,
                self.ref_shift_logits,
                self.ref_tokenizer,
                sample_text,
            )
            gap = ref_nll - target_nll
            gaps.append(gap)

            if self.save_metadata:
                entries_meta.append(
                    {
                        "source_idx": source_idx,
                        "source_type": source_type,
                        "text_preview": sample_text[:200],
                        "target_nll": target_nll,
                        "ref_nll": ref_nll,
                        "gap": gap,
                        "target_meta": target_meta,
                        "ref_meta": ref_meta,
                    }
                )

        stats = self._summarize_distribution(gaps)
        if self.save_metadata:
            stats["pool_metadata"] = entries_meta
        return stats

    def _compute_batch_scores(
        self, texts: Sequence[str], stage1_scores: Sequence[float], batch_start_idx: int
    ) -> List[float]:
        batch_scores: List[float] = []

        for offset, text in enumerate(texts):
            sample_idx = batch_start_idx + offset
            target_nll, target_meta = self._estimate_condition_nll(
                self.model,
                self.target_mask_id,
                self.target_shift_logits,
                self.tokenizer,
                text,
            )
            ref_nll, ref_meta = self._estimate_condition_nll(
                self.ref_model,
                self.ref_mask_id,
                self.ref_shift_logits,
                self.ref_tokenizer,
                text,
            )
            gap = ref_nll - target_nll
            final_score = self._calibrate_gap(gap)
            batch_scores.append(final_score)

            if self.save_metadata:
                self.metadata_buffer.append(
                    {
                        "sample_idx": sample_idx,
                        "text": text[:200],
                        "stage1_score": float(stage1_scores[offset]),
                        "target_nll": target_nll,
                        "ref_nll": ref_nll,
                        "gap": gap,
                        "final_score": final_score,
                        "calibration_center": self.calibration_stats["center"],
                        "calibration_scale": self.calibration_stats["scale"],
                        "target_meta": target_meta,
                        "ref_meta": ref_meta,
                    }
                )

        return batch_scores

    def _estimate_condition_nll(
        self,
        model,
        mask_id: int,
        shift_logits: bool,
        tokenizer,
        text: str,
    ) -> Tuple[float, Dict[str, Any]]:
        input_ids = tokenizer.encode(str(text), add_special_tokens=False)
        if not input_ids:
            return 0.0, {
                "valid": False,
                "reason": "empty_input",
                "target_token_count": 0,
                "covered_token_count": 0,
                "coverage_ratio": 0.0,
                "mc_passes": self.dllm_mc_num,
            }

        input_ids = input_ids[: self.max_length]
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones(len(input_ids), dtype=torch.bool)

        input_tensor = input_tensor.to(model.device)
        attention_mask = attention_mask.to(model.device)

        valid_positions = torch.arange(len(input_ids), device=model.device)
        num_tokens = int(valid_positions.numel())
        token_loss_sum = torch.zeros(num_tokens, dtype=torch.float32, device=model.device)
        token_count = torch.zeros(num_tokens, dtype=torch.float32, device=model.device)

        mask_ratio = min(max(self.dllm_mask_ratio, 0.0), 1.0)
        num_to_mask = max(1, int(round(mask_ratio * num_tokens)))
        num_to_mask = min(num_to_mask, num_tokens)

        with torch.no_grad():
            for _ in range(self.dllm_mc_num):
                local_indices = torch.randperm(num_tokens, device=model.device)[:num_to_mask]
                masked_positions = valid_positions[local_indices]

                masked_ids = input_tensor.clone()
                masked_ids[masked_positions] = mask_id

                out = model(
                    input_ids=masked_ids.unsqueeze(0),
                    attention_mask=(attention_mask.unsqueeze(0) if not shift_logits else None),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                if shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                masked_logits = logits[0, masked_positions, :]
                masked_labels = input_tensor[masked_positions]
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
                "target_token_count": num_tokens,
                "covered_token_count": 0,
                "coverage_ratio": 0.0,
                "mc_passes": self.dllm_mc_num,
            }

        avg_nll = token_loss_sum[covered_mask] / token_count[covered_mask]
        return float(avg_nll.mean().item()), {
            "valid": True,
            "reason": "ok",
            "target_token_count": num_tokens,
            "covered_token_count": covered_token_count,
            "coverage_ratio": covered_token_count / max(1, num_tokens),
            "mc_passes": self.dllm_mc_num,
            "num_masked_per_pass": num_to_mask,
            "avg_nll_mean": float(avg_nll.mean().item()),
            "avg_nll_std": float(avg_nll.std().item()) if covered_token_count > 1 else 0.0,
        }

    def _perturb_text(self, text: str) -> str:
        token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
        if len(token_ids) <= 1:
            return str(text)

        drop_count = int(round(len(token_ids) * self.perturbation_ratio))
        drop_count = min(max(drop_count, 1), len(token_ids) - 1)
        drop_indices = set(self.rng.choice(len(token_ids), size=drop_count, replace=False).tolist())
        kept_ids = [tok for idx, tok in enumerate(token_ids) if idx not in drop_indices]
        if not kept_ids:
            kept_ids = token_ids[:1]
        return self.tokenizer.decode(kept_ids, skip_special_tokens=True)

    def _summarize_distribution(self, values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return {"center": 0.0, "scale": 1.0}

        if self.score_stat == "mean_std":
            center = float(arr.mean())
            scale = float(arr.std())
        else:
            center = float(np.median(arr))
            scale = float(1.4826 * np.median(np.abs(arr - center)))

        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0

        return {
            "center": center,
            "scale": scale,
            "pool_size": int(arr.size),
            "pool_mean": float(arr.mean()),
            "pool_std": float(arr.std()),
            "pool_min": float(arr.min()),
            "pool_max": float(arr.max()),
        }

    def _calibrate_gap(self, gap: float) -> float:
        center = self.calibration_stats["center"]
        scale = self.calibration_stats["scale"]
        return float((gap - center) / scale)

    def _cleanup(self):
        if hasattr(self, "ref_model"):
            self.ref_model.to("cpu")
            del self.ref_model
        torch.cuda.empty_cache()
