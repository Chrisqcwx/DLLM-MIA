import json
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager


class Baselineorigin2informia2Attack(AbstractAttack):
    """
    Lightweight dllm-friendly InfoRMIA reproduction.

    This implementation follows the same step-level structure as mtc5informia:
    - run 4 whole-sequence masked forwards
    - use a different mask fraction at each step
    - sample masked positions uniformly
    - compute token-level InfoRMIA only on the masked positions of that step
    - average the 4 step scores as the final membership score
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        self.batch_size = int(config.get("batch_size", 4))
        self.max_length = int(config.get("max_length", 512))
        self.seed = int(config.get("seed", 42))
        self.num_steps = int(config.get("steps", 4))
        self.min_mask_frac = float(config.get("min_mask_frac", 0.05))
        self.max_mask_frac = float(config.get("max_mask_frac", 0.50))
        self.mask_schedule = str(
            config.get("mask_schedule", config.get("l_schedule", "linear"))
        ).lower()
        self.offline_a = float(config.get("offline_a", 1.0))
        self.log_eps = float(config.get("log_eps", 1e-12))
        self.save_metadata = bool(config.get("save_metadata", False))
        self.save_token_scores = bool(config.get("save_token_scores", True))

        if self.num_steps <= 0:
            raise ValueError("steps must be positive.")

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
            raise ValueError(
                "InfoRMIA requires the dataset to contain a 'text' column."
            )

        membership_scores: List[float] = []
        for start_idx in tqdm(
            range(0, n_samples, self.batch_size), desc=f"{self.name}"
        ):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]
            batch_scores = self._compute_batch_scores(batch["text"], start_idx)
            membership_scores.extend(batch_scores)

        dataset = self._safe_add_column(dataset, self.name, membership_scores)

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
            return 0.0, {"valid": False, "reason": "empty_input"}

        input_ids = input_ids[: self.max_length]
        seq_len = len(input_ids)
        if seq_len < 1:
            return 0.0, {"valid": False, "reason": "too_short"}

        step_scores: List[float] = []
        step_metadata: List[Dict[str, Any]] = []

        for step_idx in range(self.num_steps):
            mask_frac = self._step_mask_fraction(step_idx)
            masked_positions = self._sample_mask_positions(seq_len, mask_frac)
            if not masked_positions:
                continue

            target_probs, labels = self._run_step(
                model=self.model,
                input_ids=input_ids,
                masked_positions=masked_positions,
                mask_id=self.target_mask_id,
                shift_logits=self.target_shift_logits,
            )
            ref_probs, _ = self._run_step(
                model=self.ref_model,
                input_ids=input_ids,
                masked_positions=masked_positions,
                mask_id=self.ref_mask_id,
                shift_logits=self.ref_shift_logits,
            )

            population_probs = self._estimate_population_distribution(ref_probs)
            log_ratio = torch.log(target_probs.clamp_min(self.log_eps)) - torch.log(
                population_probs.clamp_min(self.log_eps)
            )
            token_scores = log_ratio.gather(1, labels.unsqueeze(1)).squeeze(1) - (
                population_probs * log_ratio
            ).sum(dim=1)

            step_score = (
                float(token_scores.mean().item()) if token_scores.numel() > 0 else 0.0
            )
            step_scores.append(step_score)

            step_meta: Dict[str, Any] = {
                "step": step_idx,
                "mask_fraction": mask_frac,
                "num_masked": len(masked_positions),
                "masked_positions": masked_positions,
                "token_score_mean": (
                    float(token_scores.mean().item())
                    if token_scores.numel() > 0
                    else 0.0
                ),
                "token_score_std": (
                    float(token_scores.std().item())
                    if token_scores.numel() > 1
                    else 0.0
                ),
            }
            if self.save_token_scores:
                step_meta["masked_token_ids"] = labels.detach().cpu().tolist()
                step_meta["token_scores"] = token_scores.detach().cpu().tolist()
            step_metadata.append(step_meta)

        if not step_scores:
            return 0.0, {"valid": False, "reason": "no_masked_steps"}

        final_score = float(sum(step_scores) / len(step_scores))
        meta: Dict[str, Any] = {
            "valid": True,
            "steps": self.num_steps,
            "mask_schedule": self.mask_schedule,
            "step_scores": step_scores,
            "num_effective_steps": len(step_scores),
            "token_count": seq_len,
            "final_score": final_score,
        }
        if self.save_metadata:
            meta["steps_metadata"] = step_metadata
        if self.save_token_scores:
            meta["tokens"] = self.tokenizer.convert_ids_to_tokens(input_ids)
            meta["token_ids"] = list(map(int, input_ids))

        return final_score, meta

    def _step_mask_fraction(self, step_idx: int) -> float:
        if self.num_steps == 1:
            return self.max_mask_frac

        if self.mask_schedule == "geometric":
            ratio = (self.max_mask_frac / max(self.min_mask_frac, 1e-6)) ** (
                step_idx / max(self.num_steps - 1, 1)
            )
            return float(
                min(
                    self.max_mask_frac,
                    max(self.min_mask_frac, self.min_mask_frac * ratio),
                )
            )

        return float(
            self.min_mask_frac
            + (self.max_mask_frac - self.min_mask_frac)
            * (step_idx + 1)
            / (self.num_steps + 1)
        )

    def _sample_mask_positions(self, seq_len: int, mask_frac: float) -> List[int]:
        if seq_len <= 0:
            return []
        mask_count = max(1, int(round(mask_frac * seq_len)))
        mask_count = min(mask_count, seq_len)
        positions = torch.randperm(seq_len)[:mask_count]
        positions = torch.sort(positions).values
        return positions.cpu().tolist()

    def _run_step(
        self,
        model,
        input_ids: Sequence[int],
        masked_positions: Sequence[int],
        mask_id: int,
        shift_logits: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_device = self._model_device(model)
        input_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=model_device
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_tensor, dtype=torch.bool)
        positions_tensor = torch.tensor(
            masked_positions, dtype=torch.long, device=model_device
        )

        masked_input = input_tensor.clone()
        masked_input[0, positions_tensor] = mask_id

        with torch.no_grad():
            out = model(
                input_ids=masked_input,
                attention_mask=(attention_mask if not shift_logits else None),
            )
            logits = out.logits if hasattr(out, "logits") else out[0]
            if shift_logits:
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        selected_logits = logits[0, positions_tensor, :].float()
        probs = F.softmax(selected_logits, dim=-1).to(self.device)
        labels = input_tensor[0, positions_tensor].to(self.device)
        return probs, labels

    def _estimate_population_distribution(
        self, ref_probs: torch.Tensor
    ) -> torch.Tensor:
        if self.offline_a >= 1.0:
            return ref_probs

        vocab_size = ref_probs.size(-1)
        uniform = torch.full_like(ref_probs, 1.0 / float(vocab_size))
        alpha = max(0.0, min(1.0, (1.0 + self.offline_a) / 2.0))
        return alpha * ref_probs + (1.0 - alpha) * uniform

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
