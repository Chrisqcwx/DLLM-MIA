import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import login
from datasets import Dataset
import json
import os
from datetime import datetime

from attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params

# from attack.run import init_model
from attack.misc.models import ModelManager


class MinkppAttack(AbstractAttack):
    """
    SAMA (Subset-Aggregated Membership Attack) for diffusion language models.

    Detects training membership by comparing target vs reference model losses across
    progressive masking configurations with robust subset-based aggregation.

    Args:
        texts: List of text strings to evaluate for membership.
        target_model: Fine-tuned diffusion model to test.
        ref_model: Pre-trained reference model for calibration.
        num_steps: Number of progressive masking steps (default: 16).
        min_mask_frac: Starting mask fraction (default: 0.05).
        max_mask_frac: Ending mask fraction (default: 0.50).
        num_subsets: Random subsets to sample per step (default: 128).
        subset_size: Tokens per subset (default: 10).

    Returns:
        List[float]: Membership scores in [0,1], higher indicates member.

    Algorithm:
        1. Progressively mask tokens from min_mask_frac to max_mask_frac
        2. At each step, sample num_subsets random groups of subset_size tokens
        3. Compare if ref_loss > target_loss for each subset (binary test)
        4. Aggregate with inverse-step weighting (early steps weighted more)
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # --- Core params ---
        self.num_steps = int(config.get("steps", 4))
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))
        self.mink = float(config.get("mink", 0.8))

        # v6 local-signal params (unchanged)
        self.subset_size = int(
            config.get("subset_size", 8)
        )  # l for the local random subsets
        self.num_subsets = int(config.get("num_subsets", 128))  # N subsets per step
        self.seed = int(config.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)

        # l_s grows monotonically from ~min_mask_frac*L up to ~max_mask_frac*L
        # keeping early steps cleaner and later steps broader.
        self.min_mask_frac = float(config.get("min_mask_frac", 0.05))
        self.max_mask_frac = float(config.get("max_mask_frac", 0.50))
        self.mask_schedule = config.get(
            "l_schedule", "linear"
        )  # "linear" or "geometric"

        # METADATA SAVING
        self.save_metadata = config.get("save_metadata", True)
        self.metadata_dir = config.get("metadata_dir") or os.environ.get(
            "SAMA_METADATA_DIR", "./"
        )
        self.metadata_dir = os.path.join(
            self.metadata_dir,
            f"sama_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        if self.save_metadata:
            os.makedirs(self.metadata_dir, exist_ok=True)
            # Save config for reproducibility
            with open(os.path.join(self.metadata_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self.metadata_buffer = []

        # Target model mask behavior
        if 'model_mask_id' in config and 'model_shift_logits' in config:
            self.target_mask_id = config['model_mask_id']
            self.target_shift_logits = config['model_shift_logits']
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(
                self.model
            )

        hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token)

        # Seed for reproducible masking
        torch.manual_seed(self.seed)

    # ------------------------------ Public API ------------------------------

    def run(self, dataset: Dataset) -> Dataset:
        n_samples = len(dataset)
        if n_samples == 0:
            return dataset

        membership_scores = []
        for start_idx in tqdm(
            range(0, n_samples, self.batch_size), desc=f"{self.name}"
        ):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = dataset[start_idx:end_idx]
            batch_scores = self._compute_batch_scores(batch["text"], start_idx)
            membership_scores.extend(batch_scores)

        dataset = dataset.add_column(self.name, membership_scores)

        # Save accumulated metadata
        if self.save_metadata and self.metadata_buffer:
            metadata_path = os.path.join(self.metadata_dir, "full_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata_buffer, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

        self._cleanup()
        return dataset

    # --------------------------- Internals -----------------------------

    def _compute_batch_scores(self, texts, batch_start_idx):
        B = len(texts)
        batch_metadata = []

        # Tokenize
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        seq_len = input_ids.size(1)

        # Reference copies

        # Position-aware loss buffers
        cumulative_target_losses = torch.zeros(
            B, seq_len, dtype=torch.float32, device=self.device
        )

        # Cumulative mask across steps (v8 uses fixed-l cardinality per step)

        # Per-sample step scores
        step_scores = [[] for _ in range(B)]

        # Precompute valid lengths per sample
        valid_lengths = attention_mask.sum(dim=1)  # (B,)

        # Initialize metadata for each sample
        for b in range(B):
            batch_metadata.append(
                {
                    "sample_idx": batch_start_idx + b,
                    "text": texts[b][:100],  # First 100 chars for reference
                    "valid_length": int(valid_lengths[b].item()),
                    "steps": [],
                }
            )

        for step in range(self.num_steps):
            step_metadata = []

            new_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # on target device

            # Linear schedule by default: l_s ~ [min_frac * L, max_frac * L] across steps
            # new_mask = torch.zeros_like(new_mask)
            for b in range(B):
                Lb = int(valid_lengths[b].item())
                if Lb == 0:
                    continue

                frac = torch.rand(1).clamp(0.01, 0.99).item()

                # mask 个数
                desired_total = max(1, int(round(frac * Lb)))
                # 占位的个数
                current_total = int((new_mask[b] & attention_mask[b]).sum().item())
                to_add = max(0, desired_total - current_total)
                if to_add == 0:
                    continue

                # Sample new positions uniformly without replacement among currently unmasked valid tokens
                unmasked_valid = (~new_mask[b]) & attention_mask[b]
                candidates = torch.where(unmasked_valid)[0]
                if candidates.numel() == 0:
                    continue
                if to_add > candidates.numel():
                    to_add = int(candidates.numel())

                perm = torch.randperm(candidates.numel(), device=self.device)
                chosen = candidates[perm[:to_add]]
                new_mask[b, chosen] = True

            if not new_mask.any():
                # Nothing to add this step
                continue

            # ---- Target model: compute CE over current masked context; store for *newly* masked positions
            masked_ids_target = input_ids.clone()
            masked_ids_target[new_mask] = self.target_mask_id

            with torch.no_grad():
                out = self.model(
                    input_ids=masked_ids_target,
                    attention_mask=(
                        attention_mask if not self.target_shift_logits else None
                    ),
                )
                logits = out.logits if hasattr(out, 'logits') else out[0]
                if self.target_shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

                # minkpp standardization
                logp = torch.log_softmax(logits, dim=-1)
                mu_logp = logp.mean(dim=-1)
                std_logp = logp.std(dim=-1)
                label_logp = logp.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(
                    -1
                )
                standard_logp = (label_logp - mu_logp) / (std_logp + 1e-8)

                # ce = (
                #     F.cross_entropy(
                #         logits.view(-1, logits.size(-1)),
                #         input_ids.view(-1),
                #         reduction='none',
                #     )
                #     .view(B, seq_len)
                #     .float()
                # )

                cumulative_target_losses[new_mask] = -standard_logp[new_mask].float()

            for b in range(B):
                masked_positions = torch.where(new_mask[b])[0]
                m = int(masked_positions.numel())
                if m == 0:
                    continue

                t_losses = (
                    cumulative_target_losses[b][masked_positions].detach().cpu().numpy()
                )

                score, subset_details = self._subset_binary_comparison_with_metadata(
                    t_losses,
                    None,
                    subset_size=min(self.subset_size, m),
                    num_subsets=self.num_subsets,
                )
                step_scores[b].append(score)

                # Store step metadata
                if self.save_metadata:
                    batch_metadata[b]["steps"].append(
                        {
                            "step": step,
                            "num_masked": m,
                            "mask_fraction": m / int(valid_lengths[b].item()),
                            "score": score,
                            "masked_positions": masked_positions.cpu().tolist(),
                            "target_losses_mean": float(t_losses.mean()),
                            "target_losses_std": float(t_losses.std()),
                            "subset_comparisons": subset_details,
                        }
                    )

        # Aggregate across steps with inverse weighting (unchanged)
        batch_scores = []
        for b in range(B):
            if len(step_scores[b]) == 0:
                batch_scores.append(0.0)
            else:
                # weights = 1.0 / (np.arange(len(step_scores[b])) + 1)
                # weights = weights / weights.sum()
                # final_score = float(np.average(step_scores[b], weights=weights))
                final_score = float(np.mean(step_scores[b]))
                batch_scores.append(final_score)

                if self.save_metadata:
                    batch_metadata[b]["final_score"] = final_score
                    batch_metadata[b]["step_scores"] = step_scores[b]
                    # batch_metadata[b]["weights"] = weights.tolist()

        # Add to global metadata buffer
        if self.save_metadata:
            self.metadata_buffer.extend(batch_metadata)

        return batch_scores

    def _subset_binary_comparison_with_metadata(
        self,
        target_losses: np.ndarray,
        ref_losses: np.ndarray,
        subset_size: int,
        num_subsets: int,
    ):
        # mink 选取最低的logp， 而ce loss是-log p 故选取最高的k个
        # maxk loss
        topk_num = int(len(target_losses) * self.mink)
        topk_indices = np.argpartition(target_losses, -topk_num)[-topk_num:]
        topk_losses = target_losses[topk_indices]

        return float(-topk_losses.sum()), []

    def _subset_binary_comparison(
        self,
        target_losses: np.ndarray,
        ref_losses: np.ndarray,
        subset_size: int,
        num_subsets: int,
    ) -> float:
        """Original method without metadata for backward compatibility"""
        score, _ = self._subset_binary_comparison_with_metadata(
            target_losses, ref_losses, subset_size, num_subsets
        )
        return score

    def _cleanup(self):
        if hasattr(self, 'ref_model'):
            self.ref_model.to('cpu')
            del self.ref_model
            del self.ref_tokenizer
        torch.cuda.empty_cache()
