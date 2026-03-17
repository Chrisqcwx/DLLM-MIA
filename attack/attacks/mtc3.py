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

from attack.misc.models import ModelManager


class Mtc3Attack(AbstractAttack):
    """
    MTC3: MTC with two optional improvements over the base MTC attack.

    Improvement 1 — Info-weighted masking (info_weighted_mask=True):
        Instead of sampling mask positions uniformly, positions are sampled
        proportional to the reference model's per-token loss on the unmasked
        input. Tokens with higher reference loss carry more information and
        yield a stronger membership signal when masked.
        Cost: one extra reference forward pass per batch.

    Improvement 2 — Self-calibration (self_calibrate=True):
        The raw score (fraction of "ref wins" subsets) is calibrated against
        a null distribution estimated from perturbed versions of the same text.
        Perturbations are known non-members (sentence-shuffled or window-sliced),
        so the calibrated score is a z-score: how many standard deviations above
        the null the query sits. This improves TPR at low FPR without requiring
        any labeled data.
        Cost: n_cal extra scoring passes per query text.

    Config keys (all optional, with defaults):
        steps              (int)   Number of progressive masking steps. Default: 4.
        batch_size         (int)   Batch size for dataset iteration. Default: 8.
        max_length         (int)   Token truncation length. Default: 512.
        subset_size        (int)   Tokens per subset (l). Default: 8.
        num_subsets        (int)   Number of subsets sampled per step (N). Default: 128.
        seed               (int)   RNG seed. Default: 42.
        min_mask_frac      (float) Starting mask fraction. Default: 0.05.
        max_mask_frac      (float) Ending mask fraction. Default: 0.50.
        l_schedule         (str)   "linear" or "geometric". Default: "linear".
        info_weighted_mask (bool)  Enable info-weighted masking. Default: False.
        self_calibrate     (bool)  Enable self-calibration. Default: False.
        n_cal              (int)   Number of perturbations per query. Default: 8.
        cal_strategy       (str)   "window", "shuffle", or "both". Default: "both".
        save_metadata      (bool)  Save per-step debug metadata. Default: True.
        metadata_dir       (str)   Directory for metadata output.
        reference_model_path (str) Required. Path to pre-trained diffusion reference model.
        reference_device   (str)   Device for reference model. Default: "cuda".
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

        # --- Core params ---
        self.num_steps = int(config.get("steps", 4))
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 512))

        self.subset_size = int(config.get("subset_size", 8))
        self.num_subsets = int(config.get("num_subsets", 128))
        self.seed = int(config.get("seed", 42))
        self.rng = np.random.default_rng(self.seed)

        self.min_mask_frac = float(config.get("min_mask_frac", 0.05))
        self.max_mask_frac = float(config.get("max_mask_frac", 0.50))
        self.mask_schedule = config.get("l_schedule", "linear")

        # --- Improvement switches ---
        self.info_weighted_mask = bool(config.get("info_weighted_mask", False))
        self.self_calibrate = bool(config.get("self_calibrate", False))
        self.n_cal = int(config.get("n_cal", 8))
        self.cal_strategy = config.get(
            "cal_strategy", "both"
        )  # "window"/"shuffle"/"both"

        # --- Metadata saving ---
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
            with open(os.path.join(self.metadata_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self.metadata_buffer = []

        # --- Target model mask params ---
        if 'model_mask_id' in config and 'model_shift_logits' in config:
            self.target_mask_id = config['model_mask_id']
            self.target_shift_logits = config['model_shift_logits']
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(
                self.model
            )

        # --- Reference model ---
        self.ref_device = torch.device(config.get("reference_device", "cuda"))
        ref_model_path = config.get('reference_model_path')
        if not ref_model_path:
            raise ValueError("reference_model_path must be specified")

        hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
        if hf_token:
            login(token=hf_token)

        self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
            ref_model_path, ref_model_path, self.ref_device
        )
        self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

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

        if self.save_metadata and self.metadata_buffer:
            metadata_path = os.path.join(self.metadata_dir, "full_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata_buffer, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

        self._cleanup()
        return dataset

    # --------------------------- Internals -----------------------------

    def _compute_token_weights(
        self,
        input_ids_ref: torch.Tensor,
        attention_mask_ref: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute per-token sampling weights using reference model loss on unmasked input.

        A single no-mask forward pass through the reference model yields per-token
        cross-entropy. Higher loss → harder to predict from context → more informative
        for membership detection → higher sampling weight.

        Returns:
            weights: float32 numpy array of shape (B, L), rows sum to 1,
                     padding positions have weight 0.
        """
        B, L = input_ids_ref.shape
        with torch.no_grad():
            out = self.ref_model(
                input_ids=input_ids_ref,
                attention_mask=(
                    attention_mask_ref if not self.ref_shift_logits else None
                ),
            )
            logits = out.logits if hasattr(out, 'logits') else out[0]
            if self.ref_shift_logits:
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

            ce = (
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    input_ids_ref.view(-1),
                    reduction='none',
                )
                .view(B, L)
                .float()
            )

        # Zero out padding, then normalize each row to a probability distribution.
        ce = ce * attention_mask_ref.float()
        ce = ce + 1e-8  # avoid all-zero rows for very short sequences
        row_sums = ce.sum(dim=1, keepdim=True)
        weights = (ce / row_sums).cpu().numpy()  # (B, L)
        return weights

    def _compute_raw_scores(self, texts: list, batch_start_idx: int) -> tuple:
        """
        Core scoring logic (no self-calibration).

        Returns:
            raw_scores: list of float, one per text.
            batch_metadata: list of dict, one per text (for save_metadata).
        """
        B = len(texts)
        batch_metadata = []

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

        input_ids_ref = input_ids.clone().to(self.ref_device)
        attention_mask_ref = attention_mask.clone().to(self.ref_device)

        # --- Improvement 1: compute token weights once per batch ---
        if self.info_weighted_mask:
            token_weights = self._compute_token_weights(
                input_ids_ref, attention_mask_ref
            )
            # token_weights: (B, L) numpy float32, rows normalized
        else:
            token_weights = None

        cumulative_target_losses = torch.zeros(
            B, seq_len, dtype=torch.float32, device=self.device
        )
        cumulative_ref_losses = torch.zeros(
            B, seq_len, dtype=torch.float32, device=self.ref_device
        )

        step_scores = [[] for _ in range(B)]
        valid_lengths = attention_mask.sum(dim=1)  # (B,)

        for b in range(B):
            batch_metadata.append(
                {
                    "sample_idx": batch_start_idx + b,
                    "text": texts[b][:100],
                    "valid_length": int(valid_lengths[b].item()),
                    "steps": [],
                }
            )

        for step in range(
            self.num_steps
        ):  # cumulative_mask lives outside the step loop (unlike the mtc copy.py bug)
            cumulative_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            new_mask = torch.zeros_like(cumulative_mask)

            for b in range(B):
                Lb = int(valid_lengths[b].item())
                if Lb == 0:
                    continue

                if self.mask_schedule == "geometric":
                    r = (self.max_mask_frac / max(self.min_mask_frac, 1e-6)) ** (
                        step / max(self.num_steps - 1, 1)
                    )
                    frac = min(
                        self.max_mask_frac,
                        max(self.min_mask_frac, self.min_mask_frac * r),
                    )
                else:
                    frac = self.min_mask_frac + (
                        self.max_mask_frac - self.min_mask_frac
                    ) * (step + 1) / (self.num_steps + 1)

                desired_total = max(1, int(round(frac * Lb)))
                current_total = int(
                    (cumulative_mask[b] & attention_mask[b]).sum().item()
                )
                to_add = max(0, desired_total - current_total)
                if to_add == 0:
                    continue

                unmasked_valid = (~cumulative_mask[b]) & attention_mask[b]
                candidates = torch.where(unmasked_valid)[0]
                if candidates.numel() == 0:
                    continue
                if to_add > candidates.numel():
                    to_add = int(candidates.numel())

                if token_weights is not None:
                    # --- Weighted sampling: use reference model loss as probability ---
                    cand_idx = candidates.cpu().numpy()
                    w = token_weights[b][cand_idx]
                    w_sum = w.sum()
                    if w_sum > 0:
                        w = w / w_sum
                    else:
                        w = np.ones(len(cand_idx)) / len(cand_idx)
                    chosen_local = self.rng.choice(
                        len(cand_idx), size=to_add, replace=False, p=w
                    )
                    chosen = candidates[torch.from_numpy(chosen_local)]
                else:
                    # --- Uniform sampling (original behaviour) ---
                    perm = torch.randperm(candidates.numel(), device=self.device)
                    chosen = candidates[perm[:to_add]]

                new_mask[b, chosen] = True

            if not new_mask.any():
                continue

            cumulative_mask = cumulative_mask | new_mask
            cumulative_mask_ref = cumulative_mask.to(self.ref_device)

            # ---- Target model forward ----
            masked_ids_target = input_ids.clone()
            masked_ids_target[cumulative_mask] = self.target_mask_id

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

                ce = (
                    F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        input_ids.view(-1),
                        reduction='none',
                    )
                    .view(B, seq_len)
                    .float()
                )
                cumulative_target_losses[new_mask] = ce[new_mask]

            # ---- Reference model forward ----
            masked_ids_ref = input_ids_ref.clone()
            masked_ids_ref[cumulative_mask_ref] = self.ref_mask_id

            with torch.no_grad():
                out_r = self.ref_model(
                    input_ids=masked_ids_ref,
                    attention_mask=(
                        attention_mask_ref if not self.ref_shift_logits else None
                    ),
                )
                logits_r = out_r.logits if hasattr(out_r, 'logits') else out_r[0]
                if self.ref_shift_logits:
                    logits_r = torch.cat(
                        [logits_r[:, :1, :], logits_r[:, :-1, :]], dim=1
                    )

                ce_r = (
                    F.cross_entropy(
                        logits_r.view(-1, logits_r.size(-1)),
                        input_ids_ref.view(-1),
                        reduction='none',
                    )
                    .view(B, seq_len)
                    .float()
                )
                new_mask_ref = new_mask.to(self.ref_device)
                cumulative_ref_losses[new_mask_ref] = ce_r[new_mask_ref]

            # ---- Score computation for newly masked positions ----
            for b in range(B):
                masked_positions = torch.where(new_mask[b])[0]
                m = int(masked_positions.numel())
                if m == 0:
                    continue

                t_losses = (
                    cumulative_target_losses[b][masked_positions].detach().cpu().numpy()
                )
                r_losses = (
                    cumulative_ref_losses[b][masked_positions].detach().cpu().numpy()
                )

                score, subset_details = self._subset_binary_comparison_with_metadata(
                    t_losses,
                    r_losses,
                    subset_size=min(self.subset_size, m),
                    num_subsets=self.num_subsets,
                )
                step_scores[b].append(score)

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
                            "ref_losses_mean": float(r_losses.mean()),
                            "ref_losses_std": float(r_losses.std()),
                            "loss_diff_mean": float((r_losses - t_losses).mean()),
                            "subset_comparisons": subset_details,
                        }
                    )

        raw_scores = []
        for b in range(B):
            if len(step_scores[b]) == 0:
                raw_scores.append(0.0)
            else:
                final_score = float(np.mean(step_scores[b]))
                raw_scores.append(final_score)

                if self.save_metadata:
                    batch_metadata[b]["raw_score"] = final_score
                    batch_metadata[b]["step_scores"] = step_scores[b]

        return raw_scores, batch_metadata

    def _compute_batch_scores(self, texts: list, batch_start_idx: int) -> list:
        """
        Compute final membership scores with optional self-calibration.

        When self_calibrate=False this is identical in behaviour to the base MTC.
        When self_calibrate=True, each raw score is z-scored against a local
        null distribution estimated from n_cal perturbed variants of the text.
        """
        raw_scores, batch_metadata = self._compute_raw_scores(texts, batch_start_idx)

        if not self.self_calibrate:
            if self.save_metadata:
                for b, meta in enumerate(batch_metadata):
                    meta["final_score"] = raw_scores[b]
                    meta["calibrated"] = False
                self.metadata_buffer.extend(batch_metadata)
            return raw_scores

        # --- Improvement 2: self-calibration ---
        final_scores = []
        for b, (text, raw_score) in enumerate(zip(texts, raw_scores)):
            cal_texts = self._generate_calibration_texts(text)
            if not cal_texts:
                final_scores.append(raw_score)
                if self.save_metadata:
                    batch_metadata[b]["final_score"] = raw_score
                    batch_metadata[b]["calibrated"] = False
                continue

            # Score perturbations using the same pipeline (no recursion:
            # self.self_calibrate is not consulted inside _compute_raw_scores)
            cal_raw_scores, _ = self._compute_raw_scores(cal_texts, batch_start_idx=-1)

            mu = float(np.mean(cal_raw_scores))
            sigma = float(np.std(cal_raw_scores)) + 1e-8
            calibrated = (raw_score - mu) / sigma

            final_scores.append(calibrated)

            if self.save_metadata:
                batch_metadata[b]["final_score"] = calibrated
                batch_metadata[b]["raw_score"] = raw_score
                batch_metadata[b]["cal_null_mean"] = mu
                batch_metadata[b]["cal_null_std"] = sigma
                batch_metadata[b]["cal_scores"] = cal_raw_scores
                batch_metadata[b]["calibrated"] = True

        if self.save_metadata:
            self.metadata_buffer.extend(batch_metadata)

        return final_scores

    def _generate_calibration_texts(self, text: str) -> list:
        """
        Generate perturbed variants of *text* to serve as a per-query null
        distribution. Two strategies are available:

        "window"  — decode random contiguous token sub-sequences. The content
                    is a proper subset of the original so it is unlikely to
                    appear verbatim in training data.
        "shuffle" — split on sentence boundaries and randomly reorder
                    sentences. The vocabulary is identical but the ordering
                    breaks any memorized sequence.
        "both"    — n_cal/2 from each strategy (default).

        Returns a list of at most n_cal strings (may be shorter for very short
        texts).
        """
        tokens = self.tokenizer.encode(
            text, truncation=True, max_length=self.max_length
        )
        L = len(tokens)
        cal_texts = []

        n_window = (
            self.n_cal // 2
            if self.cal_strategy == "both"
            else (self.n_cal if self.cal_strategy == "window" else 0)
        )
        n_shuffle = self.n_cal - n_window if self.cal_strategy != "window" else 0

        # --- Window strategy ---
        if n_window > 0 and L >= 8:
            # Keep 70-90% of the sequence, starting at a random offset.
            for _ in range(n_window):
                keep = int(L * (0.70 + 0.20 * self.rng.random()))
                start = int(self.rng.integers(0, max(1, L - keep)))
                sub = tokens[start : start + keep]
                cal_texts.append(self.tokenizer.decode(sub, skip_special_tokens=True))

        # --- Shuffle strategy ---
        if n_shuffle > 0:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) >= 3:
                for _ in range(n_shuffle):
                    shuffled = sentences.copy()
                    self.rng.shuffle(shuffled)
                    cal_texts.append('. '.join(shuffled) + '.')
            elif L >= 8:
                # Fall back to window when text has no usable sentence breaks.
                for _ in range(n_shuffle):
                    keep = int(L * (0.70 + 0.20 * self.rng.random()))
                    start = int(self.rng.integers(0, max(1, L - keep)))
                    sub = tokens[start : start + keep]
                    cal_texts.append(
                        self.tokenizer.decode(sub, skip_special_tokens=True)
                    )

        return cal_texts[: self.n_cal]

    # --------------------------- Subset scoring ----------------------------

    def _subset_binary_comparison_with_metadata(
        self,
        target_losses: np.ndarray,
        ref_losses: np.ndarray,
        subset_size: int,
        num_subsets: int,
    ):
        m = min(len(target_losses), len(ref_losses))
        if m == 0:
            return 0.0, []
        s = min(subset_size, m)
        if s <= 0:
            return float(ref_losses.sum() > target_losses.sum()), []

        idx_matrix = np.vstack(
            [self.rng.choice(m, size=s, replace=False) for _ in range(num_subsets)]
        ).astype(np.int64)

        t_sel = target_losses[idx_matrix].sum(axis=1)
        r_sel = ref_losses[idx_matrix].sum(axis=1)
        comparisons = r_sel > t_sel

        subset_details = []
        if self.save_metadata:
            for i in range(min(10, num_subsets)):
                subset_details.append(
                    {
                        "subset_idx": i,
                        "positions": idx_matrix[i].tolist(),
                        "target_sum": float(t_sel[i]),
                        "ref_sum": float(r_sel[i]),
                        "ref_wins": bool(comparisons[i]),
                    }
                )

        return float(comparisons.mean()), subset_details

    def _cleanup(self):
        if hasattr(self, 'ref_model'):
            self.ref_model.to('cpu')
            del self.ref_model
            del self.ref_tokenizer
        torch.cuda.empty_cache()
