import json
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import login
from tqdm import tqdm

from attack.attacks import AbstractAttack
from attack.attacks.utils import get_model_nll_params
from attack.misc.models import ModelManager

ENABLE_DIAGNOSTICS = True
DIAGNOSTICS_DIRNAME = "diagnostics_mtc5_arp_r64_mlp_baseline"
DIAGNOSTICS_PROPOSAL_MAX_SAMPLES = None
DIAGNOSTICS_STABILITY_MAX_SAMPLES = 500
DIAGNOSTICS_SUBSET_MAX_SAMPLES = None
DIAGNOSTICS_TOP_K = 32
DIAGNOSTICS_FLUSH_EVERY = 2048
DIAGNOSTIC_PROPOSAL_NAME = "q_alpha"
PROPOSAL_GATE_TOP_FRAC = 0.20
PROPOSAL_ALPHA = math.log(1.2)
PROPOSAL_U_CLIP_MAX = 1.0
ASYM_POS_QUANTILE = 0.95
ASYM_NEG_QUANTILE = 0.50
ROBUST_EPS = 1e-6
SUBSET_POOL_BETA = 8.0


class Mtc5aprAttack(AbstractAttack):
    """
    MTC5-ARP subset-aggregated membership attack for diffusion language models.

    The baseline proposal distribution samples tokens proportionally to the
    reference model's full-mask token cross entropy.
    """

    def __init__(self, name: str, model, tokenizer, config, device: torch.device):
        super().__init__(name, model, tokenizer, config, device)

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

        self.save_metadata = config.get("save_metadata", True)
        self.metadata_dir = (
            config.get("metadata_dir")
            or os.environ.get("MTC5APR_METADATA_DIR")
            or os.environ.get("SAMA_METADATA_DIR", "./")
        )
        self.metadata_dir = os.path.join(
            self.metadata_dir,
            f"mtc5apr_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        if self.save_metadata:
            os.makedirs(self.metadata_dir, exist_ok=True)
            with open(os.path.join(self.metadata_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, default=str)
        self.metadata_buffer = []

        self.enable_diagnostics = ENABLE_DIAGNOSTICS
        self.diagnostics_proposal_max_samples = DIAGNOSTICS_PROPOSAL_MAX_SAMPLES
        self.diagnostics_stability_max_samples = DIAGNOSTICS_STABILITY_MAX_SAMPLES
        self.diagnostics_subset_max_samples = DIAGNOSTICS_SUBSET_MAX_SAMPLES
        self.diagnostics_top_k = DIAGNOSTICS_TOP_K
        self.diagnostics_flush_every = DIAGNOSTICS_FLUSH_EVERY
        self.proposal_name = config.get("proposal_name", DIAGNOSTIC_PROPOSAL_NAME)
        self.proposal_gate_top_frac = float(
            config.get("proposal_gate_top_frac", PROPOSAL_GATE_TOP_FRAC)
        )
        self.proposal_alpha = float(config.get("proposal_alpha", PROPOSAL_ALPHA))
        self.proposal_u_clip_max = float(
            config.get("proposal_u_clip_max", PROPOSAL_U_CLIP_MAX)
        )
        self.asym_pos_quantile = float(
            config.get("asym_pos_quantile", ASYM_POS_QUANTILE)
        )
        self.asym_neg_quantile = float(
            config.get("asym_neg_quantile", ASYM_NEG_QUANTILE)
        )
        self.robust_eps = float(config.get("robust_eps", ROBUST_EPS))
        self.subset_pool_beta = float(config.get("subset_pool_beta", SUBSET_POOL_BETA))
        safe_attack_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in self.name
        )
        self.diagnostics_dir = os.path.abspath(
            os.path.join("attack_results", DIAGNOSTICS_DIRNAME, safe_attack_name)
        )
        self.token_proposal_path = os.path.join(
            self.diagnostics_dir, "token_proposal.jsonl"
        )
        self.token_stability_path = os.path.join(
            self.diagnostics_dir, "token_stability.jsonl"
        )
        self.subset_delta_path = os.path.join(
            self.diagnostics_dir, "subset_delta.jsonl"
        )
        self.token_proposal_buffer = []
        self.token_stability_buffer = []
        self.subset_delta_buffer = []
        if self.enable_diagnostics:
            os.makedirs(self.diagnostics_dir, exist_ok=True)
            diagnostics_config = {
                "enabled": self.enable_diagnostics,
                "proposal_max_samples": self.diagnostics_proposal_max_samples,
                "stability_max_samples": self.diagnostics_stability_max_samples,
                "subset_max_samples": self.diagnostics_subset_max_samples,
                "top_k": self.diagnostics_top_k,
                "flush_every": self.diagnostics_flush_every,
                "proposal_name": self.proposal_name,
                "proposal_gate_top_frac": self.proposal_gate_top_frac,
                "proposal_alpha": self.proposal_alpha,
                "proposal_u_clip_max": self.proposal_u_clip_max,
                "asym_pos_quantile": self.asym_pos_quantile,
                "asym_neg_quantile": self.asym_neg_quantile,
                "robust_eps": self.robust_eps,
                "subset_pool_mode": "softmax",
                "subset_pool_beta": self.subset_pool_beta,
                "attack_name": name,
                "timestamp": datetime.now().isoformat(),
            }
            with open(
                os.path.join(self.diagnostics_dir, "diagnostics_config.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(diagnostics_config, f, indent=2)

        if "model_mask_id" in config and "model_shift_logits" in config:
            self.target_mask_id = config["model_mask_id"]
            self.target_shift_logits = config["model_shift_logits"]
        else:
            self.target_mask_id, self.target_shift_logits = get_model_nll_params(
                self.model
            )

        self.ref_device = torch.device(config.get("reference_device", "cuda"))
        ref_model_path = config.get("reference_model_path")
        if not ref_model_path:
            raise ValueError("reference_model_path must be specified")

        hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)

        self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
            ref_model_path, ref_model_path, self.ref_device
        )
        self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

        self.rng2 = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)

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

        if hasattr(dataset, "info") and hasattr(dataset.info, "__dict__"):
            dataset.info.__dict__.pop("task_templates", None)
        dataset = dataset.add_column(self.name, membership_scores)

        if self.save_metadata and self.metadata_buffer:
            metadata_path = os.path.join(self.metadata_dir, "full_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata_buffer, f, indent=2)
            print(f"Metadata saved to {metadata_path}")

        self._cleanup()
        return dataset

    def _within_sample_limit(self, sample_id: int, max_samples) -> bool:
        return max_samples is None or sample_id < max_samples

    def _should_collect_proposal(self, sample_id: int) -> bool:
        return self.enable_diagnostics and self._within_sample_limit(
            sample_id, self.diagnostics_proposal_max_samples
        )

    def _should_collect_stability(self, sample_id: int) -> bool:
        return self.enable_diagnostics and self._within_sample_limit(
            sample_id, self.diagnostics_stability_max_samples
        )

    def _should_collect_subset(self, sample_id: int) -> bool:
        return self.enable_diagnostics and self._within_sample_limit(
            sample_id, self.diagnostics_subset_max_samples
        )

    def _append_jsonl_rows(self, path, rows):
        if not rows:
            return
        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _buffer_diagnostic_rows(self, buffer_name, path, rows):
        if not rows:
            return
        buffer = getattr(self, buffer_name)
        buffer.extend(rows)
        if len(buffer) >= self.diagnostics_flush_every:
            self._append_jsonl_rows(path, buffer)
            buffer.clear()

    def _flush_diagnostics(self):
        if not self.enable_diagnostics:
            return
        self._append_jsonl_rows(self.token_proposal_path, self.token_proposal_buffer)
        self.token_proposal_buffer.clear()
        self._append_jsonl_rows(self.token_stability_path, self.token_stability_buffer)
        self.token_stability_buffer.clear()
        self._append_jsonl_rows(self.subset_delta_path, self.subset_delta_buffer)
        self.subset_delta_buffer.clear()

    def _full_mask_token_ce(
        self,
        model,
        input_ids,
        attention_mask,
        mask_id,
        shift_logits,
        device,
    ):
        masked_input_ids = torch.ones_like(input_ids) * mask_id
        with torch.no_grad():
            out = model(
                input_ids=masked_input_ids,
                attention_mask=(attention_mask if not shift_logits else None),
            )
            logits = out.logits if hasattr(out, "logits") else out[0]
            if shift_logits:
                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
            ce = (
                F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    input_ids.view(-1),
                    reduction="none",
                )
                .view(input_ids.size(0), input_ids.size(1))
                .float()
            )
        ce = ce * attention_mask.float().to(device)
        return ce

    def _compute_token_weights(
        self,
        input_ids,
        attention_mask,
        input_ids_ref,
        attention_mask_ref,
        batch_start_idx,
    ):
        input_ids_ref = input_ids_ref.to(self.ref_device)
        attention_mask_ref = attention_mask_ref.to(self.ref_device)
        ref_ce = self._full_mask_token_ce(
            self.ref_model,
            input_ids_ref,
            attention_mask_ref,
            self.ref_mask_id,
            self.ref_shift_logits,
            self.ref_device,
        )

        ref_ce = ref_ce + 1e-8
        weights = ref_ce / ref_ce.sum(dim=1, keepdim=True)
        weights_target_device = weights.to(self.device)
        corrected_weights = weights_target_device.clone()

        tracked_positions = [
            torch.empty(0, dtype=torch.long, device=self.device)
            for _ in range(input_ids.size(0))
        ]

        target_ce = self._full_mask_token_ce(
            self.model,
            input_ids,
            attention_mask,
            self.target_mask_id,
            self.target_shift_logits,
            self.device,
        )

        proposal_rows = []
        for b in range(input_ids.size(0)):
            sample_id = batch_start_idx + b
            collect_proposal = self._should_collect_proposal(sample_id)
            collect_stability = self._should_collect_stability(sample_id)

            valid_positions = torch.where(attention_mask[b])[0]
            if valid_positions.numel() == 0:
                continue

            valid_positions_ref = valid_positions.to(self.ref_device)
            ref_vals = ref_ce[b][valid_positions_ref].detach().cpu()
            target_vals = target_ce[b][valid_positions].detach().cpu()
            q0_vals = weights_target_device[b][valid_positions].detach().cpu()
            token_ids = input_ids[b][valid_positions].detach().cpu()
            gap_vals = ref_vals - target_vals
            median_ref = torch.median(ref_vals)
            scale_vals = ref_vals + median_ref
            u_vals = torch.clamp(
                gap_vals / scale_vals, min=0.0, max=self.proposal_u_clip_max
            )

            gate_count = max(
                1, int(math.ceil(self.proposal_gate_top_frac * valid_positions.numel()))
            )
            gate_indices = torch.topk(q0_vals, k=gate_count).indices
            gate_vals = torch.zeros_like(q0_vals)
            gate_vals[gate_indices] = 1.0
            z_vals = gate_vals * u_vals
            q_alpha_unnorm = q0_vals * torch.exp(self.proposal_alpha * z_vals)
            q_alpha_vals = q_alpha_unnorm / q_alpha_unnorm.sum()
            corrected_weights[b, valid_positions] = q_alpha_vals.to(self.device)

            if collect_stability:
                k = min(self.diagnostics_top_k, valid_positions.numel())
                top_q0_local = torch.topk(q0_vals, k=k).indices
                top_gap_local = torch.topk(gap_vals, k=k).indices
                tracked_positions[b] = torch.unique(
                    torch.cat(
                        [valid_positions[top_q0_local], valid_positions[top_gap_local]]
                    )
                )

            if collect_proposal:
                for idx in range(valid_positions.numel()):
                    proposal_rows.append(
                        {
                            "sample_id": int(sample_id),
                            "token_pos": int(valid_positions[idx].item()),
                            "token_id": int(token_ids[idx].item()),
                            "ref_ce": float(ref_vals[idx].item()),
                            "target_ce": float(target_vals[idx].item()),
                            "gap": float(gap_vals[idx].item()),
                            "q0": float(q0_vals[idx].item()),
                            "q_alpha": float(q_alpha_vals[idx].item()),
                            "gate": float(gate_vals[idx].item()),
                            "u": float(u_vals[idx].item()),
                            "z": float(z_vals[idx].item()),
                            "valid_mask": True,
                        }
                    )

        self._buffer_diagnostic_rows(
            "token_proposal_buffer", self.token_proposal_path, proposal_rows
        )
        return corrected_weights, tracked_positions

    def _compute_batch_scores(self, texts, batch_start_idx):
        batch_size = len(texts)
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

        cumulative_target_losses = torch.zeros(
            batch_size, seq_len, dtype=torch.float32, device=self.device
        )
        cumulative_ref_losses = torch.zeros(
            batch_size, seq_len, dtype=torch.float32, device=self.ref_device
        )
        step_scores = [[] for _ in range(batch_size)]
        valid_lengths = attention_mask.sum(dim=1)

        token_weights, tracked_positions = self._compute_token_weights(
            input_ids,
            attention_mask,
            input_ids_ref,
            attention_mask_ref,
            batch_start_idx,
        )

        for b in range(batch_size):
            batch_metadata.append(
                {
                    "sample_idx": batch_start_idx + b,
                    "text": texts[b][:100],
                    "valid_length": int(valid_lengths[b].item()),
                    "steps": [],
                }
            )

        for step in range(self.num_steps):
            new_mask = torch.zeros_like(input_ids, dtype=torch.bool)

            for b in range(batch_size):
                length_b = int(valid_lengths[b].item())
                if length_b == 0:
                    continue

                if self.mask_schedule == "geometric":
                    ratio = (self.max_mask_frac / max(self.min_mask_frac, 1e-6)) ** (
                        step / max(self.num_steps - 1, 1)
                    )
                    frac = min(
                        self.max_mask_frac,
                        max(self.min_mask_frac, self.min_mask_frac * ratio),
                    )
                else:
                    frac = self.min_mask_frac + (
                        self.max_mask_frac - self.min_mask_frac
                    ) * (step + 1) / (self.num_steps + 1)

                desired_total = max(1, int(round(frac * length_b)))
                to_add = desired_total
                if to_add == 0:
                    continue

                candidates = torch.where(attention_mask[b])[0]
                if candidates.numel() == 0:
                    continue
                if to_add > candidates.numel():
                    to_add = int(candidates.numel())

                candidate_weights = token_weights[b][candidates]
                candidate_weights = candidate_weights / candidate_weights.sum()
                chosen_idx = self.rng.choice(
                    len(candidates),
                    size=to_add,
                    replace=False,
                    p=candidate_weights.detach().cpu().numpy(),
                )
                chosen_idx = torch.from_numpy(chosen_idx).to(candidates.device)
                chosen = candidates[chosen_idx]
                new_mask[b, chosen] = True

            if not new_mask.any():
                continue

            cumulative_mask = new_mask
            cumulative_mask_ref = cumulative_mask.to(self.ref_device)

            masked_ids_target = input_ids.clone()
            masked_ids_target[cumulative_mask] = self.target_mask_id

            with torch.no_grad():
                out = self.model(
                    input_ids=masked_ids_target,
                    attention_mask=(
                        attention_mask if not self.target_shift_logits else None
                    ),
                )
                logits = out.logits if hasattr(out, "logits") else out[0]
                if self.target_shift_logits:
                    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
                target_ce_ctx = (
                    F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        input_ids.view(-1),
                        reduction="none",
                    )
                    .view(batch_size, seq_len)
                    .float()
                )
                cumulative_target_losses[new_mask] = target_ce_ctx[new_mask]

            masked_ids_ref = input_ids_ref.clone()
            masked_ids_ref[cumulative_mask_ref] = self.ref_mask_id

            with torch.no_grad():
                out_ref = self.ref_model(
                    input_ids=masked_ids_ref,
                    attention_mask=(
                        attention_mask_ref if not self.ref_shift_logits else None
                    ),
                )
                logits_ref = (
                    out_ref.logits if hasattr(out_ref, "logits") else out_ref[0]
                )
                if self.ref_shift_logits:
                    logits_ref = torch.cat(
                        [logits_ref[:, :1, :], logits_ref[:, :-1, :]], dim=1
                    )
                ref_ce_ctx = (
                    F.cross_entropy(
                        logits_ref.view(-1, logits_ref.size(-1)),
                        input_ids_ref.view(-1),
                        reduction="none",
                    )
                    .view(batch_size, seq_len)
                    .float()
                )
                new_mask_ref = new_mask.to(self.ref_device)
                cumulative_ref_losses[new_mask_ref] = ref_ce_ctx[new_mask_ref]

            if self.enable_diagnostics:
                stability_rows = []
                for b in range(batch_size):
                    sample_id = batch_start_idx + b
                    if not self._should_collect_stability(sample_id):
                        continue
                    tracked = tracked_positions[b]
                    if tracked.numel() == 0:
                        continue
                    tracked_ref = tracked.to(self.ref_device)
                    mask_fraction = int(new_mask[b].sum().item()) / max(
                        int(valid_lengths[b].item()), 1
                    )
                    context_id = f"{sample_id}:{step}"
                    target_vals = target_ce_ctx[b][tracked].detach().cpu()
                    ref_vals = ref_ce_ctx[b][tracked_ref].detach().cpu()
                    gap_vals = ref_vals - target_vals
                    for idx in range(tracked.numel()):
                        stability_rows.append(
                            {
                                "sample_id": int(sample_id),
                                "step": int(step),
                                "context_id": context_id,
                                "token_pos": int(tracked[idx].item()),
                                "ref_ce_ctx": float(ref_vals[idx].item()),
                                "target_ce_ctx": float(target_vals[idx].item()),
                                "gap_ctx": float(gap_vals[idx].item()),
                                "mask_fraction": float(mask_fraction),
                            }
                        )
                self._buffer_diagnostic_rows(
                    "token_stability_buffer",
                    self.token_stability_path,
                    stability_rows,
                )

            for b in range(batch_size):
                masked_positions = torch.where(new_mask[b])[0]
                num_masked = int(masked_positions.numel())
                if num_masked == 0:
                    continue
                masked_positions_ref = masked_positions.to(self.ref_device)

                target_losses = (
                    cumulative_target_losses[b][masked_positions].detach().cpu().numpy()
                )
                ref_losses = (
                    cumulative_ref_losses[b][masked_positions_ref]
                    .detach()
                    .cpu()
                    .numpy()
                )

                score, subset_details = self._subset_binary_comparison_with_metadata(
                    target_losses,
                    ref_losses,
                    sample_id=batch_start_idx + b,
                    step=step,
                    proposal_name=self.proposal_name,
                    subset_size=min(self.subset_size, num_masked),
                    num_subsets=self.num_subsets,
                )
                step_scores[b].append(score)

                if self.save_metadata:
                    batch_metadata[b]["steps"].append(
                        {
                            "step": step,
                            "num_masked": num_masked,
                            "mask_fraction": num_masked / int(valid_lengths[b].item()),
                            "score": score,
                            "masked_positions": masked_positions.cpu().tolist(),
                            "target_losses_mean": float(target_losses.mean()),
                            "target_losses_std": float(target_losses.std()),
                            "ref_losses_mean": float(ref_losses.mean()),
                            "ref_losses_std": float(ref_losses.std()),
                            "loss_diff_mean": float(
                                (ref_losses - target_losses).mean()
                            ),
                            "subset_comparisons": subset_details,
                        }
                    )

        batch_scores = []
        for b in range(batch_size):
            if len(step_scores[b]) == 0:
                batch_scores.append(0.0)
            else:
                final_score = float(np.mean(step_scores[b]))
                batch_scores.append(final_score)
                if self.save_metadata:
                    batch_metadata[b]["final_score"] = final_score
                    batch_metadata[b]["step_scores"] = step_scores[b]

        if self.save_metadata:
            self.metadata_buffer.extend(batch_metadata)

        return batch_scores

    def _subset_binary_comparison_with_metadata(
        self,
        target_losses: np.ndarray,
        ref_losses: np.ndarray,
        sample_id: int,
        step: int,
        proposal_name: str,
        subset_size: int,
        num_subsets: int,
    ):
        num_losses = min(len(target_losses), len(ref_losses))
        if num_losses == 0:
            return 0.0, []
        subset_size = min(subset_size, num_losses)
        if subset_size <= 0:
            return float(ref_losses.sum() > target_losses.sum()), []

        subset_details = []
        idx_matrix = np.vstack(
            [
                self.rng.choice(num_losses, size=subset_size, replace=False)
                for _ in range(num_subsets)
            ]
        ).astype(np.int64)

        target_sums = target_losses[idx_matrix].sum(axis=1)
        ref_sums = ref_losses[idx_matrix].sum(axis=1)
        deltas = ref_losses - target_losses

        pos_vals = deltas[deltas > 0]
        neg_abs_vals = np.abs(deltas[deltas < 0])

        if pos_vals.size > 0:
            clip_threshold_pos = float(np.quantile(pos_vals, self.asym_pos_quantile))
        else:
            clip_threshold_pos = 0.0
        if neg_abs_vals.size > 0:
            clip_threshold_neg = float(
                np.quantile(neg_abs_vals, self.asym_neg_quantile)
            )
        else:
            clip_threshold_neg = 0.0

        clipped_deltas = deltas.copy()
        if clip_threshold_pos > 0.0:
            pos_mask = clipped_deltas > 0
            clipped_deltas[pos_mask] = np.minimum(
                clipped_deltas[pos_mask], clip_threshold_pos
            )
        if clip_threshold_neg > 0.0:
            neg_mask = clipped_deltas < 0
            clipped_deltas[neg_mask] = -np.minimum(
                np.abs(clipped_deltas[neg_mask]), clip_threshold_neg
            )

        # 截断
        clipped_selected = clipped_deltas[idx_matrix]
        asym_robust_sums = clipped_selected.sum(axis=1)
        clipped_energy = np.square(clipped_selected).sum(axis=1)
        z_scores = asym_robust_sums / np.sqrt(clipped_energy + self.robust_eps)
        phi_scores = 0.5 * (1.0 + np.vectorize(math.erf)(z_scores / math.sqrt(2.0)))
        score_center = float(phi_scores.mean())
        softmax_logits = self.subset_pool_beta * (phi_scores - score_center)
        softmax_logits = softmax_logits - np.max(softmax_logits)
        softmax_weights = np.exp(softmax_logits)
        softmax_weights = softmax_weights / softmax_weights.sum()
        pooled_score = float(np.sum(softmax_weights * phi_scores))
        comparisons = phi_scores > 0.5

        if self._should_collect_subset(sample_id):
            subset_rows = []
            for subset_id in range(num_subsets):
                subset_rows.append(
                    {
                        "sample_id": int(sample_id),
                        "step": int(step),
                        "subset_id": int(subset_id),
                        "proposal_name": proposal_name,
                        "target_sum": float(target_sums[subset_id]),
                        "ref_sum": float(ref_sums[subset_id]),
                        "delta": float(ref_sums[subset_id] - target_sums[subset_id]),
                        "asym_robust_sum": float(asym_robust_sums[subset_id]),
                        "z_score": float(z_scores[subset_id]),
                        "phi_score": float(phi_scores[subset_id]),
                        "clip_threshold_pos": float(clip_threshold_pos),
                        "clip_threshold_neg": float(clip_threshold_neg),
                        "clipped_energy": float(clipped_energy[subset_id]),
                        "subset_pool_weight": float(softmax_weights[subset_id]),
                        "win": bool(comparisons[subset_id]),
                    }
                )
            self._buffer_diagnostic_rows(
                "subset_delta_buffer", self.subset_delta_path, subset_rows
            )

        if self.save_metadata:
            for i in range(min(10, num_subsets)):
                subset_details.append(
                    {
                        "subset_idx": i,
                        "positions": idx_matrix[i].tolist(),
                        "target_sum": float(target_sums[i]),
                        "ref_sum": float(ref_sums[i]),
                        "delta": float(ref_sums[i] - target_sums[i]),
                        "asym_robust_sum": float(asym_robust_sums[i]),
                        "z_score": float(z_scores[i]),
                        "phi_score": float(phi_scores[i]),
                        "clip_threshold_pos": float(clip_threshold_pos),
                        "clip_threshold_neg": float(clip_threshold_neg),
                        "subset_pool_weight": float(softmax_weights[i]),
                        "ref_wins": bool(comparisons[i]),
                    }
                )

        return pooled_score, subset_details

    def _subset_binary_comparison(
        self,
        target_losses: np.ndarray,
        ref_losses: np.ndarray,
        subset_size: int,
        num_subsets: int,
    ) -> float:
        score, _ = self._subset_binary_comparison_with_metadata(
            target_losses,
            ref_losses,
            sample_id=-1,
            step=-1,
            proposal_name=self.proposal_name,
            subset_size=subset_size,
            num_subsets=num_subsets,
        )
        return score

    def _cleanup(self):
        self._flush_diagnostics()
        if hasattr(self, "ref_model"):
            self.ref_model.to("cpu")
            del self.ref_model
            del self.ref_tokenizer
        torch.cuda.empty_cache()


# Mtc5Attack = Mtc5aprAttack
