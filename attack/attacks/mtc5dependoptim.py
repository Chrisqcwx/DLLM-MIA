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

import torch

import torch


import torch


import torch


def sample_engine(
    attn_weights, n_samples, k_list, alpha=1.0, lambda_penalty=2.0, iterations=50
):
    """
    基于压缩感知凸松弛思想的优化采样引擎 (修正版)
    """
    L = attn_weights.shape[0]
    device = attn_weights.device

    # 1. 对称化处理
    adj = (attn_weights + attn_weights.t()).float()

    # 2. 对角线清零：我们只关心 Token 间的“互信息”，不关心“自信息”
    # 这样 w^T A w 就只计算不同 Token 之间的累积 PCMI
    adj.fill_diagonal_(0)

    # 3. 归一化（可选，有助于学习率稳定）
    if adj.max() > 0:
        adj = adj / adj.max()

    global_counts = torch.zeros(L, device=device)
    all_results = []

    for i in range(n_samples):
        target_k = k_list[i]

        # 初始化权重：在 target_k/L 附近加入噪声
        w = torch.full((L,), target_k / L, device=device, requires_grad=True)
        # 增加随机扰动，防止优化陷入完全对称的局部点
        # with torch.no_grad():
        #     w += torch.randn_like(w) * 0.01

        optimizer = torch.optim.Adam([w], lr=0.001)

        for _ in range(iterations):
            # --- 优化目标 ---
            # 内部连通性 (Quadratic Form): 衡量采样集合内的独立程度
            internal_pcmi = torch.dot(w, torch.mv(adj, w))

            # 跨采样多样性 (Linear Penalty): 衡量与历史采样的重叠度
            cross_diversity = torch.dot(w, global_counts)

            loss = (alpha * internal_pcmi) + (lambda_penalty * cross_diversity)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- 投影算子 (Projection) ---
            with torch.no_grad():
                w.clamp_(0, 1)
                current_sum = w.sum()
                if current_sum > 1e-6:
                    w.mul_(target_k / current_sum)

        # 4. 离散化输出
        indices = torch.topk(w, k=target_k).indices

        # 更新全局状态，累积已选中的频次
        global_counts[indices] += 1.0
        all_results.append(indices.detach().clone())

    return all_results


from transformers.modeling_outputs import CausalLMOutput


class DummyModel(torch.nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.dummy_param.device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        dummy_logits = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.vocab_size
        ).to(input_ids.device)
        return CausalLMOutput(logits=dummy_logits)


# MTC2 + mask for token weight
class Mtc5dependoptimAttack(AbstractAttack):
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
        self.temperature = float(config.get("temperature", 1.0))

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

        self.sample_alpha = float(config.get("sample_alpha", 1.0))
        self.lambda_penalty = float(config.get("lambda_penalty", 1.0))
        self.weight_start_layer_ratio = float(
            config.get("weight_start_layer_ratio", 0.0)
        )
        self.weight_end_layer_ratio = float(config.get("weight_end_layer_ratio", 1.0))

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

        # Load reference (diffusion LM) used for comparison
        ref_model_path = config.get("reference_model_path")
        # if not ref_model_path:
        #     raise ValueError("DF-MIA requires 'reference_model_path' in the config.")
        self.ref_device = torch.device(config.get("reference_device", str(device)))
        if ref_model_path:
            self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
                ref_model_path, ref_model_path, self.ref_device
            )
            self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(
                self.ref_model
            )
        else:
            self.ref_model = DummyModel(tokenizer.vocab_size).to(self.ref_device)
            self.ref_tokenizer = self.tokenizer
            self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.model)

        # hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN')
        # if hf_token:
        #     login(token=hf_token)

        # self.ref_model, self.ref_tokenizer, _ = ModelManager.init_model(
        #     ref_model_path, ref_model_path, self.ref_device
        # )
        # self.ref_mask_id, self.ref_shift_logits = get_model_nll_params(self.ref_model)

        # Seed for reproducible masking
        self.rng2 = torch.Generator().manual_seed(self.seed)
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
    def _compute_weights(self, input_ids_ref, attention_mask_ref):
        """
        用 reference 模型在原始（未 mask）输入上的 token-level loss 作为位置权重。
        高 loss 位置 → 高信息量 → 优先 mask。
        """
        B, L = input_ids_ref.shape
        real_L = attention_mask_ref.sum(dim=1)
        all_attns = []

        # use_
        with torch.no_grad():
            out = self.ref_model(
                input_ids=input_ids_ref,
                attention_mask=(
                    attention_mask_ref if not self.ref_shift_logits else None
                ),
                output_attentions=True,
            )
            logits = out.logits if hasattr(out, 'logits') else out[0]
            attentions = out.attentions
            assert (
                attentions is not None
            ), "Reference model must output attentions for token weighting"
            for b in range(B):
                b_attns = []
                num_layers = len(attentions)
                start_layer = int(self.weight_start_layer_ratio * num_layers)
                end_layer = int(self.weight_end_layer_ratio * num_layers)
                if end_layer <= start_layer:
                    end_layer = start_layer + 1
                for layer_weight in attentions[start_layer:end_layer]:
                    # layer_weight: (B, num_heads, L, L)
                    # 取该样本的有效部分，平均头部和层数
                    valid_attn = layer_weight[
                        b, :, : real_L[b], : real_L[b]
                    ]  # (num_heads, Lb, Lb)
                    b_attns.append(valid_attn.mean(dim=0))  # (Lb, Lb)
                all_attns.append(
                    torch.stack(b_attns, dim=0).mean(dim=0).cpu()
                )  # (Lb, Lb)
            # attentions = torch.stack([a.mean(dim=1) for a in attentions], dim=0).mean(
            #     dim=0
            # )  # (num_layers, B, num_heads, L, L) -> (B, L, L)
        return all_attns
        #     if self.ref_shift_logits:
        #         logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

        #     ce = (
        #         F.cross_entropy(
        #             logits.view(-1, logits.size(-1)),
        #             input_ids_ref.view(-1),
        #             reduction='none',
        #         )
        #         .view(B, L)
        #         .float()
        #     )  # (B, L)

        # # padding 位置权重置 0，剩余位置用 softmax 归一化
        # ce = ce * attention_mask_ref.float().to(self.ref_device)
        # # 避免全零行
        # ce = ce + 1e-8
        # weights = ce / ce.sum(dim=1, keepdim=True)  # (B, L), 归一化为概率
        # return weights  # 返回 CPU 上的权重供后续采样

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
        input_ids_ref = input_ids.clone().to(self.ref_device)
        attention_mask_ref = attention_mask.clone().to(self.ref_device)

        # Position-aware loss buffers
        cumulative_target_losses = torch.zeros(
            B, seq_len, dtype=torch.float32, device=self.device
        )
        cumulative_ref_losses = torch.zeros(
            B, seq_len, dtype=torch.float32, device=self.ref_device
        )

        # Cumulative mask across steps (v8 uses fixed-l cardinality per step)

        # Per-sample step scores
        step_scores = [[] for _ in range(B)]

        # Precompute valid lengths per sample
        valid_lengths = attention_mask.sum(dim=1)  # (B,)

        weight_matrixs = self._compute_weights(
            input_ids_ref, attention_mask_ref
        )  # (B, L)
        all_samples = []

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

            k_list = []

            for step in range(self.num_steps):
                # for b in range(B):
                Lb = int(valid_lengths[b].item())
                if Lb == 0:
                    continue

                if self.mask_schedule == "geometric":
                    # Geometric spacing between min and max fractions
                    r = (self.max_mask_frac / max(self.min_mask_frac, 1e-6)) ** (
                        step / max(self.num_steps - 1, 1)
                    )
                    frac = min(
                        self.max_mask_frac,
                        max(self.min_mask_frac, self.min_mask_frac * r),
                    )
                else:
                    # Linear spacing
                    frac = self.min_mask_frac + (
                        self.max_mask_frac - self.min_mask_frac
                    ) * (step + 1) / (self.num_steps + 1)

                # mask 个数
                desired_total = max(1, int(round(frac * Lb)))
                k_list.append(desired_total)

            b_samples = sample_engine(
                weight_matrixs[b],
                self.num_steps,
                k_list,
                alpha=self.sample_alpha,
                lambda_penalty=self.lambda_penalty,
            )
            all_samples.append(b_samples)

        for step in range(self.num_steps):
            step_metadata = []

            new_mask = torch.zeros_like(input_ids, dtype=torch.bool)  # on target device
            for b in range(B):
                chosen = all_samples[b][step].to(new_mask.device)
                new_mask[b, chosen] = True

            if not new_mask.any():
                # Nothing to add this step
                continue

            cumulative_mask = new_mask

            cumulative_mask_ref = cumulative_mask.to(self.ref_device)

            # ---- Target model: compute CE over current masked context; store for *newly* masked positions
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
                logits = logits / self.temperature
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

            # ---- Reference model (diffusion LM)
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
                logits_r = logits_r / self.temperature
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
                            "ref_losses_mean": float(r_losses.mean()),
                            "ref_losses_std": float(r_losses.std()),
                            "loss_diff_mean": float((r_losses - t_losses).mean()),
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
        m = min(len(target_losses), len(ref_losses))
        if m == 0:
            return 0.0, []
        s = min(subset_size, m)
        if s <= 0:
            return float(ref_losses.sum() > target_losses.sum()), []

        subset_details = []
        idx_matrix = np.vstack(
            [self.rng.choice(m, size=s, replace=False) for _ in range(num_subsets)]
        ).astype(np.int64)

        t_sel = target_losses[idx_matrix].sum(axis=1)
        r_sel = ref_losses[idx_matrix].sum(axis=1)
        comparisons = r_sel > t_sel

        if self.save_metadata:
            # Store sample of subset comparisons (first 10 to avoid huge files)
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
