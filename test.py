import torch
import torch.nn as nn


class UnembeddingSubspace:
    def __init__(self, lm_head_weight, k=128, m=4096, seed=42):
        """
        Args:
            lm_head_weight: 模型的 unembedding 矩阵, 形状为 [V, d]
            k: 子空间的维度 (top-k 奇异向量)
            m: 采样 token 的数量
        """
        self.d = lm_head_weight.shape[1]
        self.k = k

        # 1. 随机采样 m 个 token (排除特殊 token 可以通过切片实现)
        # torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        v_size = lm_head_weight.shape[0]
        # 假设前几个是特殊 token，这里简单随机采样
        indices = torch.randperm(v_size, generator=rng)[:m].to(lm_head_weight.device)
        W_s = lm_head_weight[indices]  # [m, d]

        # 2. 计算 W_s^T * W_s 的特征分解 (或直接对 W_s 做 SVD)
        # W_s = U S V^T, V 的前 k 列就是 W_s^T * W_s 的前 k 个特征向量
        _, _, V = torch.svd(W_s)
        B_out = V[:, :k]  # [d, k]

        # 存储投影矩阵 (P_out = Bout @ Bout.T)
        # 为了计算效率，我们不需要显式算 [d, d] 矩阵，用 [d, k] 两次矩阵乘法即可
        self.B_out = B_out.detach()

    def project(self, h):
        """
        计算 P_out @ h
        h 形状: [..., d]
        """
        # 利用结合律: (Bout @ Bout^T) @ h = Bout @ (Bout^T @ h)
        # 这种方式从 O(d^2) 降到 O(dk)
        return h @ self.B_out @ self.B_out.T

    def compute_access_ratio(self, h_list):
        """
        Args:
            h_list: 列表，包含多个样本在某一层经过 pooling 后的表示 [N, d]
            subspace_tool: 上面定义的 UnembeddingSubspace 实例
        Returns:
            A_l: 该层的访问比率 (标量)
        """
        # 转换为 tensor [N, d]
        h = torch.stack(h_list) if isinstance(h_list, list) else h_list

        # 1. 计算分子: E[||P_out @ h||^2]
        h_projected = self.project(h)
        # 在 d 维度求范数平方，然后在 N 维度求平均
        numerator = torch.norm(h_projected, p=2, dim=-1).pow(2)

        # 2. 计算分母: E[||h||^2]
        denominator = torch.norm(h, p=2, dim=-1).pow(2)

        # 3. 计算比例
        al = numerator / (denominator + 1e-9)  # 防止除零
        return al


# from transformers import QwenModelForCausalLM, QwenTokenizer

# 假设 model 是你的 LLM
# 1. 准备工具
lm_weight = torch.randn(12345, 4096).cuda()  # 获取 LM Head 权重
subspace = UnembeddingSubspace(lm_weight, k=64, m=1024)

# 2. 模拟一层隐藏状态 [Batch, Seq, d]
hidden_states = torch.randn(3, 5, 4096).cuda()

# 3. 按照协议 D.3 进行 Pooling (例如取最后一个 token)
# h = phi(H) -> [Batch, d]
# h_pooled = hidden_states[:, -1, :]

# 4. 计算 A_l
ratio = subspace.compute_access_ratio(hidden_states)
print(ratio)
print(ratio.shape)
# print(f"Layer Access Ratio (Al): {ratio:.4f}")
