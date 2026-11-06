"""第 09 课 · torch 版因果自注意力 (第 10 课 GPT 的直接积木)。

运行: python self_attention_torch.py
包含三个自检: 形状 / 因果性 (改未来不改过去) / 与官方算子对拍。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """多头因果自注意力, 手写实现 (第 10 课 Block 内的正是这个模块)。

    形状流水线 (README §5):
      x (B,T,d) → qkv (B,T,3d) → 拆成 3 份 → 按 head 变形 (B,H,T,d/H)
      → att = softmax(q kᵀ/√d_head + 掩码) (B,H,T,T)
      → y = att @ v (B,H,T,d/H) → 拼回 (B,T,d) → 输出投影
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 head 数整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)   # 一次投影出 q,k,v (高效)
        self.proj = nn.Linear(d_model, d_model)      # 输出混合投影 W_O

    def forward(self, x: torch.Tensor):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)        # 各 (B, T, d)

        # 变形出多头: (B, T, d) → (B, H, T, d/H)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)   # (B,H,T,T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        att = att.masked_fill(~mask, float("-inf"))   # 因果: 上三角 -inf
        att = F.softmax(att, dim=-1)

        y = att @ v                                   # (B, H, T, d/H)
        y = y.transpose(1, 2).reshape(B, T, d)        # 拼回多头
        return self.proj(y), att                      # proj 是 W_O


def check_shapes() -> None:
    torch.manual_seed(0)
    m = CausalSelfAttention(d_model=32, n_heads=4)
    x = torch.randn(2, 5, 32)
    y, att = m(x)
    assert y.shape == (2, 5, 32), y.shape
    assert att.shape == (2, 4, 5, 5), att.shape
    assert torch.allclose(att.sum(-1), torch.ones(2, 4, 5), atol=1e-5)
    assert (att[:, :, torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)] == 0).all()
    print("[1/3] 形状 ✓  (B,T,d)→(B,T,d), 注意力 (B,H,T,T), 行和=1, 上三角=0")


def check_causality() -> None:
    """GPT 的命门测试: 改动未来 token, 过去的输出必须分毫不变。"""
    torch.manual_seed(0)
    m = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        y1, _ = m(x)
        x2 = x.clone()
        x2[:, -1] += 100.0                       # 篡改最后一个 (未来) token
        y2, _ = m(x2)
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-6), "因果性被破坏!"
    print("[2/3] 因果性 ✓  (篡改 t=5, t<5 的输出完全不变 —— 模型无法作弊)")


def check_against_official() -> None:
    """与 PyTorch 官方融合算子 scaled_dot_product_attention 对拍。"""
    torch.manual_seed(0)
    m = CausalSelfAttention(d_model=32, n_heads=4).eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        y_ours, _ = m(x)

        B, T, d = x.shape
        q, k, v = m.qkv(x).split(d, dim=2)
        q = q.view(B, T, m.n_heads, m.d_head).transpose(1, 2)
        k = k.view(B, T, m.n_heads, m.d_head).transpose(1, 2)
        v = v.view(B, T, m.n_heads, m.d_head).transpose(1, 2)
        y_ref = F.scaled_dot_product_attention(
            q, k, v, is_causal=True)             # 官方算子 (内部=FlashAttention)
        y_ref = y_ref.transpose(1, 2).reshape(B, T, d)
        y_ref = m.proj(y_ref)
    assert torch.allclose(y_ours, y_ref, atol=1e-5)
    print("[3/3] 官方算子对拍 ✓  (手写版 ≡ scaled_dot_product_attention)")


def main() -> None:
    check_shapes()
    check_causality()
    check_against_official()
    print("\n全部自检通过 —— 这个模块将被原样搬进第 10 课的 GPT。")
    print("练习: python exercises.py (split/merge heads + 饱和实验)")


if __name__ == "__main__":
    main()
