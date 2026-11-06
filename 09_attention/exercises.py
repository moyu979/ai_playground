"""第 09 课练习。实现后运行 `python exercises.py` 自动测试。"""

import numpy as np
import torch


# 1. 多头形状变换 -----------------------------------------------------------------
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B, T, d) → (B, n_heads, T, d/n_heads)。

    关键: view 先把 d 拆成 (n_heads, d_head), 再 transpose 把 head 维提前。
    """
    raise NotImplementedError


def merge_heads(y: torch.Tensor) -> torch.Tensor:
    """(B, n_heads, T, d_head) → (B, T, n_heads*d_head)。split_heads 的逆操作。"""
    raise NotImplementedError


# 2. 带掩码的注意力权重 -------------------------------------------------------------
def attention_weights(q: torch.Tensor, k: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    """单头注意力权重矩阵 (softmax(QKᵀ/√d_k) + 可选掩码)。

    q: (Tq, d), k: (Tk, d) → 返回 (Tq, Tk)。
    mask: (Tq, Tk) 布尔阵, True = 允许关注 (False 的位置权重必须恰好为 0)。
    提示: 用 masked_fill(~mask, -inf) 后再 softmax。
    """
    raise NotImplementedError


# 3. 饱和实验 -----------------------------------------------------------------------
def saturation_experiment(d: int, n: int = 500, seed: int = 0) -> dict:
    """q, k ~ N(0, 1), 各 n 个 (n×d)。

    返回 {"raw": 不缩放时每行最大权重的均值, "scaled": 除以 √d 后的均值}。
    理论 (README §3): d 大时不缩放 → 分数方差 d → softmax 饱和 → max→1。
    测试标准: d=64 时 raw 应显著大于 scaled。
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    passed = []

    # 1 split/merge
    try:
        x = torch.randn(2, 5, 12)
        y = split_heads(x, 3)
        ok = y.shape == (2, 3, 5, 4)
        if ok:
            # head h 必须对应特征区间 [h*4, (h+1)*4)
            assert torch.allclose(y[:, 1], x[:, :, 4:8])
            back = merge_heads(y)
            ok = back.shape == x.shape and torch.allclose(back, x)
    except NotImplementedError:
        ok = False
    passed.append(("split_heads / merge_heads", ok))

    # 2 带掩码注意力
    try:
        T, d = 5, 8
        q = torch.randn(T, d)
        k = torch.randn(T, d)
        w = attention_weights(q, k)
        ok = w.shape == (T, T) and torch.allclose(w.sum(-1), torch.ones(T), atol=1e-5)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        w2 = attention_weights(q, k, mask=causal)
        upper = w2[torch.triu(torch.ones(T, T, dtype=torch.bool), 1)]
        ok = ok and (upper == 0).all() and torch.allclose(w2.sum(-1), torch.ones(T), atol=1e-5)
        # 与参考实现对拍
        ref = torch.softmax(q @ k.T / d**0.5, -1)
        ok = ok and torch.allclose(w, ref, atol=1e-6)
    except NotImplementedError:
        ok = False
    passed.append(("attention_weights (含掩码)", ok))

    # 3 饱和实验
    try:
        res = saturation_experiment(64)
        ok = (res["raw"] > 0.5 and res["scaled"] < 0.15
              and res["raw"] > 5 * res["scaled"])
        print(f"    d=64: 不缩放 max={res['raw']:.3f} | ÷√d max={res['scaled']:.3f}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("saturation_experiment", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 进入 10_transformer/ —— 终点站。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
