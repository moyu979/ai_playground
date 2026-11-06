"""第 10 课练习 (毕业考)。实现后运行 `python exercises.py` 自动测试。"""

import numpy as np
import torch
import torch.nn.functional as F


# 1. top-p (nucleus) 采样 ---------------------------------------------------------
def top_p_sample(logits: np.ndarray, p: float, rng: np.random.Generator) -> int:
    """nucleus 采样: 保留累积概率达到 p 的最小前缀集合, 重归一化后采样。

    logits: (V,) 未归一化分数。
    步骤: softmax → 降序排序 → 累积概率首次 ≥ p 处截断 → 集合内归一化 → 采样。
    p→0 应退化为 argmax; p=1 等价普通采样。
    """
    raise NotImplementedError


# 2. 手写 GELU ---------------------------------------------------------------------
def gelu(x: np.ndarray) -> np.ndarray:
    """tanh 近似版 GELU (Transformer 论文脚注 1 的公式):

    GELU(x) ≈ 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))
    测试会与 torch 的精确版对拍 (容差 1e-2, 近似公式本来就差这点)。
    """
    raise NotImplementedError


# 3. 带预热的余弦学习率 --------------------------------------------------------------
def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """warmup + cosine decay 调度器 (大模型训练标配, nanoGPT 同款)。

    规则:
    - step < warmup:  lr = base_lr * (step+1) / warmup   (线性升温, (step+1) 防止除零)
    - warmup ≤ step < total: lr = base_lr * 0.5 * (1 + cos(π * (step-warmup)/(total-warmup)))
    - step ≥ total: lr = base_lr * 0.05                 (尾段保持 5%)
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    # 1 top-p
    try:
        logits = np.array([5.0, 4.0, 1.0, 0.0, -1.0])   # 前两个约占 ~0.9
        picks = [top_p_sample(logits, 0.9, rng) for _ in range(3000)]
        ok = set(picks) <= {0, 1}                        # 长尾被砍掉
        picks_greedy = [top_p_sample(logits, 1e-9, rng) for _ in range(50)]
        ok = ok and set(picks_greedy) == {0}             # p→0 贪心
    except NotImplementedError:
        ok = False
    passed.append(("top_p_sample", ok))

    # 2 GELU
    try:
        x = np.linspace(-3, 3, 101)
        err = np.abs(gelu(x) - F.gelu(torch.tensor(x)).numpy()).max()
        ok = err < 1e-2
        print(f"    GELU 与 torch 精确版最大误差 {err:.2e}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("gelu", ok))

    # 3 学习率调度
    try:
        base, warmup, total = 1e-3, 100, 1000
        ok = (
            abs(lr_at(0, base, warmup, total) - base / 100) < 1e-12
            and abs(lr_at(99, base, warmup, total) - base) < 1e-9      # 预热顶点
            and abs(lr_at(warmup, base, warmup, total) - base) < 1e-9  # 余弦起点
            and abs(lr_at((warmup + total) // 2, base, warmup, total)
                    - base * 0.5) < 1e-6                              # 中点减半
            and abs(lr_at(total + 50, base, warmup, total) - base * 0.05) < 1e-12
        )
    except NotImplementedError:
        ok = False
    passed.append(("lr_at (warmup + cosine)", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 🎓 你已经从零走到了 Transformer。")
        print(" 接下来: 把模型调大过夜训练 / 换中文语料 / 去读 nanoGPT 源码。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
