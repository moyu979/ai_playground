"""第 06 课 · 反面教材: 三个新手经典 bug 的对照实验。

运行: python common_bugs.py
生成 figures/common_bugs.png

训练循环的"五步真言"少了任何一步会怎样? 口说无凭, 跑给你看:
  A. 正确版           (对照组)
  B. 忘记 zero_grad   → 梯度跨步累加 (第 5 课讲过分叉变量的 += 语义) → 爆炸
  C. 忘记 step        → 只前向不更新 → 损失只剩采样噪声
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                          "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "04_neural_network"))
from datasets import make_spirals  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def build() -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 3),
    )


def train_variant(mode: str, X, y, steps: int = 600, lr: float = 0.05,
                  seed: int = 0) -> list[float]:
    """mode: 'correct' | 'no_zero_grad' | 'no_step'"""
    torch.manual_seed(seed)
    model = build()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        idx = rng.integers(0, len(X), 32)
        xb = torch.tensor(X[idx], dtype=torch.float32)
        yb = torch.tensor(y[idx], dtype=torch.long)
        if mode != "no_zero_grad":
            opt.zero_grad()                      # ① (B 忘了这步)
        loss = F.cross_entropy(model(xb), yb)    # ②
        loss.backward()                          # ④ (③前向在上一行)
        if mode != "no_step":
            opt.step()                           # ⑤ (C 忘了这步)
        losses.append(loss.item())
    return losses


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    X, y = make_spirals(100, 3, rng=np.random.default_rng(0))

    variants = [
        ("correct", "正确版 (五步齐全)", "seagreen"),
        ("no_zero_grad", "忘记 zero_grad: 梯度累加 → 爆炸", "firebrick"),
        ("no_step", "忘记 step: 只前向不更新", "royalblue"),
    ]
    results = {}
    for mode, label, _ in variants:
        losses = train_variant(mode, X, y)
        results[mode] = losses
        print(f"{label:<28} 初始 {losses[0]:.3f} → 末值 {losses[-1]:.3f} "
              f"(最大 {max(losses):.1f})")

    print()
    print("怎么自查:")
    print("  B 的症状: loss 先降后狂飙; 打印 sum(p.grad.norm()) 会看到它逐轮增长。")
    print("  C 的症状: loss 只剩 mini-batch 的采样噪声, 曲线平坦。")
    print("  保险丝: 训练前把五步真言写成注释模板, 逐行对着敲。")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for mode, label, color in variants:
        ls = "--" if mode == "no_step" else "-"
        ax.plot(results[mode], color=color, ls=ls, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("损失 (对数轴)")
    ax.set_title("反面教材: 五步真言缺一步的代价 (同一个网络/数据/学习率)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "common_bugs.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/common_bugs.png")


if __name__ == "__main__":
    main()
