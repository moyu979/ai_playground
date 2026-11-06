"""第 06 课 · 用 PyTorch 重写第 4 课的螺旋分类器。

运行: python mlp_pytorch.py
生成 figures/pytorch_vs_numpy.png —— 同一网络同一数据的两种实现对比。
"""

import os
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap

# 复用第 4 课的数据集 (学习: 课程内部代码也可以复用)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "04_neural_network"))
from datasets import make_spirals  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CLASS_COLORS = ListedColormap(["#aec7e8", "#ffbb78", "#98df8a"])


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Net(nn.Module):
    """与第 4 课 MLP([2, 64, 64, 3]) 一一对应:
    nn.Linear = Dense,  torch.tanh = Activation('tanh'),
    损失里的 softmax = mlp.softmax,  optimizer.step = Dense.step。
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)                      # logits; softmax 交给 CE 损失


def train(net, X, y, lr=0.5, n_steps=8000, batch_size=32, rng=None):
    """五步真言 (README §4)。返回每步损失 (numpy 数组)。"""
    rng = rng or np.random.default_rng(0)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        idx = rng.integers(0, len(X), batch_size)
        xb = torch.tensor(X[idx], dtype=torch.float32)
        yb = torch.tensor(y[idx], dtype=torch.long)
        logits = net(xb)                        # ① 前向
        loss = F.cross_entropy(logits, yb)      # ② 损失 (softmax+NLL 融合版)
        opt.zero_grad()                         # ③ 清梯度 (分叉累加语义!)
        loss.backward()                         # ④ 反传 (第 5 课的引擎)
        opt.step()                              # ⑤ 更新 (第 2 课的 SGD)
        losses.append(loss.item())
    return np.array(losses)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    dev = get_device()

    X, y = make_spirals(120, 3, rng=rng)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    torch.manual_seed(0)
    net = Net().to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"设备: {dev}, 参数量: {n_params}")

    t0 = time.perf_counter()
    with torch.device(dev):
        torch.manual_seed(0)
        net = Net().to(dev)
        losses = train(net, X, y)
    dt = time.perf_counter() - t0
    with torch.no_grad():
        acc = float((net(X_t.to(dev)).argmax(1).cpu() == y_t).float().mean())
    print(f"训练 8000 步耗时 {dt:.1f}s, 最终损失 {losses[-1]:.4f}, 准确率 {acc:.1%}")

    # ------------------------------------------- 与第 4 课 NumPy 版对比
    npy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "04_neural_network", "figures", "spiral_loss.pkl")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(losses, alpha=0.35, color="crimson")
    w = 100
    ax.plot(np.convolve(losses, np.ones(w) / w, "valid"), color="crimson",
            label=f"PyTorch (本课, {dt:.1f}s)")
    if os.path.exists(npy_path):
        with open(npy_path, "rb") as f:
            losses_np = pickle.load(f)
        ax.plot(np.convolve(losses_np, np.ones(w) / w, "valid"),
                color="royalblue", alpha=0.8, label="NumPy 手写 (第 4 课)")
    else:
        print(f"(提示: 没找到 {npy_path}, 先跑第 4 课的 train_toy.py 可得对比曲线)")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("损失 (滑动平均)")
    ax.set_title("同一个网络: 手写 NumPy vs PyTorch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pad = 0.6
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        zz = net(grid.to(dev)).argmax(1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["#aec7e8", "#ffbb78", "#98df8a"], alpha=0.55)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CLASS_COLORS, s=8,
               edgecolors="k", linewidths=0.2)
    ax.set_title(f"PyTorch 版最终决策边界 (acc {acc:.1%})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pytorch_vs_numpy.png"), dpi=130)
    plt.close(fig)

    print(f"\n图已保存到 {FIG_DIR}/pytorch_vs_numpy.png")
    print("两条损失曲线应基本重合 —— 数学是同一个, 只是引擎更快。")


if __name__ == "__main__":
    main()
