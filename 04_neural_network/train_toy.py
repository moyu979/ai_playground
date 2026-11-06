"""第 04 课 · 训练纯 NumPy MLP。

运行: python train_toy.py
生成 figures/datasets.png, figures/spiral_training.png, figures/moons_training.png
并把螺旋训练的损失曲线存成 spiral_loss.npy (第 6 课用来和 PyTorch 对比)。
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from datasets import make_circles, make_moons, make_spirals
from mlp import Dense, MLP

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CLASS_COLORS = ListedColormap(["#aec7e8", "#ffbb78", "#98df8a"])


def get_dense_state(net: MLP):
    return [(l.W.copy(), l.b.copy()) for l in net.layers if isinstance(l, Dense)]


def set_dense_state(net: MLP, state) -> None:
    denses = [l for l in net.layers if isinstance(l, Dense)]
    for d, (W, b) in zip(denses, state):
        d.W = W.copy()
        d.b = b.copy()


def train(net, X, y, lr=0.5, n_steps=8000, batch_size=32, rng=None,
          snapshot_steps=()):
    """mini-batch SGD (第 2 课的算法, 原封不动)。返回每步损失 + 快照。"""
    rng = rng or np.random.default_rng(0)
    losses = []
    snapshots = {}
    for step in range(n_steps + 1):
        if step in snapshot_steps:
            snapshots[step] = get_dense_state(net)
        idx = rng.integers(0, len(X), batch_size)      # 有放回抽样, 简洁够用
        loss, _ = net.loss_and_grad(X[idx], y[idx])
        net.step(lr)
        losses.append(loss)
    return np.array(losses), snapshots


def plot_boundary(ax, net, X, y, title):
    """在网络当前权重下画决策区域 + 数据点。"""
    pad = 0.6
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    zz = net.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["#aec7e8", "#ffbb78", "#98df8a"], alpha=0.55)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CLASS_COLORS, s=8, edgecolors="k",
               linewidths=0.2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def run_dataset(name, X, y, n_classes, rng, n_steps=8000):
    net = MLP([2, 64, 64, n_classes], rng=rng, activation="tanh", init="xavier")
    snaps_wanted = {0, n_steps // 3, 2 * n_steps // 3, n_steps}
    losses, snaps = train(net, X, y, lr=0.5, n_steps=n_steps,
                          batch_size=32, rng=rng, snapshot_steps=snaps_wanted)
    acc = net.accuracy(X, y)
    print(f"[{name}] 参数量 {net.n_params}, 最终损失 {losses[-1]:.4f}, "
          f"训练准确率 {acc:.1%}")
    return net, losses, snaps


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # ------------------------------------------------ 图 1: 三个数据集
    data = {
        "双月": make_moons(240, rng=rng),
        "圆环": make_circles(240, rng=rng),
        "螺旋": make_spirals(120, 3, rng=rng),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, (X, y)) in zip(axes, data.items()):
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CLASS_COLORS, s=10,
                   edgecolors="k", linewidths=0.2)
        ax.set_title(f"{name} (线性不可分)")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("第 3 课的直线切不开它们 —— 需要隐藏层")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "datasets.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------ 螺旋 (主战场)
    X, y = data["螺旋"]
    net, losses, snaps = run_dataset("螺旋", X, y, 3, rng)
    with open(os.path.join(FIG_DIR, "spiral_loss.pkl"), "wb") as f:
        pickle.dump(losses, f)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, step in zip(axes.ravel()[:4], sorted(snaps)):
        set_dense_state(net, snaps[step])
        plot_boundary(ax, net, X, y, f"训练 {step} 步后的决策边界")
    ax = axes.ravel()[4]
    ax.plot(losses)
    ax.set_title("损失曲线")
    ax.set_xlabel("训练步数")
    ax.grid(True, alpha=0.3)
    axes.ravel()[5].axis("off")
    fig.suptitle("一个 64x64 隐藏单元的 MLP 学会缠绕的螺旋")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spiral_training.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------ 双月
    X, y = data["双月"]
    net, losses, snaps = run_dataset("双月", X, y, 2, rng, n_steps=4000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    plot_boundary(axes[0], net, X, y, f"双月: 最终决策边界 (acc {net.accuracy(X, y):.1%})")
    axes[1].plot(losses)
    axes[1].set_title("损失曲线")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "moons_training.png"), dpi=130)
    plt.close(fig)

    print(f"\n图表已保存到 {FIG_DIR}/ (spiral_loss.pkl 留给第 6 课对比)")
    print("建议实验:")
    print("  1. 隐藏层改成 [2, 8, 8, 3] —— 小网络还能学会螺旋吗?")
    print("  2. 把 activation 换成 'relu', init 保持 'he' —— 收敛更快还是更慢?")
    print("  3. lr 改成 2.0 —— 观察 loss 爆炸 (第 2 课的稳定性分析在此重演)")


if __name__ == "__main__":
    main()
