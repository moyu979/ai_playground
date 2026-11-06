"""第 05 课 · 用自动微分引擎搭一个 MLP 并训练 (双月数据)。

运行: python mlp_micrograd.py
生成 figures/micrograd_boundary.png

注意: 标量引擎极慢 (每个乘法都是一个 Python 对象)!
这里刻意用小网络小数据让你看清每一步; 第 6 课的 PyTorch 同样的模型快 1000 倍。
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

from micrograd import Value

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


class Neuron:
    """out = tanh(w·x + b)。所有参数都是 Value 标量。"""

    def __init__(self, n_in: int, rng: np.random.Generator):
        self.w = [Value(rng.normal(0, 1.0)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()


class Layer:
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        self.neurons = [Neuron(n_in, rng) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.layers = [Layer(sizes[i], sizes[i + 1], rng)
                       for i in range(len(sizes) - 1)]
        self.parameters = [p for layer in self.layers
                           for n in layer.neurons for p in n.w + [n.b]]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0]                      # 输出层 1 个神经元


def make_moons(n: int = 80, noise: float = 0.15, seed: int = 3):
    """第 4 课双月数据集的迷你版 (标量引擎吃不下太多样本)。"""
    rng = np.random.default_rng(seed)
    m = n // 2
    t = rng.uniform(0, np.pi, m)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.ones(m), -np.ones(m)])     # 标签 ±1 (配 tanh 输出)
    return X, y


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    X, y = make_moons()

    net = MLP([2, 8, 8, 1], rng)
    print(f"参数量: {len(net.parameters)} (每个都是 Value 标量)")

    # ---------------------------------------------------- 训练循环
    lr = 0.5
    n_epochs = 80
    history = []
    for epoch in range(n_epochs + 1):
        # 前向: 全部样本 (全批量, 标量引擎逐个算)
        preds = [net([Value(x0), Value(x1)]) for x0, x1 in X]
        # 损失: MSE, 目标 ±1
        loss = sum(((p - Value(t)) ** 2 for p, t in zip(preds, y)), Value(0.0))
        loss = loss * (1.0 / len(y))

        # 反传 + 更新 (引擎全自动!)
        for p in net.parameters:
            p.grad = 0.0
        loss.backward()
        for p in net.parameters:
            p.data -= lr * p.grad

        history.append(loss.data)
        if epoch % 10 == 0:
            acc = float(np.mean([(p.data > 0) == (t > 0)
                                 for p, t in zip(preds, y)]))
            print(f"epoch {epoch:3d}  loss {loss.data:.4f}  acc {acc:.1%}")

    # ---------------------------------------------------- 决策边界
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    xx, yy = np.meshgrid(np.linspace(-1.8, 3.0, 60), np.linspace(-1.5, 2.0, 60))
    zz = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            zz[i, j] = net([Value(xx[i, j]), Value(yy[i, j])]).data
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.8)
    ax.contour(xx, yy, zz, levels=[0.0], colors="black", linewidths=2)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c="royalblue", s=16, label="+1")
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c="firebrick", s=16, label="-1")
    ax.set_title("micrograd MLP 学到的边界 (输出值着色)")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(history)
    ax.set_title("MSE 损失")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "micrograd_boundary.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/micrograd_boundary.png")
    print("同样的网络, PyTorch (第 6 课) 训练它只需 ~0.1 秒 —— 差距就是引擎的代价。")


if __name__ == "__main__":
    main()
