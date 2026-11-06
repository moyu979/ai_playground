"""第 02 课 · 线性回归与梯度下降。

运行: python linear_regression.py
生成 figures/fit.png, figures/loss_surface.png, figures/learning_rates.png
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------- 数据与损失
def make_data(n: int = 120, w: float = 3.0, b: float = 1.0, noise: float = 0.3,
              rng: np.random.Generator | None = None):
    """真实规律 y = w*x + b + 高斯噪声。w, b 是要被"学"出来的真值。"""
    rng = rng or np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, size=n)
    y = w * x + b + rng.normal(0.0, noise, size=n)
    return x, y


def mse(x, y, w, b) -> float:
    """L(w,b) = (1/N) Σ (w x_i + b - y_i)²   ← README 公式 (1)"""
    return float(np.mean((w * x + b - y) ** 2))


def grad_mse(x, y, w, b):
    """解析梯度, README 公式 (3):
    ∂L/∂w = (2/N) Σ e_i x_i ,  ∂L/∂b = (2/N) Σ e_i ,  e_i = ŷ_i - y_i
    """
    e = w * x + b - y
    return 2.0 * np.mean(e * x), 2.0 * np.mean(e)


# ---------------------------------------------------------------- 三种解法
def closed_form(x, y) -> tuple[float, float]:
    """正规方程 θ = (XᵀX)⁻¹ Xᵀy (README 公式 2)。用 solve 而非 inv, 数值更稳。"""
    X = np.stack([x, np.ones_like(x)], axis=1)          # (N, 2)
    theta = np.linalg.solve(X.T @ X, X.T @ y)           # (2,)
    return float(theta[0]), float(theta[1])


def batch_gd(x, y, w0=-2.0, b0=-3.0, lr: float = 0.3, n_iter: int = 60):
    """全批量梯度下降, README 公式 (3)。返回参数轨迹和损失曲线。"""
    w, b = w0, b0
    path, losses = [(w, b)], [mse(x, y, w, b)]
    for _ in range(n_iter):
        gw, gb = grad_mse(x, y, w, b)
        w, b = w - lr * gw, b - lr * gb
        path.append((w, b))
        losses.append(mse(x, y, w, b))
    return np.array(path), np.array(losses)


def sgd(x, y, w0=-2.0, b0=-3.0, lr: float = 0.15, n_epochs: int = 30,
        batch_size: int = 16, rng: np.random.Generator | None = None):
    """mini-batch 随机梯度下降 (README §5): 每个 epoch 打乱后按批更新。"""
    rng = rng or np.random.default_rng(1)
    w, b = w0, b0
    path = [(w, b)]
    for _ in range(n_epochs):
        idx = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            j = idx[start : start + batch_size]
            gw, gb = grad_mse(x[j], y[j], w, b)         # 只用 batch 里的样本!
            w, b = w - lr * gw, b - lr * gb
            path.append((w, b))
    return np.array(path)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    x, y = make_data(rng=rng)
    w_true, b_true = 3.0, 1.0

    # 三种解法
    w_cf, b_cf = closed_form(x, y)
    path_gd, losses_gd = batch_gd(x, y, lr=0.3)
    path_sgd = sgd(x, y, lr=0.15)
    w_gd, b_gd = path_gd[-1]
    w_sgd, b_sgd = path_sgd[-1]

    print("=" * 66)
    print("三种解法对比 (真值 w=3.0, b=1.0)")
    print("=" * 66)
    print(f"{'方法':<16}{'w':>10}{'b':>10}{'MSE':>12}")
    print(f"{'解析解(正规方程)':<16}{w_cf:>10.4f}{b_cf:>10.4f}{mse(x, y, w_cf, b_cf):>12.5f}")
    print(f"{'批量 GD (60步)':<16}{w_gd:>10.4f}{b_gd:>10.4f}{mse(x, y, w_gd, b_gd):>12.5f}")
    print(f"{'SGD (30 epochs)':<16}{w_sgd:>10.4f}{b_sgd:>10.4f}{mse(x, y, w_sgd, b_sgd):>12.5f}")
    print("三者几乎一致 —— 但解析解只属于线性模型, GD 才能走到神经网络。")

    # ---------------------------------------------------------------- 图 1
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(x, y, s=14, alpha=0.6, label="数据 (含噪声)")
    xs = np.linspace(-1.05, 1.05, 10)
    for (w, b), name, ls in [
        ((w_true, b_true), "真值规律", ":"),
        ((w_cf, b_cf), "解析解", "--"),
        ((w_gd, b_gd), "批量 GD", "-"),
        ((w_sgd, b_sgd), "SGD", "-."),
    ]:
        ax.plot(xs, w * xs + b, ls, lw=2, label=f"{name} (w={w:.2f}, b={b:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("线性回归: 数据与拟合")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fit.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- 图 2
    # 损失曲面等高线: L(w,b) 是 (w,b) 空间的"势能面"
    W = np.linspace(-1.5, 5.5, 200)
    B = np.linspace(-4.0, 5.0, 200)
    WW, BB = np.meshgrid(W, B)
    # 向量化计算整个网格上的 MSE: E = WW*x[None,None,:] + BB - y 形状 (nb,nb,N)
    E = WW[:, :, None] * x[None, None, :] + BB[:, :, None] - y[None, None, :]
    L = (E**2).mean(axis=2)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    cs = ax.contour(WW, BB, L, levels=25, cmap="viridis", alpha=0.85)
    ax.clabel(cs, fontsize=6, fmt="%.1f")
    ax.plot(path_gd[:, 0], path_gd[:, 1], "o-", ms=3, lw=1.2, color="crimson",
            label="批量 GD 轨迹")
    ax.plot(path_sgd[:, 0], path_sgd[:, 1], ".", ms=2.5, color="royalblue",
            alpha=0.7, label="SGD 轨迹 (注意热噪声)")
    ax.plot(w_cf, b_cf, "*", ms=18, color="black", label="解析最优 (极小点)")
    ax.set_xlabel("参数 w")
    ax.set_ylabel("参数 b")
    ax.set_title("损失曲面 = 势能面; 训练 = 石头沿谷底滚落")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "loss_surface.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------------- 图 3
    lrs = [0.01, 0.3, 1.0, 2.5]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for lr in lrs:
        _, losses = batch_gd(x, y, lr=lr, n_iter=60)
        ax.plot(losses, label=f"lr={lr}")
    ax.set_yscale("log")
    ax.set_xlabel("迭代步数")
    ax.set_ylabel("MSE (对数轴)")
    ax.set_title("学习率: 太慢 / 合适 / 临界振荡 / 发散 (稳定性条件 lr < 2/λmax)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "learning_rates.png"), dpi=130)
    plt.close(fig)

    print(f"\n三张图已保存到 {FIG_DIR}/")
    print("建议实验: 把 make_data 里 noise 调大, 看 SGD 与解析解还一致吗?")
    print("          把 batch_size 调到 1 或 120, 观察 SGD 轨迹噪声的变化。")


if __name__ == "__main__":
    main()
