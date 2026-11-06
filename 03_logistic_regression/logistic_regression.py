"""第 03 课 · 逻辑回归与 softmax 回归。

运行: python logistic_regression.py
生成 figures/: sigmoid.png, binary_boundary.png, loss_curve.png, multiclass.png
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------- 基础函数
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)   # 平移不变, 防溢出 (README §5)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def bce_loss(p, y) -> float:
    """二元交叉熵, README 公式 (3)。clip 防 log(0)。"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- 数据
def make_two_blobs(n=150, rng=None):
    """两个线性可分的高斯团。"""
    rng = rng or np.random.default_rng(0)
    X0 = rng.normal([-1.5, -1.0], 0.8, size=(n // 2, 2))
    X1 = rng.normal([1.5, 1.0], 0.8, size=(n // 2, 2))
    X = np.concatenate([X0, X1])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    return X, y


def make_three_blobs(n=180, rng=None):
    """三个类的高斯团 (softmax 回归用)。"""
    rng = rng or np.random.default_rng(1)
    centers = np.array([[0.0, 2.5], [-2.2, -1.5], [2.2, -1.5]])
    X = np.concatenate([rng.normal(c, 0.7, size=(n // 3, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n // 3)
    return X, y


# ---------------------------------------------------------------- 训练 (二分类)
def train_binary(X, y, lr=0.5, n_iter=300):
    """手写逻辑回归训练。模型: p = sigmoid(W x + b), 损失: BCE (README §3)。

    梯度 (对参数): ∂L/∂W = (1/N) Σ (p_i - y_i) x_i ,  ∂L/∂b = (1/N) Σ (p_i - y_i)
    注意它就是 "预测减标签" —— README 公式 (4)。
    """
    N, d = X.shape
    W = np.zeros(d)
    b = 0.0
    losses = []
    for _ in range(n_iter):
        z = X @ W + b                      # (N,)
        p = sigmoid(z)                     # (N,)
        losses.append(bce_loss(p, y))
        err = (p - y) / N                  # (N,)  ← p - y !
        W -= lr * err @ X
        b -= lr * err.sum()
    return W, b, losses


# ---------------------------------------------------------------- 训练 (多分类)
def train_softmax(X, y, n_classes=3, lr=0.5, n_iter=400):
    """softmax 回归。W: (K, d), 每行一个类的线性分数。"""
    N, d = X.shape
    W = np.zeros((n_classes, d))
    b = np.zeros(n_classes)
    Y = np.eye(n_classes)[y]               # one-hot (N, K)
    losses = []
    for _ in range(n_iter):
        Z = X @ W.T + b                    # (N, K)
        P = softmax(Z)                     # (N, K)
        losses.append(float(-np.mean(np.log(P[np.arange(N), y] + 1e-12))))
        err = (P - Y) / N                  # (N, K)  ← 依旧是 p - y !
        W -= lr * err.T @ X
        b -= lr * err.sum(axis=0)
    return W, b, losses


# ---------------------------------------------------------------- 主流程
def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- 图 1
    z = np.linspace(-6, 6, 300)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z, sigmoid(z), lw=2, label=r"$\sigma(z)=1/(1+e^{-z})$")
    ax.plot(z, sigmoid(z) * (1 - sigmoid(z)), "--", lw=2,
            label=r"$\sigma'$ (导数)")
    ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
    ax.set_title(r"sigmoid = 费米-狄拉克分布: $\sigma(z)=1/(e^{-z}+1)$")
    ax.set_xlabel("z (线性分数)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sigmoid.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------- 二分类
    X, y = make_two_blobs(rng=rng)
    W, b, losses = train_binary(X, y)
    p = sigmoid(X @ W + b)
    acc = float(np.mean((p > 0.5) == y))
    print("=" * 62)
    print("二分类 (逻辑回归)")
    print(f"  学到: W={np.round(W, 3)}, b={b:.3f}, 训练准确率={acc:.1%}")
    print(f"  初始损失 {losses[0]:.4f} → 最终损失 {losses[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # 图 2: 数据 + 决策边界 + 概率场
    ax = axes[0]
    xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 240), np.linspace(-4.5, 4.5, 240))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pp = sigmoid(grid @ W + b).reshape(xx.shape)
    ax.contourf(xx, yy, pp, levels=20, cmap="RdBu_r", alpha=0.75)
    ax.contour(xx, yy, pp, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(*X[y == 0].T, c="royalblue", s=14, label="类 0")
    ax.scatter(*X[y == 1].T, c="firebrick", s=14, label="类 1")
    ax.set_title(f"决策边界 (黑线) = w·x+b=0, 准确率 {acc:.1%}")
    ax.legend(fontsize=9)
    # 图 3: 损失曲线
    ax = axes[1]
    ax.plot(losses)
    ax.set_xlabel("迭代步数")
    ax.set_ylabel("BCE 损失")
    ax.set_title("交叉熵损失下降")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "binary_boundary.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------------------- 多分类
    X3, y3 = make_three_blobs(rng=rng)
    W3, b3, losses3 = train_softmax(X3, y3)
    P3 = softmax(X3 @ W3.T + b3)
    acc3 = float(np.mean(np.argmax(P3, axis=1) == y3))
    print("-" * 62)
    print("三分类 (softmax 回归)")
    print(f"  W 形状 {W3.shape} (每行一个类), 训练准确率={acc3:.1%}")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    pp3 = softmax(grid @ W3.T + b3).reshape(xx.shape + (3,))
    ax.contourf(xx, yy, np.argmax(pp3, axis=2), levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["#aec7e8", "#ffbb78", "#98df8a"], alpha=0.85)
    for k, c, name in [(0, "royalblue", "类 0"), (1, "darkorange", "类 1"),
                       (2, "green", "类 2")]:
        ax.scatter(*X3[y3 == k].T, c=c, s=14, label=name)
    ax.set_title(f"softmax 回归: 三个线性边界, 准确率 {acc3:.1%}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "multiclass.png"), dpi=130)
    plt.close(fig)

    # ---------------------------------------------- 对照组: 线性模型的死角
    # 反面教材: 同一个逻辑回归, 喂线性不可分的数据, 亲眼看到它露馅
    n_c = 240
    ang = rng.uniform(0, 2 * np.pi, n_c // 2)
    outer = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    inner = 0.35 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    Xc = np.concatenate([outer, inner]) + rng.normal(0, 0.08, (n_c, 2))
    yc = np.concatenate([np.zeros(n_c // 2), np.ones(n_c // 2)])
    Wc, bc, _ = train_binary(Xc, yc, lr=0.5, n_iter=800)
    accc = float(np.mean((sigmoid(Xc @ Wc + bc) > 0.5) == yc))
    print("-" * 62)
    print(f"对照组 (圆环数据, 线性不可分): 逻辑回归只有 {accc:.1%}")
    print("→ 这条直线已是它的极限; 第 4 课的 MLP 在同样的数据上接近 100%。")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, (Xd, yd, Wd, bd, title) in zip(axes, [
        (X, y, W, b, f"线性可分 (高斯团): {acc:.1%}"),
        (Xc, yc, Wc, bc, f"圆环 (线性不可分): 只有 {accc:.1%}"),
    ]):
        xx2, yy2 = np.meshgrid(np.linspace(-4.5, 4.5, 220),
                               np.linspace(-4.5, 4.5, 220))
        grid2 = np.stack([xx2.ravel(), yy2.ravel()], axis=1)
        pp = sigmoid(grid2 @ Wd + bd).reshape(xx2.shape)
        ax.contourf(xx2, yy2, pp, levels=20, cmap="RdBu_r", alpha=0.75)
        ax.contour(xx2, yy2, pp, levels=[0.5], colors="black", linewidths=2)
        ax.scatter(*Xd[yd == 0].T, c="royalblue", s=12, label="类 0")
        ax.scatter(*Xd[yd == 1].T, c="firebrick", s=12, label="类 1")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.legend(fontsize=8)
    fig.suptitle("同一个模型: 数据可分就赢, 不可分就露馅 —— 结构的局限要靠结构解决")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "linear_fail.png"), dpi=130)
    plt.close(fig)

    print(f"\n五张图已保存到 {FIG_DIR}/")
    print("建议实验: 把两个高斯团中心拉近 (make_two_blobs 里 ±1.5 改 ±0.5),")
    print("          看决策边界怎么摆、准确率掉到多少 —— 这叫贝叶斯误差。")


if __name__ == "__main__":
    main()
