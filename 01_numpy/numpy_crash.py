"""第 01 课 · NumPy 速成演示。

运行: python numpy_crash.py
按节打印讲解, 最后生成两张图到 figures/。
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(f" {title}")
    print("=" * 62)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ 1
    section("1. ndarray: 标量/矢量/矩阵/张量, 统一是 shape 不同的数组")
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"a =\n{a}\nshape={a.shape}, dtype={a.dtype}, ndim={a.ndim}")
    print("约定: 数据批次的第一维是 batch。 (B, d) = B 个样本, 每个 d 维。")

    # ------------------------------------------------------------------ 2
    section("2. 广播: 从最后一维向前对齐, 为 1 的维度自动扩展")
    A = np.arange(6.0).reshape(2, 3)
    b = np.array([10.0, 20.0, 30.0])      # (3,)  → 视为 (1,3) → 广播到 (2,3)
    c = np.array([[100.0], [200.0]])      # (2,1) → 广播到 (2,3)
    print(f"A (2,3) + b (3,)  =\n{A + b}")
    print(f"A (2,3) + c (2,1) =\n{A + c}")
    print("错误示范: A + np.array([1., 2.]) 会报错, 因为 (2,) 对齐到 (3,) 失败。")

    # ------------------------------------------------------------------ 3
    section("3. 索引: 切片 / 布尔掩码 / 花式索引")
    X = rng.normal(size=(5, 2)) * np.array([3.0, 1.0])
    print(f"X =\n{np.round(X, 2)}")
    print(f"X[:, 0] (第 0 列, 所有样本) = {np.round(X[:, 0], 2)}")
    mask = np.linalg.norm(X, axis=1) > 2
    print(f"模长 > 2 的样本 (布尔掩码) =\n{np.round(X[mask], 2)}")
    print(f"打乱后 (花式索引) =\n{np.round(X[rng.permutation(5)], 2)}")

    # ------------------------------------------------------------------ 4
    section("4. einsum: 用指标记号写张量运算")
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 5))
    v = rng.normal(size=4)
    assert np.allclose(np.einsum("ij,jk->ik", A, B), A @ B)
    assert np.allclose(np.einsum("i,i->", v, v), v @ v)
    assert np.allclose(np.einsum("ij,kj->ik", A, A), A @ A.T)
    print("三个 einsum 全部与等价的 @ / v@v / A@A.T 对齐 ✓")
    print("  'ij,jk->ik'  矩阵乘   C_ik = Σ_j A_ij B_jk")
    print("  'i,i->'      内积     s    = Σ_i v_i v_i")
    print("  'ij,kj->ik'  A @ B.T  O_ik = Σ_j A_ij B_kj")

    # ------------------------------------------------------------------ 5
    section("5. 标准化: 神经网络代码里最常见的三行")
    X = rng.normal(loc=5.0, scale=3.0, size=(1000, 2)) * np.array([1.0, 10.0])
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print(f"标准化前: mean={np.round(X.mean(0), 2)}, std={np.round(X.std(0), 2)}")
    print(f"标准化后: mean={np.round(Xn.mean(0), 2)}, std={np.round(Xn.std(0), 2)}")
    print("((axis=0 表示沿样本维聚合 → 得到每列的统计量, 靠广播加回所有行)")

    # ------------------------------------------------------------------ 6
    section("6. 向量化 vs 循环: 差多少倍?")
    N, d = 100_000, 64
    X = rng.normal(size=(N, d))
    c = rng.normal(size=d)

    t0 = time.perf_counter()
    d_loop = np.array([np.linalg.norm(x - c) for x in X])
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_vec = np.linalg.norm(X - c, axis=1)
    t_vec = time.perf_counter() - t0

    assert np.allclose(d_loop, d_vec)
    print(f"循环版: {t_loop * 1000:8.1f} ms")
    print(f"向量化: {t_vec * 1000:8.1f} ms")
    print(f"加速比: {t_loop / t_vec:8.1f} x   ← 结果完全一致")

    # ------------------------------------------------------------------ 图
    # 图 1: 广播可视化 —— 行向量 + 列向量 相加得到一个"场"
    row = np.linspace(-2, 2, 9)[None, :]          # (1, 9)
    col = np.linspace(-2, 2, 9)[:, None]          # (9, 1)
    field = row ** 2 + col ** 2                   # 广播相加 → (9, 9)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].imshow(np.tile(row, (9, 1)), cmap="viridis")
    axes[0].set_title("row (1,9) 沿列广播")
    axes[1].imshow(np.tile(col, (1, 9)), cmap="viridis")
    axes[1].set_title("col (9,1) 沿行广播")
    im = axes[2].imshow(field, cmap="magma")
    axes[2].set_title("相加结果 (9,9) = r²+c²")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    fig.suptitle("广播: (9,1) + (1,9) → (9,9) —— 类比: 两个正交的场叠加")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "broadcasting.png"), dpi=130)
    plt.close(fig)

    # 图 2: 向量化加速比
    sizes = [10, 100, 1000, 10_000, 100_000]
    t_loop_list, t_vec_list = [], []
    for n in sizes:
        X = rng.normal(size=(n, d))
        c = rng.normal(size=d)
        t0 = time.perf_counter()
        _ = [np.linalg.norm(x - c) for x in X]
        t_loop_list.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = np.linalg.norm(X - c, axis=1)
        t_vec_list.append(time.perf_counter() - t0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(sizes, t_loop_list, "o-", label="python 循环")
    ax.loglog(sizes, t_vec_list, "s-", label="向量化")
    ax.set_xlabel("样本数 N")
    ax.set_ylabel("耗时 (s)")
    ax.set_title("向量化 vs 循环")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "vectorize_speed.png"), dpi=130)
    plt.close(fig)

    print(f"\n两张图已保存到 {FIG_DIR}/")
    print("接下来: python exercises.py 开始练习。")


if __name__ == "__main__":
    main()
