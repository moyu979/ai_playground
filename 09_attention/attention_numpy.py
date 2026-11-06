"""第 09 课 · NumPy 手撕注意力: 每个部件都画给你看。

运行: python attention_numpy.py
生成 figures/attention_machinery.png (4 个热图/柱图) + 终端上的语义演示。
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(q, k, v, mask=None, scale: bool = True):
    """标准 scaled dot-product attention (单头, NumPy)。

    q, k: (T, d_k), v: (T, d_v), mask: (T, T) 布尔阵 True=允许看。
    返回 (输出 (T, d_v), 注意力权重 (T, T))。
    """
    d_k = q.shape[-1]
    scores = q @ k.T                          # ① 相关性 (T, T)
    if scale:
        scores = scores / np.sqrt(d_k)        # ② 防 softmax 饱和
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)  # ③ 因果掩码
    w = softmax(scores, axis=-1)              # ④ 每行归一化 = 注意力分配
    return w @ v, w                           # ⑤ 加权混合 value


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    # ================================================== 语义演示 (终端)
    # 手工设计: "it" (代词) 想找一个名词做指代对象
    words = ["it", "cat", "sat", "mat"]
    d = 3
    #                     [名词性, 动词性, 物件性]
    K = np.array([[2.0, 0.0, 0.0],   # cat: 强名词
                  [0.0, 2.0, 0.0],   # sat: 动词
                  [1.5, 0.0, 1.0]])  # mat: 名词+物件
    q_it = np.array([[2.0, 0.5, 0.0]])         # "it" 在找: 名词 >> 动词
    V = rng.normal(0, 1, (3, d))               # value 随便 (重点是权重)
    _, w = attention(q_it, K, V)
    print("=" * 62)
    print("语义演示: 代词 'it' 的 query 对各 token 的注意力")
    print("=" * 62)
    for word, weight in zip(words[1:], w[0]):
        bar = "#" * int(weight * 40)
        print(f"  {word:>4}: {weight:.3f} {bar}")
    print("  → 'it' 主要看向 'cat' (名词分数高) —— 这就是指代消解的雏形。")
    print("  真实模型里 q/k/v 不是手写的, 是从数据里【学】出来的。\n")

    # ================================================== 机械演示 (图)
    T, d_k, d_v = 6, 8, 8
    labels = ["t0", "t1", "t2", "t3", "t4", "t5"]
    X = rng.normal(0, 1, (T, d_k))             # 6 个 token 向量
    W_Q = rng.normal(0, 1, (d_k, d_k)) / np.sqrt(d_k)
    W_K = rng.normal(0, 1, (d_k, d_k)) / np.sqrt(d_k)
    W_V = rng.normal(0, 1, (d_k, d_v)) / np.sqrt(d_k)
    Q, Kv, Vv = X @ W_Q, X @ W_K, X @ W_V

    causal = np.tril(np.ones((T, T), dtype=bool))     # 下三角 = 只许看过去

    out_full, w_full = attention(Q, Kv, Vv, mask=None)
    out_caus, w_caus = attention(Q, Kv, Vv, mask=causal)

    # 饱和实验的数据 (供第 4 幅图): 有无 1/√d 的对比
    q_big = rng.normal(0, 3.0, (200, 16))       # 故意用大方差 q
    k_big = rng.normal(0, 3.0, (200, 16))
    s_raw = q_big @ k_big.T
    s_scl = s_raw / np.sqrt(16)
    peak_raw = softmax(s_raw, -1).max(-1)
    peak_scl = softmax(s_scl, -1).max(-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    im = ax.imshow(w_full, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("① 无掩码注意力权重 (行=query, 列=key)")
    ax.set_xticks(range(T), labels)
    ax.set_yticks(range(T), labels)
    ax.set_xlabel("key (被看的)")
    ax.set_ylabel("query (在看的)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    ax.imshow(w_caus, cmap="viridis", vmin=0, vmax=1)
    ax.set_title("② 因果掩码后: 上三角 = 0 (不许偷看未来)")
    ax.set_xticks(range(T), labels)
    ax.set_yticks(range(T), labels)

    ax = axes[1, 0]
    idx = 20
    ax.plot(softmax(s_raw[idx], -1), "o-", label="不除 √d (方差大→饱和)")
    ax.plot(softmax(s_scl[idx], -1), "s-", label="除以 √d (分布健康)")
    ax.set_title(f"③ 同一组分数的 softmax (样本 #{idx})")
    ax.set_xlabel("key 序号")
    ax.set_ylabel("注意力权重")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    pos = [0, 1, 2]
    vals = [peak_raw.mean(), peak_scl.mean(), 1.0 / 16]
    ax.bar(["无缩放", "÷√d", "均匀参考"], vals, color=["firebrick", "seagreen", "gray"])
    ax.set_title("④ 平均最大注意力权重 (1.0=完全饱和, 1/16=均匀)")
    ax.grid(True, alpha=0.3, axis="y")
    for i, val in enumerate(vals):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center")

    fig.suptitle("注意力机制解剖: softmax(QKᵀ/√d)V", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "attention_machinery.png"), dpi=130)
    plt.close(fig)
    print(f"图已保存到 {FIG_DIR}/attention_machinery.png")
    print("④ 号图: 不除 √d 时注意力几乎饱和 (接近 one-hot) —— 梯度死区。")


if __name__ == "__main__":
    main()
