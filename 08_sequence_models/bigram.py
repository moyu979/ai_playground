"""第 08 课 · bigram 语言模型: 数数就能生成文本。

运行: python bigram.py
生成 figures/bigram_heatmap.png; 打印采样出的名字与文本、困惑度。
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def load_names() -> list[str]:
    with open(os.path.join(DATA_DIR, "names.txt")) as f:
        return [line.strip().lower() for line in f if line.strip()]


def build_vocab(names: list[str]):
    chars = sorted(set("".join(names)))
    chars = ["."] + chars                      # '.' 是起止符, 放在 0 号位
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def bigram_counts(names: list[str], stoi: dict) -> np.ndarray:
    """N[i, j] = 大写语料里 (字符 i → 字符 j) 相邻出现的次数。"""
    N = np.zeros((len(stoi), len(stoi)))
    for name in names:
        seq = ["."] + list(name) + ["."]
        for a, b in zip(seq, seq[1:]):
            N[stoi[a], stoi[b]] += 1
    return N


def sample_from(P_row: np.ndarray, rng: np.random.Generator) -> int:
    """按概率分布 P_row 采样一个下标。"""
    return int(rng.choice(len(P_row), p=P_row))


def sample_word(P: np.ndarray, itos: dict, rng, max_len: int = 20) -> str:
    """从 bigram 矩阵采样一个名字: 从 '.' 出发, 采到 '.' 为止。"""
    idx = 0                                    # '.' 的下标
    out = []
    for _ in range(max_len):
        idx = sample_from(P[idx], rng)
        if idx == 0:
            break
        out.append(itos[idx])
    return "".join(out)


def bigram_nll(names: list[str], P: np.ndarray, stoi: dict) -> tuple[float, float]:
    """测试集平均负对数似然 (交叉熵) 与困惑度。"""
    total, count = 0.0, 0
    for name in names:
        seq = ["."] + list(name) + ["."]
        for a, b in zip(seq, seq[1:]):
            total += -np.log(P[stoi[a], stoi[b]] + 1e-12)
            count += 1
    ce = total / count
    return ce, float(np.exp(ce))


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    names = load_names()
    chars, stoi, itos = build_vocab(names)
    V = len(chars)
    print(f"名字数 {len(names)}, 词表大小 {V} ('.' + 26 个字母)")

    # ------------------------------------------------ 计数 → 概率
    N = bigram_counts(names, stoi)
    P = (N + 1.0) / (N.sum(axis=1, keepdims=True) + V)   # +1 拉普拉斯平滑
    print(f"计数矩阵形状 {N.shape}, 共 {N.sum():.0f} 个 bigram")

    # ------------------------------------------------ 训练集困惑度
    ce, ppl = bigram_nll(names, P, stoi)
    uniform_ce = np.log(V)
    print(f"训练集交叉熵 {ce:.3f} → 困惑度 {ppl:.2f}")
    print(f"参考: 均匀分布的交叉熵 = ln({V}) = {uniform_ce:.3f} (困惑度 {V})")
    print("任何利用更长上下文的模型都应该明显低于这个数 —— 这就是基线的意义。")

    # ------------------------------------------------ 采样
    print("\n从 bigram 采样 10 个名字:")
    words = [sample_word(P, itos, rng) for _ in range(10)]
    print("  " + ", ".join(words))

    # ------------------------------------------------ Shakespeare 字符 bigram
    with open(os.path.join(DATA_DIR, "shakespeare.txt"), encoding="utf-8") as f:
        text = f.read()
    vocab2 = sorted(set(text))
    stoi2 = {c: i for i, c in enumerate(vocab2)}
    N2 = np.zeros((len(vocab2), len(vocab2)))
    for a, b in zip(text, text[1:]):
        N2[stoi2[a], stoi2[b]] += 1
    P2 = (N2 + 1.0) / (N2.sum(axis=1, keepdims=True) + len(vocab2))

    idx = stoi2["\n"]
    out = []
    for _ in range(400):
        idx = sample_from(P2[idx], rng)
        out.append(vocab2[idx])
    print("\n从 Shakespeare 字符 bigram 采样 400 字符 (标点密集、词不像词):")
    print("-" * 62)
    print("".join(out))
    print("-" * 62)
    print("bigram 只知道『e 后面常是空格』这类单步统计 —— 没有长程记忆。")

    # ------------------------------------------------ 热图
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(P, cmap="Blues")
    ax.set_xticks(range(V), chars)
    ax.set_yticks(range(V), chars)
    for i in range(V):
        for j in range(V):
            if N[i, j] > 0:
                ax.text(j, i, f"{N[i, j]:.0f}", ha="center", va="center",
                        fontsize=5, color="orangered" if N[i, j] > 200 else "gray")
    ax.set_xlabel("下一个字符")
    ax.set_ylabel("当前字符")
    ax.set_title(f"bigram 计数矩阵 (数字 = 出现次数; 困惑度 {ppl:.2f})")
    fig.colorbar(im, fraction=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "bigram_heatmap.png"), dpi=130)
    plt.close(fig)
    print(f"\n热图已保存到 {FIG_DIR}/bigram_heatmap.png "
          "(看第一行: 名字通常怎么开头?)")


if __name__ == "__main__":
    main()
