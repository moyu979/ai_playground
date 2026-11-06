"""第 08 课 · Bengio (2003) 式神经语言模型: embedding + MLP。

运行: python mlp_lm.py
生成 figures/mlp_lm.png (损失曲线 + 2D embedding 散点)。
注意 embedding 维度故意设为 2 —— 为了能直接画在纸上。
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLOCK_SIZE = 4      # 上下文: 看前 4 个字符
EMB_DIM = 2         # 每个字符的向量维度 (教学: 2 维可直接可视化)
HIDDEN = 64


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataset(names: list[str], stoi: dict, block_size: int):
    """滑窗切样本: context (block_size 个字符索引) → target (下一个字符)。"""
    X, Y = [], []
    for name in names:
        seq = ["."] * block_size + list(name) + ["."]
        for i in range(block_size, len(seq)):
            ctx = [stoi[c] for c in seq[i - block_size : i]]
            X.append(ctx)
            Y.append(stoi[seq[i]])
    return torch.tensor(X), torch.tensor(Y)


class MLP_LM(nn.Module):
    """C: 嵌入表 (V, 2); 视图: 4 个字符 → 4 个二维向量 → 拼成 8 维 → MLP → 27 个 logit。"""

    def __init__(self, vocab: int):
        super().__init__()
        self.C = nn.Embedding(vocab, EMB_DIM)
        self.fc1 = nn.Linear(BLOCK_SIZE * EMB_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, vocab)

    def forward(self, x):                       # x: (B, block_size)
        emb = self.C(x)                         # (B, block_size, 2)
        h = torch.tanh(self.fc1(emb.flatten(1)))  # (B, hidden)
        return self.fc2(h)                      # (B, vocab)


@torch.no_grad()
def sample_names(net, itos, rng, n: int = 12, min_len: int = 3) -> list[str]:
    dev = next(net.parameters()).device
    out = []
    for _ in range(n):
        for _try in range(50):
            ctx = torch.zeros(1, BLOCK_SIZE, dtype=torch.long, device=dev)
            letters = []
            for _step in range(20):
                logits = net(ctx)[0]                    # (V,)
                p = F.softmax(logits / 0.8, dim=-1)     # 0.8 = 采样温度
                nxt = int(torch.multinomial(p, 1))
                if nxt == 0:
                    break
                letters.append(itos[nxt])
                ctx = torch.cat([ctx[:, 1:], torch.tensor([[nxt]], device=dev)], 1)
            if len(letters) >= min_len:
                out.append("".join(letters))
                break
    return out


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    dev = get_device()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    names = [l.strip().lower() for l in
             open(os.path.join(DATA_DIR, "names.txt")) if l.strip()]
    chars = ["."] + sorted(set("".join(names)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars)

    X, Y = build_dataset(names, stoi, BLOCK_SIZE)
    print(f"样本数 {len(X)} (每个名字贡献 len+1 个滑窗样本), 词表 {V}")
    print(f"示例: {''.join(itos[i] for i in X[3].tolist())} → {itos[int(Y[3])]}")

    net = MLP_LM(V).to(dev)
    print(f"参数量: {sum(p.numel() for p in net.parameters())}")
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    losses = []
    for step in range(6001):
        idx = torch.randint(0, len(X), (64,))
        xb, yb = X[idx].to(dev), Y[idx].to(dev)
        loss = F.cross_entropy(net(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 1500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")

    ce = losses[-1]
    print(f"最终交叉熵 {ce:.3f} → 困惑度 {np.exp(ce):.2f}  "
          f"(bigram 基线约 5~7 —— 更长上下文赢了)")
    print("采样 12 个名字: " + ", ".join(sample_names(net, itos, rng)))

    # ------------------------------------------------ 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot(np.convolve(losses, np.ones(101) / 101, "valid"))
    ax.set_title("训练损失 (滑动平均)")
    ax.set_xlabel("step")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    C = net.C.weight.detach().cpu().numpy()      # (V, 2)
    ax.scatter(C[:, 0], C[:, 1], s=60, color="royalblue", zorder=2)
    for i, c in enumerate(chars):
        ax.annotate(c, (C[i, 0], C[i, 1]), fontsize=12,
                    xytext=(C[i, 0] + 0.02, C[i, 1] + 0.02))
    ax.set_title("学到的字符 embedding (2D) —— 注意元音 a e i o u 抱团")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mlp_lm.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/mlp_lm.png")
    print(" embedding 是『分布表示』的第一个证据: 意义相近 → 空间相近。")


if __name__ == "__main__":
    main()
