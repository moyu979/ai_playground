"""第 08 课 · 字符级 RNN 语言模型 (手写 RNNCell) + 梯度消失实测。

运行: python rnn_char.py   (CPU/MPS 约需 1~3 分钟)
生成 figures/rnn_analysis.png (RNN vs LSTM 损失曲线 + 梯度随时间衰减)
并把采样文本存到 figures/rnn_sample.txt
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

B, T = 32, 32          # batch, 序列长度
EMB, HIDDEN = 32, 128
STEPS = 1000


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CharRNN(nn.Module):
    """手写 RNN 单元: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)  (README §4)。"""

    def __init__(self, vocab: int, emb: int = EMB, hidden: int = HIDDEN):
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(vocab, emb)
        self.xh = nn.Linear(emb, hidden)          # 输入 → 隐状态
        self.hh = nn.Linear(hidden, hidden, bias=False)  # 隐状态 → 隐状态 (递归!)
        self.head = nn.Linear(hidden, vocab)

    def step(self, x_idx: torch.Tensor, h: torch.Tensor):
        """单步: x_idx (B,), h (B, H) → logits (B, V), 新 h。"""
        x = self.emb(x_idx)
        h = torch.tanh(self.xh(x) + self.hh(h))   # tanh 把状态压回 (-1,1)
        return self.head(h), h

    def forward(self, idx: torch.Tensor):
        """idx (B, T) → logits (B, T, V), h 序列列表 (反传后可查 .grad)。"""
        h = torch.zeros(idx.size(0), self.hidden, device=idx.device)
        outs, hs = [], []
        for t in range(idx.size(1)):
            logits, h = self.step(idx[:, t], h)
            outs.append(logits)
            hs.append(h)
        return torch.stack(outs, dim=1), hs


class CharLSTM(nn.Module):
    """同容量 nn.LSTM 对照组 (门控让信息沿『加法路径』跨时间流动)。"""

    def __init__(self, vocab: int, emb: int = EMB, hidden: int = HIDDEN):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx: torch.Tensor):
        x = self.emb(idx)
        out, _ = self.lstm(x)                     # (B, T, H)
        return self.head(out), None               # 接口与 CharRNN 对齐


def get_batch(ids: torch.Tensor):
    """随机采样 (B, T) 的文本窗口; 目标 = 输入右移一位。"""
    ix = torch.randint(0, len(ids) - T - 1, (B,))
    x = torch.stack([ids[i : i + T] for i in ix])
    y = torch.stack([ids[i + 1 : i + 1 + T] for i in ix])
    return x, y


def train_lm(model, ids, steps: int, dev, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        x, y = get_batch(ids)
        x, y = x.to(dev), y.to(dev)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # RNN 必备: 梯度裁剪
        opt.step()
        losses.append(loss.item())
        if step % 250 == 0:
            print(f"    step {step:4d}  ce {loss.item():.3f}  "
                  f"ppl {np.exp(loss.item()):.1f}")
    return losses


@torch.no_grad()
def generate(model, itos: dict, stoi: dict, dev, n: int = 300,
             temp: float = 0.8) -> str:
    """自回归采样: 从 '\n' 出发, 一次喂一个字符 (训练时是整段, 生成时逐步)。"""
    model.eval()
    if isinstance(model, CharLSTM):
        h = None
    else:
        h = torch.zeros(1, model.hidden, device=dev)
    idx = torch.tensor([stoi["\n"]], device=dev)
    out = []
    for _ in range(n):
        if isinstance(model, CharLSTM):
            x = model.emb(idx[None])
            lstm_out, h2 = model.lstm(x, h)
            logits = model.head(lstm_out[0, -1])
            h = h2
        else:
            logits, h = model.step(idx, h)
        p = F.softmax(logits[-1] / temp, dim=-1)
        idx = torch.multinomial(p, 1)
        out.append(itos[int(idx)])
    model.train()
    return "".join(out)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    dev = get_device()
    torch.manual_seed(0)
    print(f"设备: {dev}")

    text = open(os.path.join(DATA_DIR, "shakespeare.txt"), encoding="utf-8").read()
    vocab = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text])
    V = len(vocab)
    print(f"语料 {len(text):,} 字符, 词表 {V}")

    print("\n[1/3] 训练手写 CharRNN ...")
    rnn = CharRNN(V).to(dev)
    rnn_losses = train_lm(rnn, ids, STEPS, dev, seed=0)

    print("\n[2/3] 训练同容量 nn.LSTM ...")
    lstm = CharLSTM(V).to(dev)
    lstm_losses = train_lm(lstm, ids, STEPS, dev, seed=0)
    print(f"最终交叉熵: RNN {rnn_losses[-1]:.3f} | LSTM {lstm_losses[-1]:.3f}")

    print("\n[3/3] 用训练好的 RNN 采样 (温度 0.8) ...")
    sample = generate(rnn, itos, stoi, dev)
    with open(os.path.join(FIG_DIR, "rnn_sample.txt"), "w") as f:
        f.write(sample)
    print("-" * 62)
    print(sample)
    print("-" * 62)
    print("比 bigram 像样多了 (拼写、换行节奏), 但长句仍会跑题 —— 上下文瓶颈。")

    # ------------------------------------------------ 梯度消失实测
    x, y = get_batch(ids)
    x, y = x.to(dev), y.to(dev)
    logits, hs = rnn(x)
    for h in hs:
        h.retain_grad()      # 非叶子张量默认不存 .grad, 需显式保留
    loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    loss.backward()
    norms = [h.grad.norm().item() for h in hs]   # ‖∂L/∂h_t‖ 随 t

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    ax.plot(rnn_losses, label="手写 RNN", alpha=0.8)
    ax.plot(lstm_losses, label="nn.LSTM", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("交叉熵")
    ax.set_title("RNN vs LSTM 损失曲线 (短程训练互有胜负; 长依赖任务门控优势明显)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(norms, "o-", ms=3)
    ax.set_xlabel("时间步 t (1 = 离损失最近)")
    ax.set_ylabel(r"$\|\partial L/\partial h_t\|$")
    ax.set_title("梯度随时间指数衰减 (vanishing gradients, README §4)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rnn_analysis.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/rnn_analysis.png")
    print("右图就是 RNN 的死穴: 早期时间步的梯度趋近 0 → 学不到长程依赖。")
    print("第 9 课: 让每个位置直接『看见』所有位置 —— 注意力。")


if __name__ == "__main__":
    main()
