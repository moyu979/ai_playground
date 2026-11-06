"""第 10 课 · 训练字符级 GPT (tiny Shakespeare)。

运行: python train.py [--iters 2000] [--n-layer 4] [--n-embd 128] ...
产出: model.pt, figures/gpt_loss.png, figures/gpt_sample.txt
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "data", "shakespeare.txt")
FIG_DIR = os.path.join(ROOT, "figures")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    dev = get_device()
    torch.manual_seed(1337)
    print(f"设备: {dev}")

    # ------------------------------------------------ 数据: 字符级 tokenizer
    text = open(DATA, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(ids))
    train_ids, val_ids = ids[:n_train], ids[n_train:]     # 90/10 划分
    V = len(chars)
    print(f"语料 {len(text):,} 字符, 词表 {V}, 训练/验证 = {n_train:,}/{len(val_ids):,}")

    def get_batch(split: str):
        src = train_ids if split == "train" else val_ids
        ix = torch.randint(0, len(src) - args.block - 1, (args.batch,))
        x = torch.stack([src[i : i + args.block] for i in ix])
        y = torch.stack([src[i + 1 : i + 1 + args.block] for i in ix])
        return x.to(dev), y.to(dev)

    # ------------------------------------------------ 模型与优化
    cfg = GPTConfig(vocab_size=V, block_size=args.block, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(dev)
    print(f"参数量 {model.num_params():,} | 配置 {cfg.n_layer}L {cfg.n_head}H "
          f"{cfg.n_embd}d block {cfg.block_size}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)

    @torch.no_grad()
    def estimate_loss(n_batches: int = 20) -> dict:
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = []
            for _ in range(n_batches):
                x, y = get_batch(split)
                _, loss = model(x, y)
                losses.append(loss.item())
            out[split] = float(np.mean(losses))
        model.train()
        return out

    # ------------------------------------------------ 训练循环 (五步真言)
    hist = []
    for it in range(args.iters + 1):
        if it % 100 == 0 or it == args.iters:
            losses = estimate_loss()
            hist.append((it, losses["train"], losses["val"]))
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | ppl {np.exp(losses['val']):.1f}")
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 大模型标配
        opt.step()

    torch.save({"model": model.state_dict(),
                "config": vars(cfg),
                "vocab": chars}, os.path.join(ROOT, "model.pt"))

    # ------------------------------------------------ 损失曲线
    its, tr, va = zip(*hist)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(its, tr, "o-", label="train")
    ax.plot(its, va, "s-", label="val")
    ax.set_xlabel("iter")
    ax.set_ylabel("交叉熵")
    ax.set_title(f"GPT 训练曲线 ({model.num_params():,} 参数)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gpt_loss.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------ 生成样例
    context = torch.zeros((1, 1), dtype=torch.long, device=dev)  # 从空上下文开始
    sample = model.generate(context, 500, temperature=0.8, top_k=40)[0].tolist()
    text_out = "".join(itos[i] for i in sample)
    with open(os.path.join(FIG_DIR, "gpt_sample.txt"), "w") as f:
        f.write(text_out)
    print("\n" + "=" * 62)
    print("GPT 生成的『伪莎士比亚』(温度 0.8, top-k 40):")
    print("=" * 62)
    print(text_out)
    print("=" * 62)
    print(f"模型已存 {ROOT}/model.pt; 试玩: python sample.py --temperature 1.0")


if __name__ == "__main__":
    main()
