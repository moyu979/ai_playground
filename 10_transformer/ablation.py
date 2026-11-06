"""第 10 课 · 消融实验 (反面教材): 拆掉 Transformer 的零件会怎样?

运行: python ablation.py   (MPS 约 2~4 分钟)
生成 figures/ablation.png

三个模型, 同数据/同预算/同种子, 只差一个开关 (GPTConfig 里的消融开关):
  A. 完整 GPT (残差 + 位置编码)        —— 对照组
  B. 无残差连接   (use_residual=False) —— 梯度高速公路被切断
  C. 无位置编码   (use_pos_emb=False)  —— 注意力对 token 顺序全盲
口说无凭, 各训 600 步看损失。
"""

import os

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                          "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "..", "data", "shakespeare.txt")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    dev = torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cpu")
    print(f"设备: {dev}")

    text = open(DATA, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    block, batch = 64, 32

    def get_batch():
        ix = torch.randint(0, len(ids) - block - 1, (batch,))
        x = torch.stack([ids[i : i + block] for i in ix]).to(dev)
        y = torch.stack([ids[i + 1 : i + 1 + block] for i in ix]).to(dev)
        return x, y

    variants = [
        ("完整 GPT (对照组)", {}),
        ("无残差连接", {"use_residual": False}),
        ("无位置编码", {"use_pos_emb": False}),
    ]
    iters = 600
    curves = {}
    print(f"\n配置: 3 层 · 4 头 · 96 维 · block {block} · {iters} 步 · 同种子\n")
    for name, flags in variants:
        torch.manual_seed(1337)
        cfg = GPTConfig(vocab_size=len(chars), block_size=block, n_layer=3,
                        n_head=4, n_embd=96, **flags)
        model = GPT(cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                betas=(0.9, 0.95), weight_decay=0.1)
        losses = []
        for it in range(iters + 1):
            x, y = get_batch()
            _, loss = model(x, y)
            if it % 200 == 0:
                print(f"  [{name}] iter {it:4d}  ce {loss.item():.3f}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        curves[name] = losses
        print(f"  → {name}: 末值 {losses[-1]:.3f}\n")

    print("=" * 62)
    vals = {k: v[-1] for k, v in curves.items()}
    print("结论 (对数轴上看得更清楚):")
    print(f"  无残差连接: 损失 {vals['无残差连接']:.3f} vs 完整 "
          f"{vals['完整 GPT (对照组)']:.3f} —— 深层网络的梯度高速路不是装饰。")
    print(f"  无位置编码: 损失 {vals['无位置编码']:.3f} —— 注意力本身是"
          "『无序集合』运算, 顺序信息必须显式注入。")
    print("  (无位置编码仍能学到不少: 因果掩码+字符局部性兜底 —— 但上限明显更低)")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for (name, curve), color in zip(curves.items(),
                                    ["seagreen", "firebrick", "royalblue"]):
        ax.plot(np.convolve(curve, np.ones(21) / 21, "valid"), color=color,
                lw=1.6, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("交叉熵 (对数轴)")
    ax.set_title("消融实验: 拆掉一个零件, 同数据同预算同种子")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ablation.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/ablation.png")


if __name__ == "__main__":
    main()
