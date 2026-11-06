"""第 07 课 · 对照实验: 把图像摊平喂 MLP vs 卷积 (结构先验的价值)。

运行: python mlp_vs_cnn.py   (约 2 分钟, CPU 即可)
生成 figures/mlp_vs_cnn.png

两个模型, 同样的数据、同样的训练预算 (步数/批大小/优化器全相同):
  MLP  : Flatten(784) → 128 → 10     —— 看不见像素的空间排布
  CNN  : 两层 3×3 卷积 + 池化 → 10   —— 内置局部性 + 平移不变性
看两个数字: 测试准确率, 以及参数量 (CNN 用少得多的参数赢)。
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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
ROOT = os.path.dirname(os.path.abspath(__file__))


class MLP(nn.Module):
    """第 4 课的思路直接搬到图像: 摊平 → 全连接。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """小卷积网 (比 train_mnist.py 的还小一号): 卷积参数与图像大小无关。"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(torch.flatten(x, 1))


def load_data(n_train: int = 4096):
    tf = transforms.ToTensor()
    root = os.path.join(ROOT, "data")
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    te = datasets.MNIST(root, train=False, download=True, transform=tf)
    Xtr = tr.data[:n_train].float()[:, None] / 255.0
    ytr = tr.targets[:n_train]
    Xte = te.data.float()[:, None] / 255.0
    yte = te.targets
    return Xtr, ytr, Xte, yte


def train_model(model, Xtr, ytr, steps: int = 600, batch: int = 64,
                seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        idx = rng.integers(0, len(Xtr), batch)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def test_acc(model, Xte, yte) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(Xte), 1000):
        pred = model(Xte[i : i + 1000]).argmax(1)
        correct += (pred == yte[i : i + 1000]).sum().item()
    return correct / len(Xte)


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    Xtr, ytr, Xte, yte = load_data()
    models = {"MLP (摊平)": MLP(), "CNN (卷积)": SmallCNN()}
    curves, accs, params = {}, {}, {}
    for name, model in models.items():
        curves[name] = train_model(model, Xtr, ytr)
        accs[name] = test_acc(model, Xte, yte)
        params[name] = sum(p.numel() for p in model.parameters())
        print(f"{name:<12} 参数 {params[name]:>7,}  测试准确率 {accs[name]:.2%}")

    print()
    print(f"CNN 用约 1/{params['MLP (摊平)'] / params['CNN (卷积)']:.0f} 的参数"
          "换来更高的精度 —— 这就是『结构先验』的价值:")
    print("把『像素相邻相关/平移不变』写进架构, 数据和计算都花在刀刃上。")
    print("对照: 数据量再放大 10 倍, 差距会缩小 (先验的价值随数据量递减) ——")
    print("这也是大模型时代『少归纳偏置 + 大数据』路线的伏笔。")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for (name, curve), color in zip(curves.items(), ["royalblue", "firebrick"]):
        ax.plot(np.convolve(curve, np.ones(21) / 21, "valid"), color=color,
                label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("训练损失 (滑动平均)")
    ax.set_title("同等训练预算下的收敛")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = list(accs)
    bars = ax.bar(names, [accs[n] * 100 for n in names],
                  color=["royalblue", "firebrick"], width=0.5)
    for bar, n in zip(bars, names):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{accs[n]:.2%}\n({params[n]:,} 参数)", ha="center", fontsize=10)
    ax.set_ylabel("测试准确率 (%)")
    ax.set_title("4096 张训练样本 · 600 步 · 同一优化器")
    ax.set_ylim(85, 100)
    ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("对照实验: 结构先验 (卷积) vs 无结构 (摊平 MLP)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mlp_vs_cnn.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/mlp_vs_cnn.png")


if __name__ == "__main__":
    main()
