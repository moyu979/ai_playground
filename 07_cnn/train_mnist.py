"""第 07 课 · 在 MNIST 上训练 CNN (首次运行自动下载数据集 ~65MB)。

运行: python train_mnist.py [--epochs 3]
生成 figures/mnist_curves.png, mnist_filters.png,
     mnist_featuremaps.png, mnist_predictions.png
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CNN(nn.Module):
    """README §5 的结构。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)      # 28→28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)     # 14→14
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)       # 28→14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)       # 14→7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                               # logits (B, 10)


def evaluate(net, loader, dev) -> float:
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = net(xb.to(dev)).argmax(1).cpu()
            correct += (pred == yb).sum().item()
            total += len(yb)
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    dev = get_device()
    torch.manual_seed(0)
    print(f"设备: {dev}")

    # ------------------------------------------------ 数据 (自动下载)
    tf = transforms.ToTensor()          # 已缩放到 [0,1], 教学够用
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=512)

    # ------------------------------------------------ 模型与优化
    net = CNN().to(dev)
    print(f"参数量: {sum(p.numel() for p in net.parameters()):,} "
          "(对比: MLP 784→128 就要 ~10 万, 且完全不看空间结构)")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    losses, accs = [], []
    step = 0
    for epoch in range(args.epochs):
        net.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step:5d}  loss {loss.item():.4f}")
        acc = evaluate(net, test_ld, dev)
        accs.append(acc)
        print(f"epoch {epoch + 1} 结束: 测试准确率 {acc:.2%}")
    torch.save(net.state_dict(), os.path.join(FIG_DIR, "mnist_cnn.pt"))

    # ------------------------------------------------ 图 1: 曲线
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(losses, lw=0.6)
    axes[0].set_title("训练损失")
    axes[0].set_xlabel("step")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(range(1, args.epochs + 1), np.array(accs) * 100, "o-")
    axes[1].set_title("测试准确率")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("%")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_curves.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------ 图 2: 第一层卷积核
    k = net.conv1.weight.detach().cpu().numpy()          # (16, 1, 3, 3)
    fig, axes = plt.subplots(2, 8, figsize=(11, 3.2))
    for ax, w in zip(axes.ravel(), k):
        ax.imshow(w[0], cmap="RdBu_r")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("第一层 16 个 3×3 卷积核 —— 无人教它, 自动长成边缘/条纹检测器")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_filters.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------ 图 3: 特征图
    net.eval()
    with torch.no_grad():
        img, label = test_ds[0]
        fmap = F.relu(net.conv1(img[None].to(dev)))[0].cpu().numpy()  # (16,28,28)
    fig, axes = plt.subplots(2, 8, figsize=(11, 3.4))
    axes[0, 0].imshow(img[0], cmap="gray")
    axes[0, 0].set_title(f"输入 (标签 {label})", fontsize=8)
    for ax, fm in zip(axes.ravel()[1:], fmap[:15]):
        ax.imshow(fm, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("第一层输出: 15 张特征图 (亮=该处检测到对应特征)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_featuremaps.png"), dpi=130)
    plt.close(fig)

    # ------------------------------------------------ 图 4: 预测样例
    with torch.no_grad():
        xb, yb = next(iter(test_ld))
        pred = net(xb.to(dev)).argmax(1).cpu()
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 4))
    for ax, im, t, p in zip(axes.ravel(), xb, yb, pred):
        ax.imshow(im[0], cmap="gray")
        color = "green" if t == p else "red"
        ax.set_title(f"真 {t} / 预 {p}", color=color, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("测试集预测 (绿=对, 红=错)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mnist_predictions.png"), dpi=130)
    plt.close(fig)

    print(f"\n四张图已保存到 {FIG_DIR}/")
    print("建议实验: 注释掉两个 MaxPool (改 Flatten 尺寸 32*28*28), 看参数量与精度的变化。")


if __name__ == "__main__":
    main()
