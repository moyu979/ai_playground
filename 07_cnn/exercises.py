"""第 07 课练习。实现后运行 `python exercises.py` 自动测试。"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 手写 conv2d (带 stride / padding) -------------------------------------------
def conv2d_np(x: np.ndarray, k: np.ndarray, stride: int = 1,
              pad: int = 0) -> np.ndarray:
    """单通道二维互相关。

    x: (H, W), k: (Kh, Kw) → 输出 (H_out, W_out)
      H_out = (H + 2*pad - Kh) // stride + 1
    实现: 先给 x 四周 pad 圈零 (np.pad), 再双循环 (允许循环!)。
    测试时会与 F.conv2d 对拍, 包括 stride=2, pad=1 的情形。
    """
    raise NotImplementedError


# 2. 参数量公式 -------------------------------------------------------------------
def conv_params(kh: int, kw: int, c_in: int, c_out: int) -> int:
    """一个卷积层的参数量: 核 + 偏置 (README §2 的公式)。"""
    raise NotImplementedError


def fc_params(n_in: int, n_out: int) -> int:
    """全连接层参数量: 权重 + 偏置。"""
    raise NotImplementedError


# 3. Pool vs Stride ----------------------------------------------------------------
class CNNPool(nn.Module):
    """标准版: conv → relu → maxpool (即 train_mnist.py 的 CNN)。"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.fc(torch.flatten(x, 1))


class CNNStride(nn.Module):
    """现代版: 用 stride=2 的卷积做下采样 (kernel=4, pad=1)。"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)          # 28→28
        self.s1 = nn.Conv2d(8, 8, 4, stride=2, padding=1)  # 28→14 (可学习下采样)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)         # 14→14
        self.s2 = nn.Conv2d(16, 16, 4, stride=2, padding=1)  # 14→7
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.s1(F.relu(self.c1(x))))
        x = F.relu(self.s2(F.relu(self.c2(x))))
        return self.fc(torch.flatten(x, 1))


def quick_train(model: nn.Module, steps: int = 600, seed: int = 0) -> float:
    """已写好: 用 MNIST 前 4096 张快速训练, 返回测试准确率。"""
    from torchvision import datasets, transforms
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    tf = transforms.ToTensor()
    ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    X = ds.data[:4096].float()[:, None] / 255.0
    y = ds.targets[:4096]
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    Xte, yte = test.data.float()[:, None] / 255.0, test.targets

    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        idx = rng.integers(0, len(X), 64)
        loss = F.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    return float((pred == yte).float().mean())


def pool_vs_stride(steps: int = 600) -> dict:
    """分别训练 CNNPool 与 CNNStride (quick_train), 返回
    {"pool": acc, "stride": acc, "pool_params": n, "stride_params": n}。
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    # 1 conv2d 对拍
    try:
        ok = True
        for (stride, pad) in [(1, 0), (2, 0), (1, 1), (2, 1)]:
            x = rng.normal(size=(9, 9))
            k = rng.normal(size=(3, 3))
            mine = conv2d_np(x, k, stride=stride, pad=pad)
            ref = F.conv2d(torch.tensor(x)[None, None],
                           torch.tensor(k)[None, None],
                           stride=stride, padding=pad).squeeze().numpy()
            ok = ok and mine.shape == ref.shape and np.allclose(mine, ref, atol=1e-8)
    except NotImplementedError:
        ok = False
    passed.append(("conv2d_np (stride/pad)", ok))

    # 2 参数公式
    try:
        ok = (
            conv_params(3, 3, 1, 16) == 16 * 1 * 3 * 3 + 16
            and conv_params(3, 3, 16, 32) == 32 * 16 * 3 * 3 + 32
            and fc_params(784, 128) == 784 * 128 + 128
        )
    except NotImplementedError:
        ok = False
    passed.append(("conv_params / fc_params", ok))

    # 3 pool vs stride
    try:
        res = pool_vs_stride(steps=400)
        ok = (
            res["pool_params"] == sum(p.numel() for p in CNNPool().parameters())
            and res["stride_params"] == sum(p.numel() for p in CNNStride().parameters())
            and min(res["pool"], res["stride"]) > 0.90
        )
        print(f"    pool: acc={res['pool']:.2%}, params={res['pool_params']} | "
              f"stride: acc={res['stride']:.2%}, params={res['stride_params']}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("pool_vs_stride", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 进入 08_sequence_models/ 吧。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
