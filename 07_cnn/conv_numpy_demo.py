"""第 07 课 · 手写卷积, 亲眼看到"边缘检测器"。

运行: python conv_numpy_demo.py
生成 figures/conv_kernels.png, 并与 torch.nn.functional.conv2d 对拍。
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def conv2d_naive(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """单通道互相关 (深度学习语境的"卷积")。

    x: (H, W), k: (Kh, Kw) → 输出 (H-Kh+1, W-Kw+1)
    纯教学实现: 双重循环 + 内积。慢, 但每一步都看得见。
    """
    Kh, Kw = k.shape
    H, W = x.shape
    out = np.zeros((H - Kh + 1, W - Kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i : i + Kh, j : j + Kw] * k)   # 逐点乘再求和
    return out


def make_image(n: int = 64) -> np.ndarray:
    """程序生成一张图: 渐变背景 + 圆 + 方块 (不依赖任何图片文件)。"""
    yy, xx = np.mgrid[0:n, 0:n]
    img = 0.3 + 0.4 * xx / n                                   # 水平渐变背景
    img[(xx - 20) ** 2 + (yy - 32) ** 2 < 13**2] = 0.95        # 一个圆
    mask = (np.abs(xx - 44) < 11) & (np.abs(yy - 30) < 11)
    img[mask] = 0.05                                           # 一个方块
    return img


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    img = make_image()

    # 三个手工核 —— 注意它们就是"特征检测器"
    kernels = {
        "竖直边缘 $K_v$": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], float),
        "水平边缘 $K_h$": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], float),
        "模糊 $K_b$": np.ones((3, 3)) / 9.0,
    }

    # ---- 与 PyTorch 对拍: 深度学习的 conv2d 就是这个操作 ----
    x_t = torch.tensor(img)[None, None]                        # (1,1,H,W)
    for name, k in kernels.items():
        ours = conv2d_naive(img, k)
        theirs = F.conv2d(x_t, torch.tensor(k)[None, None]).squeeze().numpy()
        assert np.allclose(ours, theirs, atol=1e-5), name
    print("手写 conv2d 与 torch.nn.functional.conv2d 完全一致 ✓")

    # ---- 速度对比: 为什么需要框架/GPU ----
    big = make_image(256)
    t0 = time.perf_counter()
    conv2d_naive(big, kernels["竖直边缘 $K_v$"])
    t_naive = time.perf_counter() - t0
    big_t = torch.tensor(big)[None, None]
    k_t = torch.tensor(kernels["竖直边缘 $K_v$"])[None, None]
    t0 = time.perf_counter()
    for _ in range(50):
        F.conv2d(big_t, k_t)
    t_torch = (time.perf_counter() - t0) / 50
    print(f"256×256 图, 3×3 核:  手写循环 {t_naive * 1000:.1f} ms | "
          f"torch {t_torch * 1000:.2f} ms (还没用 GPU)")
    print("真实网络的核是【学出来的】, 且一次并行几百个 → 必须 BLAS/GPU。")

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("原图 (程序生成)")
    for ax, (name, k) in zip(axes.ravel()[1:], kernels.items()):
        out = conv2d_naive(img, k)
        vmax = np.abs(out).max()
        ax.imshow(out, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name} 的响应")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("卷积 = 用小核扫描图像: 负响应蓝、正响应红\n"
                 "$K_v$ 只看竖边, $K_h$ 只看横边 —— 训练时这些核会被自动学出")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "conv_kernels.png"), dpi=130)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}/conv_kernels.png")


if __name__ == "__main__":
    main()
