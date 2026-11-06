"""第 07 课练习 · 参考答案。"""

import numpy as np
import torch

from exercises import CNNPool, CNNStride, quick_train


def conv2d_np(x: np.ndarray, k: np.ndarray, stride: int = 1,
              pad: int = 0) -> np.ndarray:
    if pad > 0:
        x = np.pad(x, pad)
    Kh, Kw = k.shape
    H, W = x.shape
    H_out = (H - Kh) // stride + 1
    W_out = (W - Kw) // stride + 1
    out = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i * stride : i * stride + Kh, j * stride : j * stride + Kw]
            out[i, j] = np.sum(patch * k)
    return out


def conv_params(kh: int, kw: int, c_in: int, c_out: int) -> int:
    return c_out * c_in * kh * kw + c_out


def fc_params(n_in: int, n_out: int) -> int:
    return n_in * n_out + n_out


def pool_vs_stride(steps: int = 600) -> dict:
    acc_pool = quick_train(CNNPool(), steps=steps)
    acc_stride = quick_train(CNNStride(), steps=steps)
    return {
        "pool": acc_pool,
        "stride": acc_stride,
        "pool_params": sum(p.numel() for p in CNNPool().parameters()),
        "stride_params": sum(p.numel() for p in CNNStride().parameters()),
    }


if __name__ == "__main__":
    import exercises

    exercises.conv2d_np = conv2d_np
    exercises.conv_params = conv_params
    exercises.fc_params = fc_params
    exercises.pool_vs_stride = pool_vs_stride
    exercises._test()
