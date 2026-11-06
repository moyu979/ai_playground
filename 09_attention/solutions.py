"""第 09 课练习 · 参考答案。"""

import torch
import torch.nn.functional as F


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, T, d = x.shape
    return x.view(B, T, n_heads, d // n_heads).transpose(1, 2)


def merge_heads(y: torch.Tensor) -> torch.Tensor:
    B, H, T, dh = y.shape
    return y.transpose(1, 2).reshape(B, T, H * dh)


def attention_weights(q: torch.Tensor, k: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return F.softmax(scores, dim=-1)


def saturation_experiment(d: int, n: int = 500, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, d, generator=g)
    k = torch.randn(n, d, generator=g)
    s = q @ k.T
    return {
        "raw": float(F.softmax(s, -1).max(-1).values.mean()),
        "scaled": float(F.softmax(s / d**0.5, -1).max(-1).values.mean()),
    }


if __name__ == "__main__":
    import exercises

    exercises.split_heads = split_heads
    exercises.merge_heads = merge_heads
    exercises.attention_weights = attention_weights
    exercises.saturation_experiment = saturation_experiment
    exercises._test()
