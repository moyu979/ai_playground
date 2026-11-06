"""第 10 课练习 · 参考答案。"""

import numpy as np


def top_p_sample(logits: np.ndarray, p: float, rng: np.random.Generator) -> int:
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    order = np.argsort(probs)[::-1]              # 从大到小
    cum = np.cumsum(probs[order])
    cutoff = int(np.searchsorted(cum, p) + 1)    # 首次 ≥ p 的位置
    keep = order[:cutoff]
    q = probs[keep] / probs[keep].sum()
    return int(rng.choice(keep, p=q))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step < total:
        progress = (step - warmup) / (total - warmup)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    return base_lr * 0.05


if __name__ == "__main__":
    import exercises

    exercises.top_p_sample = top_p_sample
    exercises.gelu = gelu
    exercises.lr_at = lr_at
    exercises._test()
