"""第 03 课练习 · 参考答案。"""

import numpy as np


def softmax_stable(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    m = logits.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    return float(np.mean(lse - logits[np.arange(len(labels)), labels]))


def grad_z(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    N, K = logits.shape
    p = softmax_stable(logits)
    onehot = np.eye(K)[labels]
    return (p - onehot) / N


if __name__ == "__main__":
    import exercises

    exercises.softmax_stable = softmax_stable
    exercises.cross_entropy_from_logits = cross_entropy_from_logits
    exercises.grad_z = grad_z
    exercises._test()
