"""第 01 课练习 · 参考答案。先自己做, 卡住了再看这里。"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)  # 平移不变性, 防上溢
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)  # (N, 1)
    return np.where(n > 1e-12, X / (n + 1e-12), X)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(labels), num_classes))
    out[np.arange(len(labels)), labels] = 1.0  # 花式索引
    return out


def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # ||a-b||² = ||a||² + ||b||² - 2a·b
    sq_a = (A**2).sum(axis=1)[:, None]        # (N, 1)
    sq_b = (B**2).sum(axis=1)[None, :]        # (1, M)
    return sq_a + sq_b - 2.0 * (A @ B.T)


def gram_matrix(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,kj->ik", X, X)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((np.argmax(logits, axis=1) == labels).mean())


def finite_difference_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def batch_iterator(X: np.ndarray, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        yield X[idx[start : start + batch_size]]


if __name__ == "__main__":
    import exercises

    # 用参考答案顶替练习里的占位实现, 再跑同一套测试
    exercises.softmax = softmax
    exercises.normalize_rows = normalize_rows
    exercises.one_hot = one_hot
    exercises.pairwise_sqdist = pairwise_sqdist
    exercises.gram_matrix = gram_matrix
    exercises.accuracy = accuracy
    exercises.finite_difference_grad = finite_difference_grad
    exercises.batch_iterator = batch_iterator
    exercises._test()
