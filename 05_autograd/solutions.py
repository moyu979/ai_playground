"""第 05 课练习 · 参考答案。"""

import numpy as np

from micrograd import Value


def sigmoid(v: Value) -> Value:
    return 1 / (1 + (-v).exp())


def mse_loss(preds: list[Value], targets: list[float]) -> Value:
    terms = [(p - Value(t)) ** 2 for p, t in zip(preds, targets)]
    return sum(terms, Value(0.0)) * (1.0 / len(terms))


def fit_linear(x: np.ndarray, y: np.ndarray, n_iters: int = 300,
               lr: float = 0.1) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    w = Value(rng.normal(0, 0.1))
    b = Value(rng.normal(0, 0.1))
    for _ in range(n_iters):
        preds = [w * Value(xi) + b for xi in x]
        loss = mse_loss(preds, list(y))
        w.grad = 0.0
        b.grad = 0.0
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
    return w.data, b.data


if __name__ == "__main__":
    import exercises

    exercises.sigmoid = sigmoid
    exercises.mse_loss = mse_loss
    exercises.fit_linear = fit_linear
    exercises._test()
