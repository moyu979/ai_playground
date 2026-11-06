"""第 02 课练习 · 参考答案。"""

import numpy as np


def gradient_descent_momentum(grad, x0: np.ndarray, lr: float, beta: float = 0.9,
                              n_iter: int = 50) -> np.ndarray:
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(n_iter):
        v = beta * v - lr * grad(x)
        x = x + v
        path.append(x.copy())
    return np.array(path)


def fit_poly_closed_form(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    V = np.vander(x, degree + 1, increasing=True)  # V[i,k] = x_i^k
    return np.linalg.solve(V.T @ V, V.T @ y)


def r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


if __name__ == "__main__":
    import exercises

    exercises.gradient_descent_momentum = gradient_descent_momentum
    exercises.fit_poly_closed_form = fit_poly_closed_form
    exercises.r2_score = r2_score
    exercises._test()
