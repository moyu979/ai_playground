"""第 02 课练习。实现后运行 `python exercises.py` 自动测试。"""

import numpy as np


# 1. 动量梯度下降 --------------------------------------------------------------
def gradient_descent_momentum(grad, x0: np.ndarray, lr: float, beta: float = 0.9,
                              n_iter: int = 50) -> np.ndarray:
    """通用动量梯度下降 (README §6)。

    参数:
        grad:  函数, grad(x) -> np.ndarray, 返回 ∇L 在 x 处的梯度
        x0:    初始点 (任意维数组)
        lr, beta, n_iter

    返回:
        (n_iter+1, *x0.shape) 的数组, 每一步的 x (含初始点)。

    更新规则:
        v ← beta * v - lr * grad(x)
        x ← x + v
    """
    raise NotImplementedError


# 2. 多项式拟合 (正规方程 + 特征变换) -------------------------------------------
def fit_poly_closed_form(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """用 degree 次多项式拟合数据, 返回系数 (degree+1,)。

    模型: y ≈ c_0 + c_1 x + c_2 x² + ... + c_degree x^degree
    做法: 构造 Vandermonde 矩阵 V[i,k] = x_i^k (形状 (N, degree+1)),
          解正规方程 VᵀV c = Vᵀy  (用 np.linalg.solve)。
    """
    raise NotImplementedError


# 3. R² 分数 --------------------------------------------------------------------
def r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot
    SS_res = Σ (y_i - ŷ_i)²        (残差平方和)
    SS_tot = Σ (y_i - ȳ)²          (总方差)
    R²=1 完美; R²=0 等于只会预测均值; 负数比均值还差。
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    # 1 动量: 在狭长二次型 L(x)=0.5*(100 x1² + x2²) 上, 动量应比普通 GD 快
    def grad(v):
        return np.array([100.0 * v[0], v[1]])

    x_target = np.array([3.0, 4.0])
    shifted = lambda v: grad(v - x_target)  # noqa: E731

    path_m = gradient_descent_momentum(shifted, np.zeros(2), lr=0.01, beta=0.9, n_iter=80)
    # 无动量对照
    x = np.zeros(2)
    for _ in range(80):
        x = x - 0.01 * shifted(x)
    err_m = np.linalg.norm(path_m[-1] - x_target)
    err_plain = np.linalg.norm(x - x_target)
    ok = (
        path_m.shape == (81, 2)
        and np.allclose(path_m[0], 0.0)
        and err_m < 0.5
        and err_m < err_plain / 2  # 动量明显更快
    )
    passed.append(("gradient_descent_momentum (动量加速)", ok))

    # 2 多项式拟合 sin
    x = np.linspace(-np.pi, np.pi, 50)
    y = np.sin(x) + 0.05 * rng.normal(size=50)
    c = fit_poly_closed_form(x, y, degree=5)
    ok = c.shape == (6,)
    if ok:
        V = np.vander(x, 6, increasing=True)
        ok = np.mean((V @ c - y) ** 2) < 0.05  # 拟合误差应当很小
    passed.append(("fit_poly_closed_form", ok))

    # 3 R²
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ok = abs(r2_score(y, y) - 1.0) < 1e-12  # 完美预测 → 1
    ok &= abs(r2_score(y, np.full(4, y.mean()))) < 1e-12  # 均值预测 → 0
    passed.append(("r2_score", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(f" 全部 {len(passed)} 题通过! 进入 03_logistic_regression/ 吧。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。提示: 动量迭代写错常因 v 的更新顺序。")


if __name__ == "__main__":
    _test()
