"""第 05 课练习。实现后运行 `python exercises.py` 自动测试。

所有实现只能用引擎提供的运算 (Value + - * ** tanh exp relu),
不许手动写任何导数 —— 这就是本课的意义。
"""

import numpy as np

from micrograd import Value, numerical_grad


# 1. sigmoid (纯组合, 不改引擎) ---------------------------------------------------
def sigmoid(v: Value) -> Value:
    """用现有运算组合出 sigmoid(v) = 1 / (1 + exp(-v))。

    一行可解 (注意: 1/Value 会走 __rtruediv__)。
    完成后测试会数值验证它的梯度 —— 引擎自动给了它导数!
    """
    raise NotImplementedError


# 2. MSE 损失 -------------------------------------------------------------------
def mse_loss(preds: list[Value], targets: list[float]) -> Value:
    """mean_i (preds_i - targets_i)², 返回一个 Value。

    提示: 先逐个平方, sum, 再乘 1/len。sum 可以用 Python 的 sum + 起始 Value(0)。
    """
    raise NotImplementedError


# 3. 用引擎重演第 2 课: 线性回归 ---------------------------------------------------
def fit_linear(x: np.ndarray, y: np.ndarray, n_iters: int = 300,
               lr: float = 0.1) -> tuple[float, float]:
    """用 Value + SGD 拟合 y ≈ w*x + b, 返回 (w, b)。

    要求:
    - w, b 用 Value 随机初始化 (np.random.default_rng(0).normal(0, 0.1))
    - 每个 iter: 前向(全样本 MSE) → 清零梯度 → backward → 参数更新
    - 数据集小 (60 个点), 全批量即可
    测试标准: 数据由 w=2, b=1 生成, 你应拟回 (2, 1) 附近。
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    passed = []

    # 1 sigmoid 及其梯度
    v = Value(0.7)
    try:
        s = sigmoid(v)
        s.backward()
        g_auto = v.grad

        def f(x):
            return 1.0 / (1.0 + np.exp(-x))

        g_num = numerical_grad(f, [0.7])[0]
        ok = abs(s.data - f(0.7)) < 1e-9 and abs(g_auto - g_num) < 1e-6
        print(f"    sigmoid(0.7)={s.data:.4f}, dσ/dv 自动={g_auto:.6f} "
              f"数值={g_num:.6f}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("sigmoid (引擎自动求导)", ok))

    # 2 MSE
    try:
        preds = [Value(1.0), Value(2.0), Value(3.0)]
        targets = [1.5, 2.5, 2.0]
        L = mse_loss(preds, targets)
        assert abs(L.data - (0.25 + 0.25 + 1.0) / 3) < 1e-12
        L.backward()
        ok = abs(preds[0].grad - 2 * (1.0 - 1.5) / 3) < 1e-12
        print(f"    MSE={L.data:.4f}, ∂L/∂p₀={preds[0].grad:.4f}")
    except (NotImplementedError, AssertionError):
        ok = False
        print("    (未实现或数值不对)")
    passed.append(("mse_loss", ok))

    # 3 线性回归
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 60)
    y = 2.0 * x + 1.0 + rng.normal(0, 0.05, 60)
    try:
        w, b = fit_linear(x, y)
        ok = abs(w - 2.0) < 0.15 and abs(b - 1.0) < 0.15
        print(f"    拟合结果 w={w:.3f}, b={b:.3f} (真值 2, 1)")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("fit_linear (autograd 版)", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 你已经拥有一个 (迷你) PyTorch 了。进入 06_pytorch/")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
