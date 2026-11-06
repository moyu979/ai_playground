"""第 03 课练习。实现后运行 `python exercises.py` 自动测试。"""

import numpy as np


# 1. 数值稳定的 softmax ---------------------------------------------------------
def softmax_stable(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax (README §5): 先减去沿 axis 的最大值再 exp。

    验证标准: 输入含 1e4 量级时仍有限, 且每行和为 1。
    """
    raise NotImplementedError


# 2. log-sum-exp 交叉熵 ----------------------------------------------------------
def cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """多分类交叉熵, 但不许先算概率再取 log。

    logits: (N, K) 未归一化分数, labels: (N,) 正确类别。
    数学: L_i = -( z_{y_i} - log Σ_j e^{z_j} )
    实现: 对每行计算 logsumexp = m + log Σ_j e^{z_j - m} (m 为行最大值),
          然后 L = mean(logsumexp - logits[i, y_i])。
    这是大型语言模型算损失的真正方式 (第 10 课同款)。
    """
    raise NotImplementedError


# 3. 梯度检查: ∂L/∂z = p - y -----------------------------------------------------
def grad_z(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """交叉熵对 logits 的解析梯度 (softmax 之后): 应该等于 p - onehot(y)。

    实现: p = softmax(logits); 返回 (p - onehot) / N
    (除 N 是因为损失取了样本平均)。
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def finite_difference_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """中心差分数值梯度 (第 1 课的工具, 这里内置一份)。"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (f(xp) - f(xm)) / (2 * eps)
    return grad


def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    # 1 稳定 softmax
    z = np.array([[1.0, 2.0, 3.0], [10000.0, 10001.0, 10002.0]])
    s = softmax_stable(z)
    ok = np.all(np.isfinite(s)) and np.allclose(s.sum(-1), 1.0) and np.allclose(s[0], s[1])
    passed.append(("softmax_stable (大数不溢出)", ok))

    # 2 LSE 交叉熵 vs 朴素实现
    logits = rng.normal(size=(40, 5)) * 3
    labels = rng.integers(0, 5, size=40)
    p = np.clip(np.exp(logits) / np.exp(logits).sum(1, keepdims=True), 1e-30, 1)
    naive = float(-np.mean(np.log(p[np.arange(40), labels])))
    ok = abs(cross_entropy_from_logits(logits, labels) - naive) < 1e-8
    # 大数时朴素法溢出而 LSE 法存活
    big = np.array([[1000.0, 0.0, -1000.0]])
    ok = ok and np.isfinite(cross_entropy_from_logits(big, np.array([0])))
    passed.append(("cross_entropy_from_logits (LSE)", ok))

    # 3 解析梯度 vs 数值梯度
    logits = rng.normal(size=(6, 4))
    labels = rng.integers(0, 4, size=6)

    def loss_fn(z_flat):
        L = cross_entropy_from_logits(z_flat.reshape(6, 4), labels)
        return float(L)

    g_num = finite_difference_grad(loss_fn, logits.ravel()).reshape(6, 4)
    g_ana = grad_z(logits, labels)
    ok = np.allclose(g_num, g_ana, atol=1e-6)
    passed.append(("grad_z = p - onehot (数值验证)", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 你已经握着第 4 课反向传播的全部钥匙了。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
