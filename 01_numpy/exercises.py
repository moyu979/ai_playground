"""第 01 课练习。

每个函数对应一题。把 `raise NotImplementedError` 换成你的实现，
然后运行 `python exercises.py` 自动测试。全部通过会打印祝贺信息。

规则: 全部用向量化实现, 不许写显式 for 循环 (除了被明确允许的地方)。
做完或卡住 15 分钟以上, 再看 solutions.py。
"""

import numpy as np


# 1. softmax ---------------------------------------------------------------
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax。

    softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
    提示: 先减去沿 axis 的最大值 (平移不改变 softmax, 但防 exp 上溢)。
    第 3 课它会变成分类模型, 第 10 课它是 GPT 输出的概率分布。
    """
    raise NotImplementedError


# 2. 行归一化 ----------------------------------------------------------------
def normalize_rows(X: np.ndarray) -> np.ndarray:
    """把 X 的每一行变成单位向量 (L2 范数 = 1)。

    提示: np.linalg.norm(X, axis=1, keepdims=True) 得到 (N,1),
    再靠广播相除。范数接近 0 时返回原行 (用 np.where 或加小量均可)。
    """
    raise NotImplementedError


# 3. one-hot 编码 -------------------------------------------------------------
def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """labels: (N,) 整数数组 (0..num_classes-1) → 返回 (N, num_classes) 的 one-hot。

    提示: 花式索引一行可解: 先建 zeros, 再用 np.arange(N) 和 labels 联合索引。
    """
    raise NotImplementedError


# 4. 两两距离矩阵 (广播的经典应用) ---------------------------------------------
def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A: (N, d), B: (M, d) → 返回 (N, M) 矩阵 D, D[i,j] = ||A_i - B_j||²。

    不许写双重循环。提示:
      ||a - b||² = ||a||² + ||b||² - 2 a·b
    右边三项都能用广播/矩阵乘一次算完。
    """
    raise NotImplementedError


# 5. einsum 练习 ---------------------------------------------------------------
def gram_matrix(X: np.ndarray) -> np.ndarray:
    """X: (N, d) → Gram 矩阵 G = X Xᵀ, 形状 (N, N)。
    要求: 用 np.einsum 写指标记号 (不得使用 @)。
    """
    raise NotImplementedError


# 6. 准确率 ---------------------------------------------------------------------
def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """logits: (N, num_classes) 未归一化得分, labels: (N,) 正确类别。
    返回 argmax 预测的正确率 (0~1 的 float)。
    """
    raise NotImplementedError


# 7. 数值梯度 (后面每课做梯度检查都靠它!) ----------------------------------------
def finite_difference_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """用中心差分数值计算标量函数 f: R^d → R 在 x 处的梯度。

    grad_i ≈ [f(x + eps·e_i) - f(x - eps·e_i)] / (2 eps)
    这个函数允许写循环 (逐个扰动每个坐标, 共 d 次前向)。
    它很慢, 但它『正确』——第 4 课我们用它检查手推的反向传播。
    """
    raise NotImplementedError


# 8. mini-batch 迭代器 ------------------------------------------------------------
def batch_iterator(X: np.ndarray, batch_size: int, rng: np.random.Generator):
    """把 (N, d) 的 X 切成若干个 (batch_size, d) 的批次并打乱顺序。

    要求:
    - 每个 epoch 内样本不重复 (先 permutation, 再切)
    - 最后不满一批的尾巴也要 yield 出去
    - 这是一个生成器 (用 yield), 训练循环里 `for xb in batch_iterator(...)`
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    # 1 softmax
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    s = softmax(z, axis=-1)
    ok = np.allclose(s.sum(axis=-1), 1.0) and np.all(np.isfinite(s)) and np.allclose(
        s[1], s[0]
    )
    passed.append(("softmax (含大数稳定性)", ok))

    # 2 normalize_rows
    X = rng.normal(size=(4, 3)) * 10
    Xn = normalize_rows(X)
    ok = Xn.shape == (4, 3) and np.allclose(np.linalg.norm(Xn, axis=1), 1.0)
    passed.append(("normalize_rows", ok))

    # 3 one_hot
    y = np.array([0, 2, 1, 2])
    oh = one_hot(y, 3)
    ok = oh.shape == (4, 3) and np.allclose(oh.sum(1), 1.0) and np.allclose(oh[1], [0, 0, 1])
    passed.append(("one_hot", ok))

    # 4 pairwise_sqdist
    A = rng.normal(size=(7, 3))
    B = rng.normal(size=(5, 3))
    D = pairwise_sqdist(A, B)
    D_ref = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    ok = D.shape == (7, 5) and np.allclose(D, D_ref, atol=1e-8)
    passed.append(("pairwise_sqdist (广播)", ok))

    # 5 gram
    X = rng.normal(size=(6, 4))
    ok = np.allclose(gram_matrix(X), X @ X.T, atol=1e-10)
    passed.append(("gram_matrix (einsum)", ok))

    # 6 accuracy
    logits = np.array([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [9.0, 0.0, 0.0]])
    labels = np.array([0, 1, 2])
    ok = abs(accuracy(logits, labels) - 2 / 3) < 1e-12
    passed.append(("accuracy", ok))

    # 7 数值梯度
    f = lambda v: float(np.sum(np.sin(v) * v**2))  # noqa: E731
    x = rng.normal(size=5)
    g_num = finite_difference_grad(f, x)
    g_ana = np.cos(x) * x**2 + np.sin(x) * 2 * x
    ok = np.allclose(g_num, g_ana, atol=1e-6)
    passed.append(("finite_difference_grad", ok))

    # 8 batch iterator
    X = np.arange(70, dtype=float).reshape(70, 1)
    seen = []
    for xb in batch_iterator(X, batch_size=16, rng=rng):
        seen.append(xb)
    all_rows = np.concatenate(seen, axis=0).ravel().astype(int)
    ok = (
        len(seen) == 5  # 4 批 16 + 1 批 6
        and len(all_rows) == 70
        and set(all_rows) == set(range(70))
        and seen[0].shape[0] == 16
        and seen[-1].shape[0] == 6
    )
    passed.append(("batch_iterator", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(f" 全部 {len(passed)} 题通过! 进入 02_gradient_descent/ 吧。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。继续加油, 卡住可以看提示或问我。")


if __name__ == "__main__":
    _test()
