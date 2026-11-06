# 第 01 课 · NumPy 速成 — 为深度学习定制的 90 分钟

> 🌐 配套互动网页：[01_numpy.html](../pages/01_numpy.html)（讲解精华 + 可玩 demo，无需任何安装）


> 物理人视角：NumPy 就是你的"张量计算器"。以后 PyTorch 的 Tensor 几乎是它的超集，
> 这 90 分钟的投入在后面每一课都会复利。

## 为什么是 NumPy

你写过的物理模拟大概长这样：对一个场 `f[x][y]` 做演化，写双重循环。
NumPy 的思维方式是反过来的：**不问"每个格点怎么算"，只问"整个场怎么一次性变换"**。
这就是**向量化（vectorization）**——把循环下沉到 C/Fortran/BLAS 层。

深度学习 = 大规模矩阵乘法 + 非线性变换。模型参数是矩阵，数据是矩阵，梯度还是矩阵。
NumPy 熟练度直接决定你读代码的速度。

## 1. ndarray：一切皆数组

```python
import numpy as np

a = np.array([[1., 2., 3.],
              [4., 5., 6.]])   # shape (2, 3) 的二维数组 = 矩阵
a.shape      # (2, 3)  ← 想成"张量的量纲"
a.dtype      # float64 ← 默认;  深度学习几乎总用 float32
a.ndim       # 2
```

- 物理里你区分标量/矢量/张量；NumPy 里它们统一是 `ndarray`，只是 `shape` 不同：
  标量 `()`，矢量 `(3,)`，矩阵 `(3, 3)`，三阶张量 `(3, 3, 3)`。
- **约定（本课程全程遵守）**：数据批次的第一维是 batch。
  一个 batch 的输入是 `(B, d)`，B 个样本、每个样本 d 维。

## 2. 广播（broadcasting）— 最重要的一条规则

两个数组运算时，NumPy 从**最后一个维度向前**对齐，维度相等或其中一方为 1 时兼容，
为 1 的那维自动"复制"（虚拟地，不花内存）：

```
A (2, 3)      B (3,)    →  B 视为 (1, 3) → 复制为 (2, 3)   ✓
A (2, 3)      C (2, 1)  →  C 复制为 (2, 3)                  ✓
A (2, 3)      D (2,)    →  (2,) 对齐到 (3,) 失败            ✗ 报错
```

经典用法（后面天天见）：

```python
X -= X.mean(axis=0)          # 每列去均值 (数据标准化)
X /= X.std(axis=0) + 1e-8    # 每列除以标准差; +1e-8 防除零, 神经网络代码的经典后缀
logits += np.array([-2., 1., 3.])   # batch 中每个样本都加同一个偏置
```

## 3. 索引：切片、掩码、花式索引

```python
X[0]          # 第 0 行 (第 0 个样本)
X[:, 0]       # 所有行的第 0 列  ← 注意别写成 X[0] 后再[0]
X[mask]       # 布尔掩码: 取出 mask 为 True 的行 —— 筛数据全靠它
X[[2, 0, 1]]  # 花式索引: 按 (2,0,1) 的顺序重排行 —— 打乱数据集全靠它
```

## 4. einsum — 物理人的母语

`np.einsum` 直接用指标记号写张量运算，你在 QFT/广义相对论里写过的东西原样可用：

```python
C = np.einsum('ij,jk->ik', A, B)     # 矩阵乘: C_ik = Σ_j A_ij B_jk
s = np.einsum('i,i->',  v,  v)       # 内积:   s   = Σ_i v_i v_i
O = np.einsum('ij,kj->ik', A, B)     # A @ B.T: O_ik = Σ_j A_ij B_kj
```

规则就是：重复指标求和，箭头右边没出现的指标被"吃掉"。
第 9 课的注意力机制 `softmax(QKᵀ/√d)V` 用 einsum 写一行就出来。

## 5. 随机数与可复现性

```python
rng = np.random.default_rng(seed=42)   # 新式写法; 永远带上 seed, 实验才可复现
rng.normal(0, 1, size=(100, 2))        # 高斯噪声
rng.uniform(-1, 1, size=(3, 3))        # 均匀分布 (初始化权重用)
rng.permutation(100)                   # 打乱索引
```

## 6. 向量化思维的两个模板

**模板 A：逐样本循环 → 一次算完**

```python
# ❌ 慢: 对每个样本单独算距离
d = np.array([np.linalg.norm(x - c) for x in X])

# ✓ 快: 广播 (N,1,d) - (1,d) → (N,d), 一次算出 N 个距离
d = np.linalg.norm(X - c, axis=1)
```

**模板 B：手写公式 → 矩阵形式**

最小二乘的正规方程你肯定眼熟：

$$\theta = (X^T X)^{-1} X^T y$$

NumPy 里就是一行 `theta = np.linalg.solve(X.T @ X, X.T @ y)`
（生产代码用 `solve` 而不是 `inv` 再乘——数值更稳定，和物理里避免显式求逆同理）。

## 运行本课代码

```bash
python numpy_crash.py    # 交互式演示: 每一节都会打印结果并生成图
python exercises.py      # 练习: 把 NotImplementedError 补全, 自动测试
python solutions.py      # 参考答案 (先自己做!)
```

## 练习（exercises.py，约 8 题）

覆盖：softmax（第 3 课主角）、行归一化、one-hot、**广播算两两距离**、einsum 算
Gram 矩阵、accuracy、**数值梯度**（第 4 课梯度检查的工具）、mini-batch 迭代器
（每课训练循环都要用）。

做完这些，你写后面所有课的代码都不会被 NumPy 卡住。
