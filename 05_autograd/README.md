# 第 05 课 · 自动微分引擎 — 100 行代码造一个 PyTorch 心脏

> 🌐 配套互动网页：[05_autograd.html](../pages/05_autograd.html)（讲解精华 + 可玩 demo，无需任何安装）


> 上节课你亲手推了一遍反向传播，也体会到了它的痛：**换个结构就要重推**，
> 而且 `tanh'`、矩阵转置、除 $B$……每一处都可能写错。
> 本课把这件事自动化。造完这个引擎，`loss.backward()` 对你永远是透明的。
> 设计与实现致敬 Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)
> （配套视频 *The spelled-out intro to neural networks and backpropagation* 强烈推荐）。

## 1. 核心思想：把链式法则外包给计算图

任何损失——不管多复杂——都是由**一串初等运算**拼出来的：
加、乘、幂、tanh……每种运算自己知道局部导数：

| 运算 | 前向 | 局部导数 (∂out/∂每个输入) |
|---|---|---|
| 加 | $c = a+b$ | $\partial c/\partial a = 1$, $\partial c/\partial b = 1$ |
| 乘 | $c = a\cdot b$ | $\partial c/\partial a = b$, $\partial c/\partial b = a$ |
| 幂 | $c = a^k$ | $\partial c/\partial a = k a^{k-1}$ |
| tanh | $c = \tanh a$ | $\partial c/\partial a = 1 - c^2$ |

**链式法则**：损失对某变量的导数 = （损失对该运算输出的导数）×（该运算对输入的局部导数），
沿所有路径求和。

所以每个运算只需做两件事：
1. 前向：算出输出值，**记住自己吃过哪些输入**
2. 被反传调用时：`输入.grad += 上游梯度 × 局部导数`

剩下的调度（谁先谁后）交给**拓扑排序**：按计算图从输出往输入的逆拓扑序，
保证处理每个节点时它的上游梯度已经齐了。

## 2. `Value` 类：会自动求导的"数字"

```python
class Value:
    data        # 数值本身
    grad        # ∂L/∂self, 反传时累积
    _prev       # 我的输入节点 (孩子)
    _backward   # 闭包: 把 out.grad 分发给孩子
```

**为什么是 `+=`？** 一个变量常被多处使用（分叉）：
`c = a+b; d = a*b` 中 $a$ 有两条路到 $L$，总梯度 = 两条路之和
（多元链式法则）。所以每次反传经过 $a$ 都累加，初始化为 0。

**一次完整的 `backward()`**（`micrograd.py` 第 60~75 行）：

```python
topo = []                       # 后序 DFS → 拓扑序
visited = set()
def build(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev: build(child)
        topo.append(v)
build(self)                     # topo: 输入在前, 输出在后
self.grad = 1.0                 # dL/dL = 1
for v in reversed(topo):        # 从输出往输入
    v._backward()               # 把 v.grad 分发给 v 的孩子们
```

## 3. 它和第 4 课是什么关系？

| | 第 4 课（手写 numpy MLP） | 第 5 课（autograd 引擎） |
|---|---|---|
| 反传谁 | 写死的层序列 | **任意**由这些运算构成的表达式 |
| 链式法则 | 你推导的 (公式 4~8) | 拓扑序自动执行 |
| 加新结构 | 重推公式 | 直接写前向，梯度免费 |
| 代价 | 快（矩阵乘，BLAS） | 慢（标量运算，教学用） |

PyTorch = 这个思想 + 矩阵级运算 + GPU。你第 4 课推导的四个公式，
在这里退化成 `add/mul/tanh` 三行的 `_backward` 闭包——**数学并没有变**。

## 4. 运行

```bash
python micrograd.py        # 演示: 表达式求导 + 数值梯度检查
python mlp_micrograd.py    # 用引擎搭 MLP 训练双月, 画决策边界
python exercises.py
```

## 练习

1. **组合运算**：用现有运算写出 sigmoid $1/(1+e^{-v})$，数值验证其梯度。
   （体会"新函数不需要改引擎"——这正是自动微分和手推的本质区别。）
2. **MSE 损失**：用 Value 写均方误差，对一个 toy 回归问题反传。
3. **重演第 2 课**：用引擎从零做线性回归 $y = wx+b$（梯度全靠引擎），
   3 分钟内你会看到"声明式求导"的威力。

## 预告

引擎虽好，但标量运算太慢。第 6 课切换 PyTorch：同样的思想，
张量级运算 + GPU + 现成的层。你会看到本课 `Value` 与 `torch.Tensor` 的
一一对应——以及它快了 1000 倍的原因。
