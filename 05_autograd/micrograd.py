"""第 05 课 · micrograd 风格的标量自动微分引擎。

运行: python micrograd.py  (演示表达式求导 + 数值梯度检查)

致敬 karpathy/micrograd。核心只有两个动作:
  1. 前向时每个运算记录自己的输入和局部导数 (闭包 _backward)
  2. backward() 按逆拓扑序执行每个节点的 _backward
"""

from __future__ import annotations


class Value:
    """一个标量, 附带完整的求导记录。"""

    def __init__(self, data: float, _children: tuple = (), _op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------ 运算符
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad          # ∂(a+b)/∂a = 1
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad   # ∂(ab)/∂a = b
            other.grad += self.data * out.grad   # ∂(ab)/∂b = a

        out._backward = _backward
        return out

    def __pow__(self, k):
        """只支持常数幂 k (够用: 平方、倒数)。"""
        assert isinstance(k, (int, float)), "幂指数必须是常数"
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        out = Value(np.tanh(self.data), (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad   # tanh' = 1 - tanh²

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad            # exp' = exp

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------- 组合运算符 (由上面拼出)
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other**-1 if isinstance(other, Value) else 1 / other)

    def __rtruediv__(self, other):
        return other * self**-1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ------------------------------------------------------------ 反向传播
    def backward(self) -> None:
        """从本节点出发, 对计算图中所有节点累积 grad (README §2)。"""
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)               # 后序: 孩子先于父母入栈

        build(self)
        self.grad = 1.0                      # dL/dL = 1
        for v in reversed(topo):             # 逆拓扑序 = 从输出到输入
            v._backward()

    # ------------------------------------------------------------ 其他
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __hash__(self):
        return id(self)


import numpy as np  # noqa: E402  (tanh 用)


def numerical_grad(f, xs: list[float], eps: float = 1e-6) -> list[float]:
    """中心差分数值梯度 (第 1 课的工具, 标量版)。"""
    grads = []
    for i in range(len(xs)):
        xp = list(xs)
        xm = list(xs)
        xp[i] += eps
        xm[i] -= eps
        grads.append((f(*xp) - f(*xm)) / (2 * eps))
    return grads


def demo() -> None:
    print("=" * 62)
    print("演示 1: 一个小表达式的自动求导")
    print("=" * 62)
    # f(a, b) = tanh(a*b + b) * (a - 2)
    a, b = Value(1.5), Value(-0.8)
    L = (a * b + b).tanh() * (a - 2)
    L.backward()
    print(f"L = tanh(a*b + b) * (a - 2) 在 a=1.5, b=-0.8 处 = {L.data:.6f}")
    print(f"自动梯度: dL/da = {a.grad:.6f}, dL/db = {b.grad:.6f}")

    def f(a_, b_):
        return float(np.tanh(a_ * b_ + b_) * (a_ - 2))

    g = numerical_grad(f, [1.5, -0.8])
    print(f"数值梯度: dL/da = {g[0]:.6f}, dL/db = {g[1]:.6f}")
    assert abs(a.grad - g[0]) < 1e-5 and abs(b.grad - g[1]) < 1e-5
    print("两种梯度一致 ✓  (引擎正确性的第一次证明)")

    print()
    print("=" * 62)
    print("演示 2: 分叉变量的梯度累加 (为什么 _backward 里用 +=)")
    print("=" * 62)
    x = Value(2.0)
    y = x * x + x          # x 出现两次:  dy/dx = 2x + 1
    y.backward()
    print(f"y = x² + x 在 x=2: y={y.data}, dy/dx={x.grad} (解析值 {2 * 2 + 1})")
    assert abs(x.grad - 5.0) < 1e-9
    print("累加正确 ✓")


if __name__ == "__main__":
    demo()
