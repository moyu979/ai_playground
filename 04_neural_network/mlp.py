"""第 04 课 · 纯 NumPy 多层感知机 (反向传播手工实现)。

每个类/方法都标注了对应 README 的公式编号。
PyTorch 版对照见 ../06_pytorch/mlp_pytorch.py —— 结构一一对应。
"""

import numpy as np


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class Dense:
    """全连接层: forward = x @ W + b (README §2)。

    权重 W 形状 (n_in, n_out), 行向量约定 → 代码即公式。
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 init: str = "he"):
        # README §5: 初始化的方差决定信号能不能活着穿过网络
        if init == "he":        # ReLU 配套
            std = np.sqrt(2.0 / n_in)
        elif init == "xavier":  # tanh 配套
            std = np.sqrt(1.0 / n_in)
        elif init == "big":     # 反面教材: 方差爆炸
            std = 1.0
        elif init == "zeros":   # 反面教材: 对称性永不破缺
            std = 0.0
        else:
            raise ValueError(f"unknown init {init}")
        self.W = rng.normal(0.0, std, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x                      # 反传要用 a^(l-1), 缓存 (公式 7)
        return x @ self.W + self.b

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """输入 δ^(l) = ∂L/∂z^(l), 形状 (B, n_out)。返回 ∂L/∂a^(l-1)。"""
        self.dW = self._x.T @ dz         # 公式 (7): (n_in,B)(B,n_out)
        self.db = dz.sum(axis=0)         # 公式 (8): (n_out,)
        return dz @ self.W.T             # δ^(l-1) 的矩阵乘部分, 交给上层

    def step(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db

    @property
    def n_params(self) -> int:
        return self.W.size + self.b.size


class Activation:
    """逐元素激活函数 φ。backward 用 φ' (公式 6 的 ⊙ 因子)。"""

    def __init__(self, name: str):
        if name not in ("tanh", "relu"):
            raise ValueError(name)
        self.name = name
        self._a: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._a = np.tanh(z) if self.name == "tanh" else np.maximum(z, 0.0)
        return self._a

    def backward(self, da: np.ndarray) -> np.ndarray:
        if self.name == "tanh":
            return da * (1.0 - self._a**2)      # φ'(z) = 1 - tanh²
        return da * (self._a > 0.0)             # ReLU': 正区 1, 其余 0

    def step(self, lr: float) -> None:
        pass  # 无参数

    @property
    def n_params(self) -> int:
        return 0


class MLP:
    """多层感知机 = [Dense, Activation, ..., Dense] + softmax 交叉熵。

    用法:
        net = MLP([2, 32, 32, 3], rng)
        loss, P = net.loss_and_grad(X, y)   # 前向 + 反向 (梯度已存在层内)
        net.step(lr)                        # SGD 更新
    """

    def __init__(self, sizes: list[int], rng: np.random.Generator,
                 activation: str = "tanh", init: str = "he"):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Dense(sizes[i], sizes[i + 1], rng, init=init))
            if i < len(sizes) - 2:                       # 输出层后不加激活
                layers.append(Activation(activation))
        self.layers = layers
        self.n_params = sum(l.n_params for l in layers)

    # ------------------------------------------------------------- 前向
    def forward(self, X: np.ndarray) -> np.ndarray:
        """返回 logits (B, K), 未过 softmax。"""
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    # --------------------------------------------------- 损失 + 反向
    def loss_and_grad(self, X: np.ndarray, y: np.ndarray):
        """前向 + 反向一次完成。返回 (loss, 概率 P)。

        梯度已存进每个 Dense 的 dW/db, 调用 step(lr) 即可更新。
        """
        logits = self.forward(X)                          # (B, K)
        P = softmax(logits)                               # (B, K)
        B = len(X)
        loss = float(-np.mean(np.log(P[np.arange(B), y] + 1e-12)))

        # ---- 反向 (README §4) ----
        Y = np.eye(P.shape[1])[y]                         # one-hot (B, K)
        delta = (P - Y) / B                               # 公式 (4): δ^L = (P-Y)/B
        for layer in reversed(self.layers):
            delta = layer.backward(delta)                 # 逐层往回传 (公式 6/7/8)
        return loss, P

    # ------------------------------------------------------------- 更新
    def step(self, lr: float) -> None:
        for layer in self.layers:
            layer.step(lr)

    # ------------------------------------------------------------- 推理
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
