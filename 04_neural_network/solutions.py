"""第 04 课练习 · 参考答案。"""

import numpy as np

from exercises import get_flat_params, make_loss_fn, set_flat_params
from mlp import Dense, MLP


def grad_check(net: MLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    # 1) 解析梯度: 前向+反向一次
    net.loss_and_grad(X, y)
    g_ana = []
    for l in net.layers:
        if isinstance(l, Dense):
            g_ana.append(l.dW.ravel())
    for l in net.layers:
        if isinstance(l, Dense):
            g_ana.append(l.db)
    g_ana = np.concatenate(g_ana)

    # 2) 数值梯度 (中心差分)
    f = make_loss_fn(net, X, y)
    theta = get_flat_params(net)
    g_num = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += eps
        tm[i] -= eps
        g_num[i] = (f(tp) - f(tm)) / (2 * eps)

    # 3) 相对误差
    return float(np.max(np.abs(g_num - g_ana)
                        / np.maximum(1e-8, np.abs(g_num) + np.abs(g_ana))))


def init_experiment(X, y, lr=0.5, n_steps=3000, rng=None):
    rng = rng or np.random.default_rng(0)
    losses = {}
    for init in ("zeros", "big", "he"):
        net = MLP([2, 16, 16, 16, 3], rng=rng, activation="tanh", init=init)
        hist = []
        for _ in range(n_steps + 1):
            loss, _ = net.loss_and_grad(X, y)   # 全批量
            net.step(lr)
            hist.append(loss)
        losses[init] = np.array(hist)
    return losses


def activation_experiment(X, y, n_steps=6000, rng=None):
    rng = rng or np.random.default_rng(0)
    accs = {}
    for act in ("tanh", "relu"):
        net = MLP([2, 64, 64, 3], rng=rng, activation=act, init="he")
        for _ in range(n_steps):
            idx = rng.integers(0, len(X), 32)
            loss, _ = net.loss_and_grad(X[idx], y[idx])
            net.step(0.5)
        accs[act] = net.accuracy(X, y)
    return accs


if __name__ == "__main__":
    import exercises

    exercises.grad_check = grad_check
    exercises.init_experiment = init_experiment
    exercises.activation_experiment = activation_experiment
    exercises._test()
