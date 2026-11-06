"""第 04 课练习。实现后运行 `python exercises.py` 自动测试。

第 1 题 (梯度检查) 是全课程最重要的练习, 请务必亲手完成——
它教会你"如何证明自己的反传代码是对的", 这个技能你会用一辈子。
"""

import numpy as np

from datasets import make_moons, make_spirals
from mlp import Dense, MLP


# ---------------------------------------------------------------- 辅助函数 (已写好)
def get_flat_params(net: MLP) -> np.ndarray:
    """把网络所有 Dense 参数摊平成一个向量。"""
    parts = [l.W.ravel() for l in net.layers if isinstance(l, Dense)]
    parts += [l.b.ravel() for l in net.layers if isinstance(l, Dense)]
    return np.concatenate(parts)


def set_flat_params(net: MLP, theta: np.ndarray) -> None:
    i = 0
    for l in net.layers:
        if isinstance(l, Dense):
            n = l.W.size
            l.W = theta[i : i + n].reshape(l.W.shape).copy()
            i += n
    for l in net.layers:
        if isinstance(l, Dense):
            n = l.b.size
            l.b = theta[i : i + n].copy()
            i += n


def make_loss_fn(net: MLP, X: np.ndarray, y: np.ndarray):
    """返回 f(theta): 给定摊平参数, 前向一遍算损失 (不反传)。"""

    def f(theta: np.ndarray) -> float:
        set_flat_params(net, theta)
        logits = net.forward(X)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return float(-np.mean(np.log(p[np.arange(len(X)), y] + 1e-12)))

    return f


# ---------------------------------------------------------------- 第 1 题
def grad_check(net: MLP, X: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """手推梯度 vs 数值梯度, 返回最大相对误差。

    步骤:
    1. loss_and_grad 一次, 收集每个 Dense 的 dW/db (拼成大向量 g_ana)
    2. 摊平参数 theta, 对每个坐标 i:
         g_num[i] = [f(theta+eps·e_i) - f(theta-eps·e_i)] / (2 eps)
    3. rel_err[i] = |g_num-g_ana| / max(1e-8, |g_num|+|g_ana|)
    返回 max(rel_err)。目标: < 1e-6。
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 第 2 题
def init_experiment(X, y, lr=0.5, n_steps=3000,
                    rng: np.random.Generator | None = None) -> dict:
    """对三种初始化各训练一个网络 (结构 [2, 16, 16, 16, 3], tanh, 全批量)。

    数据请用螺旋 (够难才分得出初始化好坏)。
    返回 {"zeros": losses数组, "big": ..., "he": ...}
    每个数组长度 n_steps+1, 记录每步的损失。
    """
    raise NotImplementedError


# ---------------------------------------------------------------- 第 3 题
def activation_experiment(X, y, n_steps=6000,
                          rng: np.random.Generator | None = None) -> dict:
    """对比 tanh 与 relu (结构 [2, 64, 64, 3], init="he", mini-batch 32)。

    返回 {"tanh": 最终准确率, "relu": 最终准确率}
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    # 1 梯度检查
    X, y = make_moons(40, rng=rng)
    net = MLP([2, 4, 3], rng=rng, activation="tanh", init="he")
    try:
        err = grad_check(net, X, y)
        ok = err < 1e-6
        print(f"    梯度最大相对误差 = {err:.2e}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("grad_check", ok))

    # 2 初始化实验 (螺旋 + 深网络, 初始化的差异才够明显)
    X, y = make_spirals(100, 3, rng=rng)
    try:
        losses = init_experiment(X, y, rng=rng)
        ln3 = np.log(3.0)
        ok = (
            set(losses) == {"zeros", "big", "he"}
            and abs(losses["zeros"][-1] - ln3) < 1e-3   # 全零: 死在 ln3
            and losses["he"][-1] < 0.5                  # He: 正常收敛
            and losses["he"][-1] < losses["big"][-1]    # 大方差: 更差
        )
        print(f"    zeros 末值 {losses['zeros'][-1]:.4f} (≈ln3={ln3:.4f}), "
              f"big {losses['big'][-1]:.4f}, he {losses['he'][-1]:.4f}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("init_experiment", ok))

    # 3 激活函数对比
    X, y = make_spirals(100, 3, rng=rng)
    try:
        accs = activation_experiment(X, y, rng=rng)
        ok = set(accs) == {"tanh", "relu"} and min(accs.values()) > 0.65
        print(f"    tanh acc={accs.get('tanh', 0):.1%}, relu acc={accs.get('relu', 0):.1%}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("activation_experiment", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 你的反传代码经过数值验证, 是『可证明正确』的。")
        print(" 进入 05_autograd/ 吧。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。梯度检查不过? 检查除 B 的位置和 tanh' 公式。")


if __name__ == "__main__":
    _test()
