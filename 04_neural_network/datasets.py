"""第 04 课 · 2D 玩具数据集 (不依赖 sklearn)。"""

import numpy as np


def make_moons(n: int = 200, noise: float = 0.15,
               rng: np.random.Generator | None = None):
    """两个交叠的半月。返回 X (n,2), y (n,) ∈ {0,1}。"""
    rng = rng or np.random.default_rng(0)
    m = n // 2
    t = rng.uniform(0, np.pi, m)
    upper = np.stack([np.cos(t), np.sin(t)], axis=1)            # 上半月
    lower = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)  # 下半月(平移)
    X = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
    return X, y


def make_circles(n: int = 200, noise: float = 0.08, factor: float = 0.4,
                 rng: np.random.Generator | None = None):
    """同心圆环 (内圈 vs 外圈), 线性不可分的经典案例。"""
    rng = rng or np.random.default_rng(0)
    m = n // 2
    ang = rng.uniform(0, 2 * np.pi, m)
    outer = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    inner = factor * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    X = np.concatenate([outer, inner]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(m, dtype=int), np.ones(m, dtype=int)])
    return X, y


def make_spirals(n_per_class: int = 120, n_classes: int = 3, noise: float = 0.2,
                 rng: np.random.Generator | None = None):
    """螺旋臂 (karpathy 同款)。三条臂缠绕, 是 MLP 能力的试金石。

    注意半径给到 4: 臂间径向间距 ≈ (2π/3)·dr/dt ≈ 0.74, 明显大于噪声 0.2,
    数据本身可分 —— 学不会就是网络的问题, 不是数据的问题。
    """
    rng = rng or np.random.default_rng(0)
    X, y = [], []
    r = np.linspace(0.1, 4.0, n_per_class)
    for k in range(n_classes):
        t = np.linspace(0, 3.5 * np.pi, n_per_class) + 2 * np.pi * k / n_classes
        pts = np.stack([r * np.sin(t), r * np.cos(t)], axis=1)
        X.append(pts + rng.normal(0, noise, pts.shape))
        y.append(np.full(n_per_class, k))
    return np.concatenate(X), np.concatenate(y).astype(int)
