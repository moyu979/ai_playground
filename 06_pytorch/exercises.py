"""第 06 课练习。实现后运行 `python exercises.py` 自动测试。"""

import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "04_neural_network"))
from datasets import make_spirals  # noqa: E402

DEVICE = torch.device("cpu")  # 练习用 CPU 即可, 快且稳定


def build_net(hidden: int = 64, n_classes: int = 3) -> nn.Module:
    return nn.Sequential(
        nn.Linear(2, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, n_classes),
    )


def train_model(model: nn.Module, X: np.ndarray, y: np.ndarray,
                opt_name: str = "sgd", lr: float = 0.5, n_steps: int = 4000,
                batch_size: int = 32, seed: int = 0) -> list[float]:
    """已写好的通用训练函数 (五步真言)。返回每步损失列表。"""
    rng = np.random.default_rng(seed)
    opt = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[opt_name](
        model.parameters(), lr=lr)
    losses = []
    for _ in range(n_steps):
        idx = rng.integers(0, len(X), batch_size)
        xb = torch.tensor(X[idx], dtype=torch.float32)
        yb = torch.tensor(y[idx], dtype=torch.long)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        pred = model(xb).argmax(1).numpy()
    return float((pred == y).mean())


# 1. nn.Sequential 版训练 --------------------------------------------------------
def sequential_experiment(X, y, n_steps=4000) -> float:
    """用 build_net() (nn.Sequential) 训练并返回最终训练准确率。

    一行建网络 + train_model() + accuracy() 三步即可。
    """
    raise NotImplementedError


# 2. 早停 ------------------------------------------------------------------------
def early_stopping(X, y, val_frac=0.3, n_steps=6000, patience: int = 1000) -> float:
    """划分 train/val, 训练并返回『最佳 val 准确率』。

    要求:
    - 前 (1-val_frac) 样本训练, 后 val_frac 验证 (先 rng.permutation 打乱)
    - 每 50 步评一次 val; 记录历史最佳 acc 与当时的 state_dict (deepcopy)
    - 连续 patience 步没有提升就提前停止 (早停)
    - 建议固定 torch 初始化种子, 结果才可复现
    返回最佳 val 准确率。
    """
    raise NotImplementedError


# 3. 优化器对决 --------------------------------------------------------------------
def optimizer_duel(X, y, n_steps=3000) -> dict:
    """SGD(lr=0.5) vs Adam(lr=0.01), 各训一个新网络。

    返回 {"sgd": 最终损失, "adam": 最终损失} (各自损失列表的最后一个值)。
    """
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    X, y = make_spirals(100, 3, rng=rng)
    passed = []

    try:
        acc = sequential_experiment(X, y)
        ok = acc > 0.85
        print(f"    Sequential 训练准确率 = {acc:.1%}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("sequential_experiment", ok))

    try:
        X2, y2 = make_spirals(150, 3, rng=rng)   # 早停实验用更多数据更稳
        best = early_stopping(X2, y2)
        ok = best > 0.7
        print(f"    早停最佳 val 准确率 = {best:.1%}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("early_stopping", ok))

    try:
        res = optimizer_duel(X, y)
        ok = set(res) == {"sgd", "adam"} and res["adam"] < res["sgd"]
        print(f"    SGD 末损失 {res.get('sgd', float('nan')):.4f} | "
              f"Adam 末损失 {res.get('adam', float('nan')):.4f}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("optimizer_duel", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 进入 07_cnn/ 吧。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
