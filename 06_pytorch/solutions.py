"""第 06 课练习 · 参考答案。"""

import copy

import numpy as np
import torch

from exercises import (DEVICE, accuracy, build_net, train_model)


def sequential_experiment(X, y, n_steps=8000) -> float:
    model = build_net().to(DEVICE)
    train_model(model, X, y, n_steps=n_steps)
    return accuracy(model, X, y)


def early_stopping(X, y, val_frac=0.3, n_steps=6000, patience: int = 1000,
                   seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)                     # 固定初始化, 结果可复现
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    model = build_net().to(DEVICE)
    rng2 = np.random.default_rng(seed + 1)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)   # 收敛快, 早停才有意义

    best_acc, best_state = -1.0, None
    steps_since_best = 0
    for step in range(n_steps + 1):
        if step % 50 == 0:
            acc = accuracy(model, X[val_idx], y[val_idx])
            if acc > best_acc:
                best_acc = acc
                best_state = copy.deepcopy(model.state_dict())
                steps_since_best = 0
            else:
                steps_since_best += 50
                if steps_since_best > patience:
                    break
        if step == n_steps:
            break
        j = rng2.integers(0, len(tr_idx), 64)
        xb = torch.tensor(X[tr_idx[j]], dtype=torch.float32)
        yb = torch.tensor(y[tr_idx[j]], dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best_acc


def optimizer_duel(X, y, n_steps=3000) -> dict:
    res = {}
    for name, lr in (("sgd", 0.5), ("adam", 0.01)):
        model = build_net().to(DEVICE)
        losses = train_model(model, X, y, opt_name=name, lr=lr, n_steps=n_steps)
        res[name] = float(losses[-1])
    return res


if __name__ == "__main__":
    import exercises

    exercises.sequential_experiment = sequential_experiment
    exercises.early_stopping = early_stopping
    exercises.optimizer_duel = optimizer_duel
    exercises._test()
