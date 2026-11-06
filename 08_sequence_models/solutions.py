"""第 08 课练习 · 参考答案。"""

import numpy as np
import torch
import torch.nn as nn

from exercises import SeqRNN, train_lm


def trigram_model(names: list[str], stoi: dict) -> dict:
    V = len(stoi)
    N: dict = {}
    for name in names:
        seq = [".", "."] + list(name) + ["."]
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            key = (stoi[a], stoi[b])
            if key not in N:
                N[key] = np.ones(V)              # +1 平滑: 从 1 开始数
            N[key][stoi[c]] += 1
    return {k: v / v.sum() for k, v in N.items()}


def trigram_ce(names: list[str], model: dict, stoi: dict) -> float:
    total, count = 0.0, 0
    for name in names:
        seq = [".", "."] + list(name) + ["."]
        for a, b, c in zip(seq, seq[1:], seq[2:]):
            p = model[(stoi[a], stoi[b])][stoi[c]]
            total += -np.log(p + 1e-12)
            count += 1
    return total / count


def temperature_sample(p: np.ndarray, T: float, rng: np.random.Generator) -> int:
    q = np.power(np.asarray(p, dtype=float), 1.0 / T)
    q = q / q.sum()
    return int(rng.choice(len(q), p=q))


class SeqGRU(nn.Module):
    def __init__(self, vocab, emb=32, hidden=96):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx):
        x = self.emb(idx)
        out, _ = self.rnn(x)
        return self.head(out)


def gru_vs_rnn(ids: torch.Tensor, steps: int = 500) -> dict:
    v = _vocab(ids)
    return {
        "rnn": train_lm(SeqRNN(v), ids, steps, seed=0),
        "gru": train_lm(SeqGRU(v), ids, steps, seed=0),
    }


def _vocab(ids: torch.Tensor) -> int:
    return int(ids.max().item()) + 1


if __name__ == "__main__":
    import exercises

    exercises.trigram_model = trigram_model
    exercises.trigram_ce = trigram_ce
    exercises.temperature_sample = temperature_sample
    exercises.SeqGRU = SeqGRU
    exercises.gru_vs_rnn = gru_vs_rnn
    exercises._test()
