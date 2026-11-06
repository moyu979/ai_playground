"""第 08 课练习。实现后运行 `python exercises.py` 自动测试。

提示: 第 1 题与 bigram.py 的逻辑高度平行, 只是条件从 1 个字符变 2 个。
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_names_and_vocab():
    names = [l.strip().lower() for l in
             open(os.path.join(DATA_DIR, "names.txt")) if l.strip()]
    chars = ["."] + sorted(set("".join(names)))
    stoi = {c: i for i, c in enumerate(chars)}
    return names, chars, stoi


def bigram_ce(names: list[str], stoi: dict) -> float:
    """已写好的 bigram 基线 (与 bigram.py 相同, 含拉普拉斯平滑)。"""
    V = len(stoi)
    N = np.zeros((V, V))
    for name in names:
        seq = ["."] + list(name) + ["."]
        for a, b in zip(seq, seq[1:]):
            N[stoi[a], stoi[b]] += 1
    P = (N + 1.0) / (N.sum(axis=1, keepdims=True) + V)
    total, count = 0.0, 0
    for name in names:
        seq = ["."] + list(name) + ["."]
        for a, b in zip(seq, seq[1:]):
            total += -np.log(P[stoi[a], stoi[b]] + 1e-12)
            count += 1
    return total / count


# 1. trigram 模型 -----------------------------------------------------------------
def trigram_model(names: list[str], stoi: dict) -> dict:
    """构建 trigram 模型: 返回 dict, key = (前前字符 idx, 前一字符 idx),
    value = 长度为 V 的概率向量 (下一字符分布, 含 +1 平滑)。

    语料遍历方式: 对每个名字取 seq = ['.','.', *list(name), '.'],
    对相邻三元组 (a, b, c) 更新计数 N[(stoi[a], stoi[b])][stoi[c]] += 1。
    """
    raise NotImplementedError


def trigram_ce(names: list[str], model: dict, stoi: dict) -> float:
    """用上面的 trigram 模型计算平均负对数似然 (交叉熵)。"""
    raise NotImplementedError


# 2. 温度采样 -----------------------------------------------------------------------
def temperature_sample(p: np.ndarray, T: float, rng: np.random.Generator) -> int:
    """按温度 T 采样一个下标 (README 练习 2)。

    公式: q_i ∝ p_i^(1/T), 归一化后按 q 采样 (rng.choice)。
    T → 0: 贪心 (只选最大概率); T → ∞: 趋近均匀。
    第 10 课 GPT 的生成参数就是它。
    """
    raise NotImplementedError


# 3. GRU 对决 -----------------------------------------------------------------------
class SeqRNN(nn.Module):
    """已写好的模板 (与 rnn_char.py 的 CharRNN 相同接口)。"""

    def __init__(self, vocab, emb=32, hidden=96):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.RNN(emb, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, idx):                    # (B, T) → (B, T, V)
        x = self.emb(idx)
        out, _ = self.rnn(x)
        return self.head(out)


class SeqGRU(nn.Module):
    """仿照 SeqRNN 写一个 GRU 版: 把 nn.RNN 换成 nn.GRU, 结构完全一样。"""

    def __init__(self, vocab, emb=32, hidden=96):
        super().__init__()
        raise NotImplementedError


def train_lm(model, ids: torch.Tensor, steps: int, seed: int = 0) -> float:
    """已写好: 训练 steps 步, 返回最终交叉熵。B=32, T=32, Adam 1e-3。"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, T = 32, 32
    for _ in range(steps):
        ix = torch.randint(0, len(ids) - T - 1, (B,))
        x = torch.stack([ids[i : i + T] for i in ix])
        y = torch.stack([ids[i + 1 : i + 1 + T] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, model.head.out_features),
                               y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return float(loss.item())


def gru_vs_rnn(ids: torch.Tensor, steps: int = 500) -> dict:
    """分别训练 SeqRNN 和 SeqGRU (同一 seed), 返回 {"rnn": ce, "gru": ce}。"""
    raise NotImplementedError


# ============================ 自动测试 =======================================
def _test() -> None:
    rng = np.random.default_rng(0)
    passed = []

    names, chars, stoi = load_names_and_vocab()
    V = len(chars)

    # 1 trigram 应显著优于 bigram
    try:
        model = trigram_model(names, stoi)
        ce_tri = trigram_ce(names, model, stoi)
        ce_bi = bigram_ce(names, stoi)
        ok = ce_tri < ce_bi
        print(f"    bigram ce={ce_bi:.3f}  trigram ce={ce_tri:.3f}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("trigram < bigram", ok))

    # 2 温度采样行为
    try:
        p = np.full(V, 0.01)
        p[7] = 0.5                                  # 一个显著峰值
        picks_low = [temperature_sample(p, 0.1, rng) for _ in range(500)]
        picks_high = [temperature_sample(p, 20.0, rng) for _ in range(2000)]
        frac_low = np.mean(np.array(picks_low) == 7)
        top_high = max(np.bincount(picks_high, minlength=V) / len(picks_high))
        ok = frac_low > 0.95 and top_high < 0.10
        print(f"    T=0.1 峰值命中率 {frac_low:.0%} | T=20 最高频占比 {top_high:.1%}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("temperature_sample", ok))

    # 3 GRU 应优于裸 RNN
    text = open(os.path.join(DATA_DIR, "shakespeare.txt"), encoding="utf-8").read()
    vocab2 = sorted(set(text))
    s2 = {c: i for i, c in enumerate(vocab2)}
    ids = torch.tensor([s2[c] for c in text[:60000]])
    try:
        res = gru_vs_rnn(ids)
        ok = set(res) == {"rnn", "gru"} and res["gru"] < res["rnn"]
        print(f"    RNN ce={res.get('rnn', float('nan')):.3f} | "
              f"GRU ce={res.get('gru', float('nan')):.3f}")
    except NotImplementedError:
        ok = False
        print("    (未实现)")
    passed.append(("gru_vs_rnn", ok))

    print("=" * 62)
    n_ok = sum(ok for _, ok in passed)
    for name, ok in passed:
        print(f" [{'通过' if ok else '未通过'}] {name}")
    print("=" * 62)
    if n_ok == len(passed):
        print(" 全部通过! 进入 09_attention/ —— 终章即将开始。")
    else:
        print(f" 通过 {n_ok}/{len(passed)}。")


if __name__ == "__main__":
    _test()
