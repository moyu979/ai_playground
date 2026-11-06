"""第 10 课 · 完整的字符级 GPT (decoder-only Transformer)。

结构与 nanoGPT / GPT-2 一致 (README §2 的对应表):
  token + 位置 embedding → N × [因果注意力 + MLP, 各带残差与 pre-LN] → LN → 输出头

每个类都标注了对应的课程/论文位置。
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64        # 上下文长度 T
    n_layer: int = 4            # Block 数 N
    n_head: int = 4             # 注意力头数 H
    n_embd: int = 128           # 模型宽度 d
    dropout: float = 0.0
    # ---- 消融开关 (反面教材, ablation.py 用; 正常训练保持 True) ----
    use_residual: bool = True   # False = 拆掉残差连接
    use_pos_emb: bool = True    # False = 拆掉位置编码


class CausalSelfAttention(nn.Module):
    """第 9 课同款 (这里用官方融合算子, 数学与手写版逐位一致)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.d_head = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # 残差分支的输出投影
        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # softmax(QKᵀ/√d_head)V, is_causal=True 即因果掩码 (第 9 课手写过)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_drop if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, T, d)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """位置级前馈网络: d → 4d → d, GELU (论文 §3.3)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Transformer 块: x = x + Attn(LN(x)); x = x + MLP(LN(x))  (pre-LN, GPT-2 风格)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.use_residual = cfg.use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 残差: 梯度高速公路 (ResNet 思想)。ablation.py 会拆掉它做对照。
        h = self.attn(self.ln1(x))
        x = x + h if self.use_residual else h
        h = self.mlp(self.ln2(x))
        x = x + h if self.use_residual else h
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # 可学习位置 (GPT-2)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight            # weight tying

        self.apply(self._init_weights)
        # 残差分支输出投影再缩放: 让残差流方差不随深度增长 (README §2)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "序列超过上下文长度"
        pos = torch.arange(T, device=idx.device)

        x = self.tok_emb(idx)
        if self.cfg.use_pos_emb:      # ablation.py 会拆掉位置编码做对照
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                             # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """自回归采样 (README §4): 每步取分布, 采样一个 token, 滑动窗口。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # 只保留最近上下文
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)   # 玻尔兹曼温度!
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:   # 习惯: 统计时不计 embedding
            n -= self.tok_emb.weight.numel()
        return n


if __name__ == "__main__":
    # 自检: 前向形状 + 因果性 + 参数量
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(cfg)
    x = torch.randint(0, 65, (2, 32))
    y = torch.randint(0, 65, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, 65) and 0 < loss.item() < 10

    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 65           # 篡改最后一个 token
    with torch.no_grad():
        l1, _ = model(x)
        l2, _ = model(x2)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5), "因果性被破坏!"
    print(f"GPT 自检通过 ✓  非嵌入参数量 {model.num_params():,}")
    print("运行 python train.py 开始训练。")
