# 第 10 课 · Transformer — 组装你自己的 GPT

> 🌐 配套互动网页：[10_transformer.html](../pages/10_transformer.html)（讲解精华 + 可玩 demo，无需任何安装）


> 十课旅程的终点，也是现代 AI 的起点。你将亲手写出 nanoGPT 风格的完整模型，
> 在莎士比亚语料上训练，然后看着它吐出"伪莎士比亚"。
> 这份 `model.py` 与 GPT-2 的唯一区别是**尺寸**——结构上每一个部件都一样。

## 1. 全家福

```
                        ┌─────────────────────────────────────┐
 token 索引 (B,T) ──►  tok_emb + pos_emb                    (B,T,d)
                        │                                     │
                        │  × N 层 [Block]:                    │
                        │    x = x + Attn(LN₁(x))   ◄─ 残差   │
                        │    x = x + MLP(LN₂(x))    ◄─ 残差   │
                        │                                     │
                        │  LN → Linear(d→V)                   │
                        └─────────────────────────────────────┘
 logits (B,T,V) ──► softmax ──► 交叉熵 (对齐右移一位的目标)
```

数据流：`token → 向量 → N 次"注意力+MLP 精炼" → 对每个位置预测下一个 token 的分布`。

## 2. 各部件与出处的对应

| 部件 | 出处 | 一句话理由 |
|---|---|---|
| 因果自注意力 | 第 9 课 | 每个位置并行读取全部历史 |
| **残差连接** `x + F(x)` | ResNet (He 2015) | 梯度有"高速公路"直通浅层；第 8 课 LSTM 门控的加法路径是同一思想。**实测**（`ablation.py`）：拆掉后同预算损失 3.19 vs 完整 2.05 |
| **Pre-LN** `x + F(LN(x))` | GPT-2 改动 | LayerNorm 放进残差分支内，深层训练稳定（原论文是 post-LN） |
| LayerNorm | — | 把每个 token 的特征向量归一化（减均值除方差+仿射），稳定各层分布 |
| MLP: d → 4d → d | Vaswani §3.3 | 注意力负责"混合位置间信息"，MLP 负责在每个位置内"思考"，GELU 激活 |
| learned 位置 embedding | GPT-2 | 注意力对位置天然盲，必须显式注入"第几个 token"（原论文用正弦，GPT 改可学习） |
| weight tying | — | 输出投影与输入 embedding 共享矩阵：省参数，小词表下略提效 |

**初始化**（`model.py` 里 `_init_weights`）：全参数 $\mathcal N(0, 0.02)$，
残差分支的输出投影再乘 $1/\sqrt{2N}$（$N$=层数）——让残差流的方差在深度上
保持 $\approx$ 常数（第 4 课的信号传播分析，深层版）。

## 3. 训练配置（本课的小 GPT）

| 超参 | 值 | nanoGPT 参考值 |
|---|---|---|
| n_layer / n_head / n_embd | 4 / 4 / 128 | 12 / 12 / 768 |
| block_size (上下文) | 64 | 1024 |
| 参数量 | ~0.8 M | 124 M |
| batch | 64 | 12 (× 梯度累积) |
| 优化器 | AdamW (β₁=0.9, β₂=0.95), wd=0.1, lr=3e-4 | 同款 |
| 梯度裁剪 | 1.0 | 同款 |
| 训练步数 | 2000 (Mac 上 ~2 分钟) | 600k |

注意：**唯一的区别是数字，不是结构**。GPT-2/3/4 与这个玩具之间隔着的只是
算力、数据和一堆工程。缩放定律 (Kaplan 2020)：损失随参数/数据/算力按幂律平滑下降——
"大力出奇迹"是被实验证实过的物理。

## 4. 采样：从分布到文本

训练完的模型给出每个位置的下一个 token 分布 $p$，生成时要从中**采样**：

- **温度** $T$：$p_i \propto e^{\text{logit}_i/T}$（第 3 课的玻尔兹曼分布！）
  - $T \to 0$：贪心，永远选最大（稳定但乏味、易复读）
  - $T = 1$：模型原分布；$T > 1$：更发散
- **top-k**：只保留概率前 k 的候选再归一化（砍掉长尾胡话）
- **top-p (nucleus)**：保留累积概率达 p 的最小集合（练习 1 请你实现）

## 5. 运行

```bash
python train.py                 # 训练 + 存 model.pt + 生成样例 (~2~3 分钟)
python train.py --iters 500     # 快速验证版
python sample.py --temperature 0.8 --top-k 40   # 用训好的模型续写
python ablation.py              # 消融实验: 拆掉残差/位置编码的对照 (反面教材)
python exercises.py
```

训练结束会打印 val loss、损失曲线 (`figures/gpt_loss.png`)、生成样例
(`figures/gpt_sample.txt`)。我已替你跑过一次，样例贴在下方（训练日志另见 `figures/`）。

## 6. 论文对照表 (Attention Is All You Need, Vaswani et al. 2017)

| 论文 | 代码 |
|---|---|
| §3.1 encoder-decoder | 我们只留 decoder（GPT 路线）；`model.py` 的 GPT 类 |
| §3.2.1 Scaled Dot-Product Attention | `CausalSelfAttention.forward` |
| §3.2.2 Multi-Head | 同上，`split/transpose/reshape` |
| §3.3 Position-wise FFN | `MLP` 类 (d→4d→d, GELU) |
| §3.4 Positional Encoding | `GPT.pos_emb`（GPT 用可学习版本） |
| §3.5 Residual + LayerNorm | `Block.forward`（post-LN → 我们用 pre-LN） |
| §5.4 Label smoothing | **不用**（GPT 系列不用，直配交叉熵） |

## 我的训练结果（Mac, MPS, 2000 步, ~0.8M 参数）

最终 val 交叉熵 **1.786（困惑度 6.0）**，起点是均匀分布的 4.17（困惑度 64.6）。
生成的"伪莎士比亚"（完整版见 `figures/gpt_sample.txt`）：

```
POMPEY:
This maide lear golloss and bloody, by this,
In a crome the scastle
Heirt him husburin heart cale me such brone
To a course. That put is fone esent of the me;
That he did thave of more.

QUEEN MARGARET:
And no come bid it ha chartimes hell.
```

0.8M 参数已经能拼出莎士比亚式的对白结构、人名与换行节奏（词汇还是"梦话"级——
这正是 scaling 空间）；把配置改成 6/6/384（nanoGPT 的"复现 GPT-2 124M"的 1/10 档）
过夜训练，输出就相当像样了——练手绝佳目标。

## 练习

1. **top-p 采样**：实现 nucleus sampling 并测试边界行为。
2. **手写 GELU**：用 tanh 近似公式实现 GELU，与 `F.gelu` 对拍。
3. **带预热的余弦学习率**：实现 `warmup + cosine decay` 调度器（大模型训练标配）。

## 毕业之后

- 把 `--n-layer 6 --n-embd 384` 跑一个过夜训练（MPS 约 2~4 小时）
- 换语料：把 `data/shakespeare.txt` 换成任何 UTF-8 文本（中文小说直接可用！
  字符级 tokenizer 天然支持），重新 `python train.py`
- 读 nanoGPT 源码，你会发现每一行都眼熟
- Karpathy 的 *Let's reproduce GPT-2 (124M)*：从这里到工业级的完整地图
