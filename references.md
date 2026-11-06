# 参考资源地图

按学习阶段整理的优质资源。标 ⭐ 的是强烈推荐。

## 视频课程

- ⭐ **Andrej Karpathy — Neural Networks: Zero to Hero**（英文，B 站有熟肉）
  https://karpathy.ai/zero-to-hero.html
  本课程的"精神源头"。从 micrograd 一直讲到 GPT，手写代码无剪辑。配合第 5、8、10 课食用。
- ⭐ **3Blue1Brown — 神经网络系列**（官方有中文字幕）
  https://www.3blue1brown.com/topics/neural-networks
  第 1~4 集讲前向传播/梯度下降/反向传播的直觉（配合第 2~4 课），Transformers 一集配合第 9~10 课。
- **李宏毅机器学习**（中文，台大）
  https://speech.ee.ntu.edu.tw/~hylee/index.php
  中文世界最好的体系课之一，Transformer / self-attention 讲解极佳。

## 教材

- ⭐ **《动手学深度学习》(Dive into Deep Learning) 中文版** — 李沐
  https://zh.d2l.ai/  — 免费在线，代码 + 数学 + 图俱全，可当参考书用。
- **Neural Networks and Deep Learning** — Michael Nielsen（免费）
  http://neuralnetworksanddeeplearning.com/ — 反向传播推导极清楚，配合第 4 课。
- **Dive into Deep Learning** 英文版 / **Deep Learning** — Goodfellow, Bengio, Courville
  https://www.deeplearningbook.org/ — 偏理论，工具书。
- **The Little Book of Deep Learning** — François Fleuret（免费 PDF，短小精悍）

## 论文（按出现顺序）

| 论文 | 年份 | 对应课程 |
|---|---|---|
| Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors* | 1986 | 04 |
| LeCun et al., *Gradient-Based Learning Applied to Document Recognition* (LeNet) | 1998 | 07 |
| Bengio et al., *A Neural Probabilistic Language Model* | 2003 | 08 |
| Hochreiter & Schmidhuber, *LSTM* | 1997 | 08 |
| He et al., *Deep Residual Learning* (ResNet) | 2015 | 10 (残差连接) |
| Vaswani et al., *Attention Is All You Need* | 2017 | 09, 10 |
| Radford et al., *Improving Language Understanding by GPT-2* | 2019 | 10 |
| Brown et al., *Language Models are Few-Shot Learners* (GPT-3) | 2020 | 10 |

## 代码仓库（都值得读）

- ⭐ **karpathy/micrograd** — https://github.com/karpathy/micrograd （第 5 课的原型，~100 行）
- ⭐ **karpathy/nanoGPT** — https://github.com/karpathy/nanoGPT （第 10 课的原型）
- **karpathy/minGPT** — https://github.com/karpathy/minGPT （注释更细的 GPT）
- **labml.ai — Annotated Transformer** — https://nn.labml.ai/transformers/ （论文逐行实现）
- **The Illustrated Transformer** — Jay Alammar（图解）
  https://jalammar.github.io/illustrated-transformer/
- **Peter Bloem — Transformers from Scratch** — https://peterbloem.nl/blog/transformers

## 物理人专属桥梁

- ⭐ **Mehta et al., *A High-Bias, Low-Variance Introduction to Machine Learning for Physicists*** (2019)
  https://arxiv.org/abs/1803.08823 — 物理学家写的 ML 综述，统计力学视角，几乎零门槛。
- **Statistical mechanics of deep learning** — Bahri et al., Annu. Rev. Condens. Matter Phys. (2020)
-softmax ↔ 玻尔兹曼分布：$p_i = e^{E_i/T}/Z$ 与 $p_i = e^{z_i}/\sum_j e^{z_j}$ 是同一个公式，
  $T$ 出现在第 10 课的采样温度里。
- 反向传播 ↔ **伴随方法**：优化由 PDE 约束的目标时对伴随变量递推梯度，与神经网络的
  逐层反传在数学上同构。
