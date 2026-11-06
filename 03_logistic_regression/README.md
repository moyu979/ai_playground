# 第 03 课 · 逻辑回归 — 当输出需要成为"概率"

> 🌐 配套互动网页：[03_sigmoid.html](../pages/03_sigmoid.html)（讲解精华 + 可玩 demo，无需任何安装）


> 物理视角：你早就见过今天的主角。
> **sigmoid 就是费米–狄拉克分布** $\sigma(x) = 1/(e^{-x}+1)$（无简并费米子占据率）；
> **softmax 就是玻尔兹曼分布** $p_i \propto e^{-E_i/T}$——把 logit 看成负能量，
> 温度 $T=1$。这不是巧合，这是"从分数到概率"的两次最大熵选择。

## 1. 分类问题：线性模型不够了

回归预测连续值，分类预测离散类别。手写 digit 是 0~9，垃圾邮件是 0/1。
直接用上一课的直线 $\hat y = wx+b$ 有两个毛病：

1. 输出是 $(-\infty, +\infty)$，不是概率
2. 平方损失对分类是凸的但梯度在最错的地方反而不大，且统计上不再对应最大似然

**解法：线性分数 → 概率的非线性映射。**

## 2. 二分类：sigmoid

给每个样本算一个"分数" $z = w\cdot x + b$，再过 sigmoid：

$$p(y=1|x) = \sigma(z) = \frac{1}{1+e^{-z}}$$

性质：$\sigma'(z) = \sigma(z)(1-\sigma(z))$（求导只有一次乘法，这就是它历史上流行的原因），
$z\to\pm\infty$ 时饱和到 1/0。

## 3. 损失：交叉熵 = 负对数似然

数据在模型下的对数似然（伯努利）：

$$\log p(D|w,b) = \sum_i \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]$$

最大化似然 = 最小化它的负数（按样本平均），即**二元交叉熵**：

$$L = -\frac{1}{N}\sum_i \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]$$

> 为什么不能用平方损失？理论上：平方损失对应高斯似然，分类标签是伯努利；
> 实践上：sigmoid+平方损失的损失面非凸，交叉熵损失面是凸的（对逻辑回归）。

**对 $z$ 求梯度（先自己推一遍，结果美得惊人）：**

$$\frac{\partial L}{\partial z_i} = p_i - y_i \quad\Leftarrow\quad \text{预测减标签，仅此而已}$$

sigmoid 的饱和在求导时被 BCE 的 $\log$ 恰好抵消——这就是"配对"的意义。
错误越离谱，梯度越大；预测正确，梯度归零。

## 4. 多分类：softmax

$K$ 个类，每类一个线性分数 $z_k = w_k \cdot x + b_k$（权重矩阵 $W \in \mathbb R^{K\times d}$）：

$$p_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$$

损失变成 $K$ 类交叉熵 $L = -\frac1N \sum_i \log p_{y_i}$，对 $z$ 的梯度依旧是
**概率减 one-hot**：$\partial L/\partial z_{ik} = p_{ik} - \delta_{k,y_i}$。

> **玻尔兹曼彩蛋**：logit $z_k = -E_k/T$，则 $p_k = e^{-E_k/T}/Z$。
> $Z$ 就是配分函数。第 10 课 GPT 生成文本时的"温度采样"会直接用这个 $T$。

## 5. 数值稳定的 softmax（重要工程细节）

$z$ 很大时 $e^{z}$ 上溢。利用 softmax 的平移不变性 $\text{softmax}(z) = \text{softmax}(z - c)$，
实现时**永远先减去最大值**：

```python
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # 平移不变, 防溢出
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
```

同理交叉熵用 $\log p = z_{y} - \log\sum_j e^{z_j}$（log-sum-exp）直接算，别先算 $p$ 再 $\log$。
练习 2 要求你实现它。

## 6. 决策边界：仍是线性的

$\sigma(z)=0.5$ 处 $z=0$，所以逻辑回归的决策边界是超平面 $w \cdot x + b = 0$。
看代码生成的图 2/图 4：它能切分线性可分的数据，对环形/螺旋形无能为力——
**这正是第 4 课引入隐藏层的动机**。

## 运行

```bash
python logistic_regression.py   # 图: sigmoid、二分类边界、损失曲线、三分类区域、
                                #     linear_fail.png (对照组: 圆环数据上露馅)
python exercises.py
```

> **对照组思维**：说"逻辑回归简单有效"不算数——脚本最后会把同一个模型扔到
> 线性不可分的圆环数据上，让你亲眼看到它卡在一条直线上的样子。
> （这也是第 4 课存在的理由。）

## 练习

1. **稳定 softmax**：写减最大值版本，用 $10^4$ 量级输入验证不溢出。
2. **log-sum-exp 交叉熵**：不经过概率直接算 $\log p_y$，和朴素实现对比。
3. **数值梯度检查**：用第 1 课的 `finite_difference_grad` 验证 $\partial L/\partial z = p - y$。
   （从这课起，"梯度检查"是你代码的标配保险。）

## 预告

逻辑回归 = "没有隐藏层的神经网络"。第 4 课加上隐藏层，就拥有了万能逼近——
代价是反向传播的手工推导。所有素材今天已备齐：线性变换、sigmoid、交叉熵、
$\hat p - y$ 梯度。缺的只是"多层链式求导的系统性方法"。
