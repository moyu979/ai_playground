# Playground 源码深读 — 原理 · 数学 · 每一步代码

> 配合 `index.html` 阅读。读完你会确认一件事：
> **这个网页里没有任何魔法** —— 它就是第 04 课你手写过的那个 NumPy MLP，
> 换了 JavaScript 的皮，外加三块"画布"把它的内心戏画出来。
> 全部引擎 ~300 行，零框架、零依赖。

## 0. 全景：页面里到底在跑什么

```
┌─ 数据 ─────────────┐   ┌─ 引擎 (本文件核心) ────────────┐   ┌─ 渲染 ──────────┐
│ genData(): 四种    │ → │ buildNet()  搭网络 (随机初始化) │ → │ renderBoundary  │
│ 2D 点, 标签 ±1     │   │ forward()   前向传播            │   │  46×46 网格上    │
│                    │   │ trainBatch() 反传 + SGD (16/批) │   │  逐点算输出染色  │
└────────────────────┘   └────────────────────────────────┘   │ renderNet 画权重 │
                                                              │ renderLoss 画曲线│
                                                              └─────────────────┘
```

数据流：点「开始训练」后，`requestAnimationFrame` 每帧调一次 `stepN(12)` = 跑
12 个 mini-batch，然后重画三块画布。60 Hz 屏幕 ≈ 每秒 720 个 batch。

## 1. 原理：我们让网络学什么

- **任务**：2D 二分类。每个点 $p = (x_1, x_2)$，标签 $y \in \{+1, -1\}$。
- **模型**：全连接网络 $[2, \text{hidden}\dots, 1]$，隐藏层激活 $\phi \in \{\tanh, \text{ReLU}, \text{sigmoid}\}$，
  **输出层固定 tanh**。
- **为什么输出用 tanh**：tanh 值域 $(-1,1)$ 正好和标签 $\pm 1$ 对齐，
  输出值直接当"颜色深浅"画成决策边界图（TF Playground 的同款约定）。
- **损失**：均方误差 $L = \frac1B \sum_{b} (\hat y_b - y_b)^2$。
- **训练**：mini-batch SGD，$B = 16$ —— 和第 02/04 课完全同一个算法。

## 2. 数学：完整推导（对着代码看）

### 2.1 前向传播

第 $l$ 层（行向量记号，batch 友好）：

$$z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}, \qquad a^{(l)} = \phi(z^{(l)})$$

形状：$a^{(l-1)} \in \mathbb{R}^{n_{l-1}}$，$W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$（输入维→输出维）。

### 2.2 损失与输出层梯度

单个样本：$L = (\hat y - y)^2$，$\hat y = \tanh(z^{(L)})$。链式法则拆开：

$$\frac{\partial L}{\partial z^{(L)}} = \underbrace{2(\hat y - y)}_{\partial L/\partial \hat y} \cdot \underbrace{(1 - \hat y^2)}_{\tanh'(z)=1-\tanh^2}$$

**这就是引擎里最关键的一行**（见 §3.3）：

```js
let delta = [2 * (out - y) * (1 - out * out)];   // 输出层: tanh' = 1 - a²
```

对照第 04 课：那里输出是 softmax+交叉熵，$\delta^{(L)} = (P-Y)/B$；
这里是 tanh+MSE，$\delta^{(L)} = 2(\hat y-y)(1-\hat y^2)$。**骨架一模一样，只换了"输出层头部"**。

### 2.3 反向传播递推（第 04 课公式 6）

$$\delta^{(l)} = \big(\delta^{(l+1)} (W^{(l+1)})^T\big) \odot \phi'(a^{(l)})$$

两个动作：矩阵乘把"下游对我的总敏感度"收上来，再乘本层激活的局部导数。
三个激活函数的导数都能**用激活值 $a$ 直接表示**（所以代码只缓存 $a$、不用存 $z$）：

| $\phi$ | $\phi'(z)$ | 用 $a=\phi(z)$ 表示 |
|---|---|---|
| tanh | $1-\tanh^2 z$ | $1-a^2$ |
| ReLU | $\mathbb{1}[z>0]$ | $\mathbb{1}[a>0]$ |
| sigmoid | $\sigma(1-\sigma)$ | $a(1-a)$ |

### 2.4 参数梯度（公式 7、8）与更新

$$\frac{\partial L}{\partial W^{(l)}} = \frac{1}{B}(a^{(l-1)})^T \delta^{(l)}, \qquad
\frac{\partial L}{\partial b^{(l)}} = \frac{1}{B}\sum_b \delta^{(l)}_b, \qquad
W \leftarrow W - \eta \frac{\partial L}{\partial W}$$

### 2.5 梯度下降页的数学（Tab 2）

线性模型 $\hat y = wx+b$，$L(w,b) = \frac1N\sum (wx_i+b-y_i)^2$，解析梯度：

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum e_i x_i, \qquad
\frac{\partial L}{\partial b} = \frac{2}{N}\sum e_i, \qquad e_i = wx_i + b - y_i$$

**稳定性分析**（这就是本页数据选 $x\in[-1,1]$ 的原因，和第 02 课同款）：
Hessian $H = 2\begin{psmallmatrix}\overline{x^2} & \bar x\\ \bar x & 1\end{psmallmatrix}
\approx \begin{psmallmatrix}0.67 & 0\\ 0 & 2\end{psmallmatrix}$（均匀分布 $\overline{x^2}=1/3$，$\bar x=0$），
特征值 $\{0.67,\ 2\}$，于是：

- 收敛条件 $\eta < 2/\lambda_{\max} = 1$。**lr=1.4 时 $b$ 方向误差因子
  $|1-1.4\times 2| = 1.8 > 1$ → 振荡发散**（$w$ 方向 $|1-1.4\times0.67|=0.06$ 反而没事——
  发散总是从"最陡"的特征方向开始）。
- 动量把稳定域扩成 $\eta < 2(1+\beta)/\lambda_{\max} = 1.9$（$\beta=0.9$）：
  所以 **lr=1.4 + 动量 = 获救**。"惯性平均振荡"不是玄学，是这个不等式。
- lr=0.005：两个方向都收缩得极慢——慢弛豫。

页面上的"已发散"徽章、红星（解析最优点 $\theta^* = (X^TX)^{-1}X^Ty$）、
轨迹，都是这几行公式的可视化。数值验证：lr=0.3 收敛到 $w=1.223, b=-0.609$
（真值 $1.2/-0.6$）✓。

## 3. 代码逐段讲解（`index.html` 的 `<script>`）

### 3.1 数据结构：网络是什么

```js
function buildNet() {
  const sizes = [2, ...hidden, 1];            // 例: hidden=[4,4] → [2,4,4,1]
  const layers = [];
  for (let l = 1; l < sizes.length; l++) {
    const nIn = sizes[l - 1], nOut = sizes[l];
    const scale = activation === "relu" ? Math.sqrt(2 / nIn) : Math.sqrt(1 / nIn);
    const W = [], b = [];
    for (let j = 0; j < nOut; j++) {
      const row = [];
      for (let i = 0; i < nIn; i++) row.push(gaussian() * scale);
      W.push(row); b.push(0);
    }
    layers.push({ W, b });                    // 每层 = {W: (nIn×nOut), b: (nOut,)}
  }
  return { sizes, layers };
}
```

- 网络就是**一个嵌套数组**：`net.layers[l].W[j][i]` = 第 $l$ 层、第 $j$ 个输出神经元、
  连到第 $i$ 个输入的权重。没有类、没有框架。
- `scale` 就是第 04 课 §5 的初始化方差分析：ReLU 配 **He**（$\sqrt{2/n_{in}}$），
  tanh/sigmoid 配 **Xavier**（$\sqrt{1/n_{in}}$）。改这一行就能复现"大方差爆炸、
  全零死寂"的反面教材。
- `gaussian()` 是 Box–Muller 变换 —— 你在统计物理里算过的高斯采样。

### 3.2 前向：`forward()`，以及为什么缓存 `acts`

```js
function forward(net, x0, activation) {
  const acts = [x0.slice()];                  // acts[0] = 输入, 缓存每一层的输出
  let a = x0;
  for (let l = 0; l < net.layers.length; l++) {
    const { W, b } = net.layers[l];
    const out = [];
    for (let j = 0; j < W.length; j++) {
      let z = b[j];
      for (let i = 0; i < W[j].length; i++) z += W[j][i] * a[i];   // z = W·a + b
      out.push(l === net.layers.length - 1 ? Math.tanh(z)      // 输出层固定 tanh
                                           : actF(activation, z));
    }
    acts.push(out);
    a = out;
  }
  return acts;
}
```

- 双重循环就是 $z_j = \sum_i W_{ji} a_i + b_j$，教科书式实现。
- **`acts` 是反传的生命线**：公式 (7) 要用 $a^{(l-1)}$、公式 (6) 要用 $a^{(l)}$，
  所以前向把每层输出都存下来——这就是第 04 课 `Dense.forward` 里 `self._x = x`
  的 JS 版。深度学习框架里这块缓存叫 "activation memory"（显存大头就是它）。

### 3.3 反传 + SGD：`trainBatch()`，一行一行对公式

```js
function trainBatch(net, data, activation, lr) {
  const B = 16;                               // mini-batch 大小
  const G = zeroGrads(net);                   // 梯度累加器 gW/gB, 全 0 起步
  let lossSum = 0;

  for (let s = 0; s < B; s++) {
    const p = data[(Math.random() * data.length) | 0];   // 有放回抽 1 个样本
    const acts = forward(net, [p.x, p.y], activation);
    const out = acts[acts.length - 1][0];     // ŷ
    const y = p.label;
    lossSum += (out - y) * (out - y);         // 累计损失 (只为了画曲线)

    // ---- 反向开始: 输出层 δ (§2.2) ----
    let delta = [2 * (out - y) * (1 - out * out)];

    // ---- 从最后一层往第 0 层扫 (公式 6/7/8) ----
    for (let l = net.layers.length - 1; l >= 0; l--) {
      const aPrev = acts[l];                  // a^(l-1), 公式 (7) 要用
      const { W } = net.layers[l];
      for (let j = 0; j < W.length; j++) {
        G[l].gB[j] += delta[j];                              // 公式 (8)
        for (let i = 0; i < aPrev.length; i++)
          G[l].gW[j][i] += delta[j] * aPrev[i];              // 公式 (7)
      }
      if (l > 0) {                            // 往更早的层传 δ
        const aCur = acts[l];
        const da = new Array(aCur.length).fill(0);
        for (let j = 0; j < W.length; j++)
          for (let i = 0; i < aCur.length; i++)
            da[i] += delta[j] * W[j][i];      // δ^(l+1) · (W^(l+1))ᵀ 的分量
        delta = da.map((g, i) => g * dActF(activation, aCur[i]));  // ⊙ φ' (公式 6)
      }
    }
  }

  // ---- SGD 更新: 平均梯度 × lr (和第 04 课 Dense.step 相同) ----
  for (let l = 0; l < net.layers.length; l++) {
    const { W, b } = net.layers[l];
    for (let j = 0; j < W.length; j++) {
      b[j] -= lr * G[l].gB[j] / B;
      for (let i = 0; i < W[j].length; i++)
        W[j][i] -= lr * G[l].gW[j][i] / B;
    }
  }
  return lossSum / B;
}
```

读懂这段的四个要点：

1. **逐样本反传，最后除 $B$**：数学上 == 先算 16 个样本的梯度求平均 ==，
   和矩阵版一次算完等价。逐样本写法看得见每一项；矩阵写法（第 06 课起）快
   ~1000 倍。**数学没变，变的是宿主语言和循环顺序。**
2. **`G` 累加器为什么存在**：mini-batch = 16 份"局部梯度"求和。第 06 课
   `common_bugs.py` 里"忘记 zero_grad"炸掉的正是这个累加器被跨 batch 延续——
   这里每个 batch 新建 `G`，等价于每步 `zero_grad()`。
3. **`delta` 的重用**：变量名 `delta` 就是公式里的 $\delta^{(l)}$。
   循环体干两件事：用当前 $\delta$ 收割本层参数梯度（7/8），然后把 $\delta$ 更新成
   上一层的（6）。`da[i] += delta[j] * W[j][i]` 就是矩阵乘
   $\delta (W)^T$ 的**展开写法**——第 06 课起这行交给 BLAS。
4. **`dActF`（公式 6 的 ⊙ 因子）**：

```js
function dActF(name, a) {          // 用激活值 a 表示的 φ'
  if (name === "relu") return a > 0 ? 1 : 0;
  if (name === "sigmoid") return a * (1 - a);
  return 1 - a * a;                // tanh
}
```

### 3.4 渲染：三块画布各在画什么

- **`renderBoundary()`（决策边界）**：在 $[-5,5]^2$ 上铺 $46\times46$ 网格，
  每个格点跑一遍 `forward()`，输出 $v\in(-1,1)$ → 橙（正）/蓝（负）、透明度
  $\propto|v|$。你看到的"彩色云"就是网络在 2116 个点上投票的结果。
  训练时每帧重画 → 边界"活"起来。
- **`renderNet()`（网络图）**：每条边的线宽 $= \min(5,\ 0.5 + |w|\cdot 2.2)$，
  颜色 = 符号。训练时看着某条线变粗，就是那个权重在变大。
- **`renderLoss()`**：画的是损失的**指数滑动平均**
  （`ema = 0.95·ema + 0.05·loss`）——单步损失太抖（SGD 的"热噪声"，
  第 02 课 §5），平滑后趋势才 readable。

### 3.5 梯度下降页（Tab 2）：最短的完整范例

```js
function gdGrad(w, b) {                        // ∇L 的解析式 (§2.5)
  let gw = 0, gb = 0;
  for (let i = 0; i < GD.xs.length; i++) {
    const e = w * GD.xs[i] + b - GD.ys[i];
    gw += e * GD.xs[i]; gb += e;
  }
  return [2 * gw / GD.xs.length, 2 * gb / GD.xs.length];
}
// 每帧: v = β·v − lr·g ;  θ += v
GD.v[0] = mom * GD.v[0] - lr * g[0];
GD.W[0] += GD.v[0];
```

10 行 = 第 02 课的全部。没有 autograd、没有框架——**梯度是你自己写的**。
曲面热图是把 140×140 个 $(w,b)$ 网格点的 $L$ 值直接染成亮度。

## 4. 三方对照表：网页 JS ↔ 第 04 课 NumPy ↔ 公式

| 概念 | `index.html` | `04_neural_network/mlp.py` | 公式 |
|---|---|---|---|
| 搭网络 | `buildNet()` | `MLP.__init__` | §2 |
| 初始化尺度 | `Math.sqrt(2/nIn)` 等 | `Dense.__init__` (he/xavier) | §5 |
| 前向 + 缓存 | `forward()` → `acts` | `Dense.forward` → `self._x` | §2.1 |
| 输出层 δ | `2*(out-y)*(1-out*out)` | `(P - Y) / B` | (4) |
| 递推 δ | `da[i] += delta[j]*W[j][i]` + `dActF` | `Activation.backward` | (6) |
| 权重梯度 | `G[l].gW[j][i] += delta[j]*aPrev[i]` | `self.dW = self._x.T @ dz` | (7) |
| 偏置梯度 | `G[l].gB[j] += delta[j]` | `self.db = dz.sum(0)` | (8) |
| SGD 更新 | 循环 `-= lr*g/B` | `Dense.step(lr)` | 第 02 课 |
| 损失 | MSE `(out-y)²` | softmax 交叉熵 | 第 03 课 |
| 梯度检查 | —（你来做，见下） | `exercises.grad_check` | 第 04 课练习 1 |

唯一的"故意不同"：网页用 tanh+MSE（为了画边界好看），课程用 softmax+交叉熵
（为了接语言模型）。第 04 课练习里你已经做过 tanh 版——同一个引擎两种头。

## 5. 动手验证：别信我，信实验

1. **反传公式是"必要"的吗？** 打开 `index.html`，把 §3.3 那行
   `let delta = [2 * (out - y) * (1 - out * out)];` 改成
   `let delta = [2 * (out - y)];`（忘掉 tanh'）→ 刷新 → 训练。
   它会**学得极其吃力甚至发散**。梯度错一项，网络就是瘸的——
   这就是为什么第 04 课练习 1 要做数值梯度检查。
2. **B=16 → 1**：把 `const B = 16;` 改成 `1` → 单样本 SGD，损失曲线立刻变得
   很"吵"（第 02 课 §5 的热噪声可视化）。
3. **页面 ↔ Python 对照**：第 06 课 `mlp_pytorch.py` 已把 NumPy 曲线和
   PyTorch 曲线叠在一起；你可以再把这个网页的损失导出来画进去
   （`lossHist` 就在内存里，控制台 `copy(JSON.stringify(lossHist))`）。
4. **控制台即实验室**：F12 打开控制台，可以直接 `net.layers[0].W` 看权重、
   `hidden = [8,8,8]; fullReset(true)` 换结构——页面所有状态都是全局可玩的。

## 6. 延伸

- 想看"矩阵版 + 框架版"的同一套数学 → 第 06 课。
- 想看这套引擎长到 65 维输出、12 层深、带注意力 → 第 10 课 `model.py`。
- 改坏了？`git checkout` 不存在——找我，或者去 `_run_all_tests.sh` 里跑一遍
  课程全量测试确认"网页之外一切正常"，然后把 `index.html` 恢复重写一份也只
  需要我一句话的事。
