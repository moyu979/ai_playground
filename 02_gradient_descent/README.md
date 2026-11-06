# 第 02 课 · 线性回归与梯度下降 — 深度学习的"第一推动"

> 物理视角：梯度下降就是**过阻尼动力学**。把损失 $L(\theta)$ 看成势能面，
> 参数是坐标，训练就是坐标沿 $\dot\theta = -\eta \nabla L$ 的弛豫过程。
> 今天学的每一个细节（学习率、动量、随机性）都是动力学的真实物理。

## 1. 问题：从数据学一条直线

你有 $N$ 个观测 $(x_i, y_i)$，假设（模型）：

$$\hat y_i = w x_i + b$$

**学习 = 找到让模型"最符合数据"的 $w, b$。** 把"最符合"写成可优化的函数（损失）：

$$L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (\hat y_i - y_i)^2$$

为什么是平方？不只是"看起来顺眼"——若噪声是独立高斯 $\epsilon_i \sim \mathcal N(0, \sigma^2)$，
则最小化平方损失**等价于**最大化似然 $p(y|x) \propto e^{-\sum_i (\hat y_i - y_i)^2 / 2\sigma^2}$。
最小二乘 = 高斯噪声下的最大似然。这是 Legendre 和 Gauss 当年为算小行星轨道发明的数学。

## 2. 解析解存在，但它不是重点

定义设计矩阵 $X \in \mathbb R^{N\times 2}$（列 1 是 $x$，列 2 恒为 1），正规方程：

$$\nabla_\theta L = 0 \;\Rightarrow\; \theta = (X^T X)^{-1} X^T y$$

一行代码可解。**但深度学习里没有解析解**——一旦往中间插入非线性层，$\nabla L = 0$
就是数百万维的非线性方程组。所以真正的主角是：**数值迭代**。

## 3. 梯度下降

在参数空间里沿负梯度走一小步，重复：

$$\theta_{t+1} = \theta_t - \eta\, \nabla_\theta L(\theta_t)$$

对本课的模型（$\hat y_i = wx_i + b$，令 $e_i = \hat y_i - y_i$）：

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum_i e_i x_i, \qquad
  \frac{\partial L}{\partial b} = \frac{2}{N}\sum_i e_i$$

> **物理类比**：$\dot\theta = -\eta\nabla L$ 是梯度流（gradient flow），
> 等价于动力学 $m\ddot\theta + \gamma\dot\theta = -\nabla L$ 的强摩擦极限——
> 石头在蜂蜜里滚下山。**学习率 $\eta$ 就是弛豫时间尺度的倒数。**

## 4. 学习率：稳定性的线性化分析

二次损失附近，$L$ 可以展开为 $L(\theta) \approx L^* + \frac12 (\theta-\theta^*)^T H (\theta-\theta^*)$，
$H$ 是 Hessian（正定）。代回更新式做线性稳定性分析：

$$\theta_{t+1} - \theta^* = (I - \eta H)(\theta_t - \theta^*)$$

**收敛 ⟺ $I - \eta H$ 的谱半径 < 1 ⟺ $0 < \eta < 2/\lambda_{\max}(H)$。**

- $\eta$ 太小：慢弛豫，跑不完预算
- $\eta$ 太大：$\lambda_{\max}$ 方向的误差每步放大，**发散**（你会在图 3 里亲眼看到）
- 这和刚性 ODE 显式积分器的稳定性条件 $h < 2/\lambda_{\max}$ 是同一件事

## 5. SGD 与 mini-batch：训练的"热噪声"

全数据梯度要扫 $N$ 个样本，太贵。**随机梯度下降**每次只抽一小撮（batch，典型 32~256）：

$$\nabla L \approx \frac{1}{B}\sum_{i \in \text{batch}} \nabla \ell_i$$

它是无偏估计，但带噪声——像有限温度下的 Langevin 动力学：
**平均仍往低处走，瞬时轨迹在等高线附近抖动**。这种噪声不是缺陷：
- 计算：batch 100 倍小，步数 100 倍多，性价比高
- 优化：噪声能帮参数跳出尖锐的小极小值（参考模拟退火：$T$ 高时探索，$T$ 低时收敛）

## 6. 动量：给石头一点惯性

$$v_{t+1} = \beta v_t - \eta \nabla L(\theta_t), \qquad \theta_{t+1} = \theta_t + v_{t+1}$$

超阻尼极限被打破了：现在有速度记忆。在峡谷状损失面（梯度在陡的方向大、缓的方向小）
上，动量会**平均掉陡方向的振荡**、**累积缓方向的速度**——练习第 1 题让你亲眼看到差别。

## 运行

```bash
python linear_regression.py   # 生成 3 张图 + 打印三种解法对比
python exercises.py           # 练习
```

产出图表（`figures/`）：
1. `fit.png` — 数据与三条拟合线（解析解 / 批量 GD / SGD 几乎重合）
2. `loss_surface.png` — 损失等高线 + GD 与 SGD 的参数轨迹（注意 SGD 的"热噪声"）
3. `learning_rates.png` — 四个学习率：0.01 慢、0.3 刚好、1.0 临界振荡、2.5 发散

## 练习

1. **动量**：实现 `gradient_descent_momentum`，在一个狭长的二次型上对比有无动量的轨迹。
2. **多项式拟合**：用正规方程 + 多项式特征拟合 $\sin x$，体验"模型容量"与过拟合。
3. **$R^2$**：回归的标准评分，理解"模型解释了多少方差"。

## 下一课预告

线性模型只能画直线。第 3 课我们让它输出概率（sigmoid / softmax），
第 4 课插入非线性层——从此进入神经网络的世界。
