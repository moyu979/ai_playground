# 第 06 课 · PyTorch 入门 — 你的 micrograd, 工业级版本

> PyTorch = 你已经会的东西的工业级实现：
> **NumPy 的张量 API + 第 5 课的自动微分引擎（张量版）+ GPU/MPS 加速 + 现成的神经网络积木**。
> 本课结束后，你写的每一行 PyTorch 都能对应到前 5 课的某个公式。

## 1. Tensor ≈ ndarray（API 几乎照抄）

```python
import torch
X = torch.tensor([[1., 2.], [3., 4.]])   # 对应 np.array
X = torch.from_numpy(arr)                # 零拷贝互转 (CPU)
X @ X.T, X.sum(0), X.reshape(2, 2)       # 运算符一一对应
```

差别只有三处：
1. `dtype` 默认 `float32`（NumPy 是 64）——深度学习用单精度换速度
2. 张量可以住在 **device** 上：`"cpu" / "mps" (Apple GPU) / "cuda" (NVIDIA)`
3. 张量可以携带**求导记录**

## 2. autograd：第 5 课的 Value, 矩阵版

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * x + x          # 和第 5 课演示 2 同一个表达式
y.backward()
x.grad                 # tensor(5.) —— 2x+1, 对上了!
```

概念对照：

| micrograd (第 5 课) | PyTorch |
|---|---|
| `Value.data` | `tensor.data` / `tensor.detach()` |
| `Value.grad` | `tensor.grad` |
| `loss.backward()` | `loss.backward()` (一模一样!) |
| 手动 `p.grad = 0.0` | `optimizer.zero_grad()` |
| 关闭求导记录 | `with torch.no_grad():` |

> `zero_grad` 为什么存在？第 5 课讲过梯度是**累加**的（分叉变量 +=）。
> PyTorch 默认延续这一语义，所以每步训练前要把上一步的梯度清零。
> 忘了清零 = 梯度随时间求和 = 训练诡异发散 —— 新手最常见 bug 之一。

## 3. nn.Module：层的容器与参数管理器

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)     # = 第 4 课的 Dense(2→64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))     # = Activation('tanh')
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)              # 输出 logits (softmax 在损失里)
```

继承 `nn.Module` 后自动获得：
- `net.parameters()` —— 所有可学习参数的迭代器（交给 optimizer）
- `net.train() / net.eval()` —— 切换 Dropout/BN 行为（第 7 课会遇到）
- `net.to(device)` —— 一行把全部参数搬上 GPU

**nn.CrossEntropyLoss = softmax + NLL 的融合稳定版**（第 3 课 log-sum-exp 的实现），
输入是 logits，不是概率——所以网络最后一层不加激活。

## 4. 训练循环：五步真言（全课程通用模板，背下来）

```python
for xb, yb in batches:
    logits = net(xb)                     # ① 前向
    loss = F.cross_entropy(logits, yb)   # ② 损失
    optimizer.zero_grad()                # ③ 清梯度 (§2 的坑)
    loss.backward()                      # ④ 反传 (第 5 课的引擎)
    optimizer.step()                     # ⑤ 更新 (第 2 课的 SGD)
```

`optimizer.step()` 做的事就是第 4 课 `Dense.step(lr)` 里的
`W -= lr * dW`，只是对所有参数批量执行。Adam 等花式优化器是更新规则的变体
（动量 = 第 2 课练习 1；自适应学习率 = 练习 3）。

## 5. 运行

```bash
python torch_basics.py     # 张量/autograd 热身 (打印教学)
python mlp_pytorch.py      # 用 PyTorch 重打第 4 课的螺旋之战, 对比损失曲线
python common_bugs.py      # 反面教材: 忘记 zero_grad / step 的对照实验
python exercises.py
```

`mlp_pytorch.py` 会自动加载第 4 课保存的 NumPy 损失曲线
(`../04_neural_network/figures/spiral_loss.pkl`) 画在同图对比——
**同一个网络、同一个数据，两种实现应基本重合**。这是"手写=框架"的直接证据。

## 练习

1. **nn.Sequential**：不用 class，用 `nn.Sequential(nn.Linear(...), nn.Tanh(), ...)`
   重建同结构网络并训练。
2. **早停**：划分 train/val，监控 val 准确率，保存最佳模型状态
   (`copy.deepcopy(net.state_dict())`)，返回最佳 val 准确率。
3. **优化器对决**：SGD vs Adam，同一网络各跑一遍，画损失对比。

## 下一课预告

MLP 把输入当"一袋独立数字"。图像有空间结构——相邻像素相关、
平移不该改变结论。第 7 课用**卷积**把这种结构先验写进网络
（物理人：这就是"系统具有平移对称性 → 算符可交换 → 用 Green 函数思考"）。
