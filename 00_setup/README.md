# 第 00 课 · 环境搭建

## 已经为你做好的事

在安装阶段，我已经在你的机器上执行了等价于下面两条命令的操作：

```bash
conda create -y -n aiplay python=3.12        # 独立环境，不污染 base
~/miniconda3/envs/aiplay/bin/python -m pip install numpy matplotlib torch torchvision
```

为什么不用 base 环境的 Python 3.14？太新了，PyTorch 的轮子（预编译包）支持还不稳定。
深度学习生态通常滞后于最新 Python 半年到一年，**3.10~3.12 是安全区**。

## 日常使用

```bash
conda activate aiplay      # 进入环境（每次开新终端都要执行一次）
python check_env.py        # 自检
conda deactivate           # 退出环境
```

> 不想每次 activate？直接用绝对路径也行：
> `~/miniconda3/envs/aiplay/bin/python xxx.py`

## 环境里有什么

| 包 | 作用 | 对应课程 |
|---|---|---|
| numpy | 数值计算地基，数组运算 | 01~05 全靠它 |
| matplotlib | 画图（所有脚本自动存图到 `figures/`） | 全部 |
| torch | 深度学习框架（Apple Silicon 会用 **MPS** GPU 加速） | 06~10 |
| torchvision | 数据集 + 视觉工具 | 07 (MNIST) |

## 验证

```bash
python check_env.py
```

全部 `[OK]` 即通过。最后一项 **MPS** 显示 `True` 说明你的 Mac 芯片 (M 系列) 可以给
PyTorch 加速——第 10 课训练 GPT 时你会感谢它。

## 换一台机器怎么办

```bash
conda create -y -n aiplay python=3.12
conda activate aiplay
pip install -r requirements.txt
```

NVIDIA 显卡的机器（含 CUDA）同样适用，PyTorch 会自动检测；把脚本里的
`device = "mps" if ...` 换成 `device = "cuda" if torch.cuda.is_available() else "cpu"` 即可
（第 6 课会统一写成函数）。
