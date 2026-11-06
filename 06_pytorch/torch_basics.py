"""第 06 课 · PyTorch 基础热身 (打印教学)。

运行: python torch_basics.py
"""

import torch


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(f" {title}")
    print("=" * 62)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(0)

    # ---------------------------------------------------------------- 1
    section("1. Tensor ≈ ndarray")
    import numpy as np
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(a)              # 零拷贝
    print(f"numpy → tensor:\n{t}")
    print(f"t @ t.T (矩阵乘) =\n{t @ t.T}")
    print(f"t.sum(0) = {t.sum(0)}, t.reshape(3,2) 首行 = {t.reshape(3, 2)[0]}")
    print("API 与 numpy 几乎一一对应; 默认 float32; 可搬到 GPU。")

    # ---------------------------------------------------------------- 2
    section("2. autograd: 第 5 课的 Value, 矩阵版")
    x = torch.tensor(2.0, requires_grad=True)
    y = x * x + x                        # dy/dx = 2x + 1
    y.backward()
    print(f"y = x² + x @ x=2: y={y.item()}, x.grad={x.item()}")
    assert abs(x.grad.item() - 5.0) < 1e-6
    print("与第 5 课 micrograd 的结果完全一致 ✓ (连 API 都几乎同名)")

    W = torch.randn(3, 2, requires_grad=True)
    X = torch.randn(4, 3)
    out = (X @ W).sum()
    out.backward()
    print(f"\n矩阵版: out = sum(X @ W), W.grad 形状 = {tuple(W.grad.shape)}")
    print("反传对整个矩阵一次完成 —— 每个元素都对, 但你不用再手推任何公式。")

    # ---------------------------------------------------------------- 3
    section("3. no_grad: 推理时关掉求导记录 (省内存)")
    x = torch.tensor(3.0, requires_grad=True)
    with torch.no_grad():
        y = x * 2                        # 不记录计算图
    print(f"no_grad 下的 y.requires_grad = {y.requires_grad}")
    print("评估/生成文本时永远套上它。")

    # ---------------------------------------------------------------- 4
    section("4. device: 把计算搬上 GPU (Apple Silicon = MPS)")
    dev = get_device()
    print(f"本机设备: {dev}")
    a = torch.randn(1024, 1024, device=dev)
    b = a @ a
    print(f"1024×1024 matmul 在 {dev} 上完成, 均值 {b.mean().item():.3f}")
    print("模型和数据必须在同一 device 上: net.to(device), X.to(device)。")

    # ---------------------------------------------------------------- 5
    section("5. 随机性与可复现")
    torch.manual_seed(42)
    print(f"两次初始化同一 seed → {torch.randn(3)} 与 ↓")
    torch.manual_seed(42)
    print(f"                    {torch.randn(3)} 完全一致")


if __name__ == "__main__":
    main()
