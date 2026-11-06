"""环境自检脚本。

运行:
    python check_env.py
"""

import os
import sys
import time


def main() -> None:
    print("=" * 62)
    print(" AI Playground 环境自检")
    print("=" * 62)

    results = []

    # 1. Python 版本 -------------------------------------------------------
    v = sys.version_info
    ok = v >= (3, 10)
    results.append(ok)
    print(f"[{'OK' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro}"
          + ("" if ok else "  (需要 >= 3.10)"))

    # 2. numpy -------------------------------------------------------------
    try:
        import numpy as np
        print(f"[OK] numpy {np.__version__}")
        x = np.random.rand(1000, 1000)
        t0 = time.perf_counter()
        y = x @ x
        dt = time.perf_counter() - t0
        print(f"     1000x1000 矩阵乘法耗时 {dt * 1000:.1f} ms (BLAS 加速正常应 <100ms)")
        results.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] numpy: {e}")
        results.append(False)

    # 3. matplotlib --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无窗口环境也能存图
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot([0, 1, 2], [0, 1, 4], marker="o")
        ax.set_title("env test")
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "env_test.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print(f"[OK] matplotlib {matplotlib.__version__}  (测试图已存: {out})")
        results.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] matplotlib: {e}")
        results.append(False)

    # 4. torch -------------------------------------------------------------
    try:
        import torch
        print(f"[OK] torch {torch.__version__}")
        a = torch.rand(512, 512)
        b = a @ a
        print(f"     CPU 矩阵乘法 OK, 结果均值 {b.mean().item():.4f}")
        results.append(True)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] torch: {e}")
        results.append(False)
        return  # 后面的测试依赖 torch

    # 5. MPS 加速 (Apple Silicon) -------------------------------------------
    if torch.backends.mps.is_available():
        try:
            dev = torch.device("mps")
            a = torch.rand(1024, 1024, device=dev)
            torch.cuda.synchronize if False else None
            t0 = time.perf_counter()
            for _ in range(10):
                _ = a @ a
            dt = (time.perf_counter() - t0) / 10
            print(f"[OK] MPS (Apple GPU) 可用, 1024x1024 matmul 平均 {dt * 1000:.1f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] MPS 存在但测试失败: {e} (回退 CPU 也没问题)")
    else:
        print("[WARN] MPS 不可用, 将使用 CPU (课程照常可学, GPT 训练稍慢)")

    print("=" * 62)
    if all(results):
        print(" 全部通过! 接下来进入 01_numpy/ 开始第一课。")
    else:
        print(" 有失败项, 请把上面的输出发给我, 我来帮你排查。")
    print("=" * 62)


if __name__ == "__main__":
    main()
