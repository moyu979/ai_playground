#!/bin/bash
# 一键验证 ai-playground 全部课程代码 (依赖 aiplay conda 环境; 其他机器会退回 python3)
# 用法: bash _run_all_tests.sh 2>&1 | tee _test_log.txt
PY="${HOME}/miniconda3/envs/aiplay/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
cd "$(dirname "$0")"

run() {
  echo ""
  echo "=========================================================="
  echo ">>> 运行 $1"
  echo "=========================================================="
  "$PY" -u "$1" || echo "!!! FAILED: $1"
}

# ---------- 快速组: 第 1~6、9 课演示 ----------
run 01_numpy/numpy_crash.py
run 01_numpy/solutions.py
run 02_gradient_descent/linear_regression.py
run 02_gradient_descent/solutions.py
run 03_logistic_regression/logistic_regression.py
run 03_logistic_regression/solutions.py
run 04_neural_network/train_toy.py
run 04_neural_network/solutions.py
run 05_autograd/micrograd.py
run 05_autograd/mlp_micrograd.py
run 05_autograd/solutions.py
run 06_pytorch/torch_basics.py
run 06_pytorch/mlp_pytorch.py
run 06_pytorch/common_bugs.py
run 06_pytorch/solutions.py
run 09_attention/attention_numpy.py
run 09_attention/self_attention_torch.py
run 09_attention/solutions.py
run 10_transformer/model.py
run 10_transformer/solutions.py

# ---------- 慢速组: 第 7、8、10 课 (训练) ----------
run 07_cnn/conv_numpy_demo.py
run 07_cnn/train_mnist.py --epochs 3
run 07_cnn/mlp_vs_cnn.py
run 07_cnn/solutions.py
run 08_sequence_models/bigram.py
run 08_sequence_models/mlp_lm.py
run 08_sequence_models/rnn_char.py
run 08_sequence_models/solutions.py
run 10_transformer/ablation.py
run 10_transformer/train.py

echo ""
echo "=========================================================="
echo ">>> 全部脚本执行完毕 (grep FAILED 查看失败项)"
echo "=========================================================="
grep -c "FAILED" /dev/null 2>/dev/null || true
