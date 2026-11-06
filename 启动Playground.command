#!/bin/bash
# 双击即可启动 AI Playground 并自动打开浏览器 (macOS)
# 原理: 在 ai-playground 目录起一个静态文件服务器, 端口 8734
cd "$(dirname "$0")" || exit 1        # 必须以 ai-playground 为根, 课程里的相对链接才有效

PORT="${PLAYGROUND_PORT:-8734}"
URL="http://127.0.0.1:${PORT}/playground/index.html"

# 已经有服务器在跑? 直接开页面
if curl -s -o /dev/null --max-time 2 "$URL"; then
  echo "✓ 服务器已在运行, 直接打开页面: $URL"
  open "$URL"
  exit 0
fi

# 找 python: 优先课程环境 aiplay, 退回系统 python3
PY="$HOME/miniconda3/envs/aiplay/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "✓ 启动本地服务器: $URL"
echo "  停止方式: 回到这个终端窗口按 Ctrl+C (或直接关掉终端窗口)"
open "$URL"
exec "$PY" -m http.server "$PORT" --bind 127.0.0.1
