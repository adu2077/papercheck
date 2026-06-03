#!/usr/bin/env bash
# 用 skill 自己的 .venv 跑 papercheck，参数透传给 CLI。
set -euo pipefail
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$SKILL_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "papercheck 尚未安装，请先运行：bash \"$SKILL_DIR/scripts/setup.sh\"" >&2
  exit 1
fi
exec "$PY" -m papercheck "$@"
