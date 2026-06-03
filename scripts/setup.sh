#!/usr/bin/env bash
# 幂等安装：在 skill/仓库目录下建独立 .venv 并装好 papercheck。
# 跨任何项目调用本 skill 都用这个隔离环境，不污染用户当前项目。
set -euo pipefail
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SKILL_DIR"

PY="$SKILL_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "[papercheck] 创建独立 venv (Python 3.12)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  else
    python3 -m venv .venv
  fi
fi

# 已装则秒过；未装则安装本包及依赖
if "$PY" -c "import papercheck" >/dev/null 2>&1; then
  echo "[papercheck] 已就绪：$PY"
else
  echo "[papercheck] 安装 papercheck 及依赖…"
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$SKILL_DIR/.venv" uv pip install -e . -q
  else
    "$SKILL_DIR/.venv/bin/pip" install -e . -q
  fi
  echo "[papercheck] 安装完成：$PY"
fi
