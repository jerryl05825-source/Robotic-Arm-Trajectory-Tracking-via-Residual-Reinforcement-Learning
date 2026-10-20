#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -n "${NAO_PYTHON:-}" ]]; then
  py="$NAO_PYTHON"
elif command -v python >/dev/null && python -c 'import mujoco' 2>/dev/null; then
  py="$(command -v python)"
elif [[ -x /home/gpnao/miniconda3/envs/mujoco/bin/python ]]; then
  py=/home/gpnao/miniconda3/envs/mujoco/bin/python
else
  py=python3
fi
exec "$py" simulate.py "$@"
