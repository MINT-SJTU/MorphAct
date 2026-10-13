#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  cat <<'EOF'
Usage: bash tools/install_openpi.sh [--env-dir /absolute/path/to/openpi-env]

Create the OpenPI environment from the pinned lockfile, install the
local MORPHACT package, and apply the bundled Transformers compatibility files.

The default environment is $MORPHACT_ARTIFACT_ROOT/envs/openpi, where
MORPHACT_ARTIFACT_ROOT defaults to the repository's artifacts/ directory.
EOF
}

env_dir=${MORPHACT_OPENPI_ENV:-}
while (( $# )); do
  case "$1" in
    --env-dir)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
        echo '--env-dir requires an absolute path.' >&2
        exit 2
      fi
      env_dir=$2
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z $env_dir ]]; then
  artifact_root=${MORPHACT_ARTIFACT_ROOT:-"$repo_root/artifacts"}
  env_dir="$artifact_root/envs/openpi"
fi
case "$env_dir" in
  /*) ;;
  *) echo "Environment path must be absolute: $env_dir" >&2; exit 2 ;;
esac

uv_bin=${UV_BIN:-$(command -v uv || true)}
if [[ -z "$uv_bin" || ! -x "$uv_bin" ]]; then
  echo 'uv was not found. Install uv or set UV_BIN=/absolute/path/to/uv.' >&2
  exit 1
fi
# Compatibility patches must not mutate uv's shared hardlink cache.
mkdir -p "$(dirname "$env_dir")"
GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT="$env_dir" \
  "$uv_bin" sync --frozen --link-mode copy --project "$repo_root/openpi" --python 3.11
python_bin="$env_dir/bin/python"
"$uv_bin" pip install --python "$python_bin" --no-deps --editable "$repo_root"
"$python_bin" - "$repo_root" <<'PY'
from pathlib import Path
import shutil
import sys
import transformers

if transformers.__version__ != "4.53.2":
    raise SystemExit("The OpenPI integration requires Transformers 4.53.2")
root = Path(sys.argv[1])
source = root / "openpi/src/openpi/models_pytorch/transformers_replace"
destination = Path(transformers.__file__).resolve().parent
for path in source.rglob("*.py"):
    target = destination / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_suffix(target.suffix + ".before_morphact")
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(path, target)
print("Installed bundled OpenPI compatibility files into", destination)
PY
"$python_bin" -c 'import morphact; print("MORPHACT:", morphact.__file__)'
