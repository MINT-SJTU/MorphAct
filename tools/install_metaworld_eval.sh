#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
usage() {
  cat <<'EOF'
Usage: bash tools/install_metaworld_eval.sh [--env-dir /absolute/path/to/metaworld-eval-env]

Install the MetaWorld evaluator environment. The default is
$MORPHACT_ARTIFACT_ROOT/envs/metaworld-eval, where
MORPHACT_ARTIFACT_ROOT defaults to the repository's artifacts/ directory.
EOF
}

env_dir=${MORPHACT_METAWORLD_EVAL_ENV:-}
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
  env_dir="$artifact_root/envs/metaworld-eval"
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
eval_dir=$repo_root/openpi/examples/metaworld
if [[ ! -x "$env_dir/bin/python" ]]; then
  mkdir -p "$(dirname "$env_dir")"
  "$uv_bin" venv "$env_dir" --python 3.11
fi
"$uv_bin" pip install --python "$env_dir/bin/python" -r "$eval_dir/requirements.txt"
"$uv_bin" pip install --python "$env_dir/bin/python" --no-deps \
  --editable "$repo_root/openpi/packages/openpi-client"
