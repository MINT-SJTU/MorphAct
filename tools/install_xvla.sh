#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() {
  cat <<'EOF'
Usage: bash tools/install_xvla.sh [--env-dir /absolute/path/to/xvla-env]

Create the pinned X-VLA environment. The default is
$MORPHACT_ARTIFACT_ROOT/envs/xvla, where MORPHACT_ARTIFACT_ROOT defaults to
the repository's artifacts/ directory.
EOF
}

ENV_DIR=${MORPHACT_XVLA_ENV:-}
while (( $# )); do
  case "$1" in
    --env-dir)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then
        echo '--env-dir requires an absolute path.' >&2
        exit 2
      fi
      ENV_DIR=$2
      shift 2
      ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -z $ENV_DIR ]]; then
  ARTIFACT_ROOT=${MORPHACT_ARTIFACT_ROOT:-"$REPO_ROOT/artifacts"}
  ENV_DIR="$ARTIFACT_ROOT/envs/xvla"
fi
case "$ENV_DIR" in
  /*) ;;
  *) echo "Environment path must be absolute: $ENV_DIR" >&2; exit 2 ;;
esac

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "Install uv or set UV_BIN to its absolute path." >&2
  exit 1
fi

mkdir -p "$(dirname "$ENV_DIR")"
UV_PROJECT_ENVIRONMENT="$ENV_DIR" \
  "$UV_BIN" sync --project "$REPO_ROOT/xvla" --python 3.11 --locked --all-extras
