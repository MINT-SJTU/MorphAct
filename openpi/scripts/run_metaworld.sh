#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_metaworld.sh train pi0|pi05 [--dry-run]
       bash scripts/run_metaworld.sh eval  pi0|pi05 [--dry-run]

Train: set BASE_CHECKPOINT, HF_LEROBOT_HOME,
ASSETS_BASE_DIR and CUDA_VISIBLE_DEVICES.
The Evo-1 MetaWorld LeRobot v2.1 revision and all 50 tasks are fixed in OpenPI.
RESUME_CHECKPOINT uses its normalization assets; RESUME_NORM_STATS_SOURCE=assets
explicitly selects the current ASSETS_BASE_DIR instead. SAMPLER_SEED controls
DDP shuffling (default 0), independently of SEED (default 42).
Eval: set MORPHACT_CHECKPOINT, BASE_CHECKPOINT, RESULT_DIR and CUDA_VISIBLE_DEVICES.
EVAL_PYTHON is the separate MetaWorld evaluator environment. Results require a
new RESULT_DIR and complete 50 tasks x 20 episodes with zero errors.
MORPHACT_ARTIFACT_ROOT defaults to the repository's artifacts/ directory.
HELP
}

mode=${1:-}
model=${2:-}
if [[ $mode == --help || $mode == -h || -z $mode ]]; then usage; exit 0; fi
if [[ $mode != train && $mode != eval ]]; then usage >&2; exit 2; fi
if [[ $model != pi0 && $model != pi05 ]]; then usage >&2; exit 2; fi
shift 2
dry_run=0
while (($#)); do
  case $1 in
    --dry-run) dry_run=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

openpi_dir=$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd -P "$openpi_dir/.." && pwd)
artifact_root=${MORPHACT_ARTIFACT_ROOT:-"$repo_root/artifacts"}
python_bin=${PYTHON_BIN:-$(command -v python || true)}
export PYTHONPATH="$repo_root:$openpi_dir/src:$openpi_dir/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export USE_TF=${USE_TF:-0}
export USE_FLAX=${USE_FLAX:-1}
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"$artifact_root/openpi/cache"}
export HF_HOME=${HF_HOME:-"$artifact_root/huggingface"}
export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-"$HF_HOME/lerobot"}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-"$artifact_root/pycache"}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

base_checkpoint=${BASE_CHECKPOINT:-}
if [[ -z $base_checkpoint ]]; then echo 'Set BASE_CHECKPOINT to matching model.safetensors.' >&2; exit 2; fi
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then echo 'Set CUDA_VISIBLE_DEVICES.' >&2; exit 2; fi
config_name=${model}_metaworld

if [[ $mode == train ]]; then
  num_gpus=${NUM_GPUS:-4}
  per_gpu_batch_size=${PER_GPU_BATCH_SIZE:-16}
  accumulation=${GRADIENT_ACCUMULATION_STEPS:-1}
  for value in "$num_gpus" "$per_gpu_batch_size" "$accumulation"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || { echo 'GPU and batch values must be positive integers.' >&2; exit 2; }
  done
  IFS=',' read -r -a gpu_ids <<<"$CUDA_VISIBLE_DEVICES"
  if ((${#gpu_ids[@]} != num_gpus)); then echo "NUM_GPUS=$num_gpus differs from CUDA_VISIBLE_DEVICES." >&2; exit 2; fi
  assets_base_dir=${ASSETS_BASE_DIR:-"$artifact_root/openpi/assets"}
  resume_norm_stats_source=${RESUME_NORM_STATS_SOURCE:-checkpoint}
  case $resume_norm_stats_source in checkpoint|assets) ;; *) echo 'RESUME_NORM_STATS_SOURCE must be checkpoint or assets.' >&2; exit 2 ;; esac
  checkpoint_base_dir=${CHECKPOINT_BASE_DIR:-"$artifact_root/checkpoints"}
  exp_name=${EXP_NAME:-"morphact-${model}-mt50"}
  launcher=("$python_bin")
  if ((num_gpus > 1)); then launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$num_gpus"); fi
  command=(
    "${launcher[@]}" "$openpi_dir/scripts/train_morphact.py"
    --config-name "$config_name" --exp-name "$exp_name"
    --base-checkpoint "$base_checkpoint" --assets-base-dir "$assets_base_dir"
    --checkpoint-base-dir "$checkpoint_base_dir"
    --num-devices "$num_gpus" --per-device-batch-size "$per_gpu_batch_size"
    --gradient-accumulation-steps "$accumulation"
    --num-workers "${NUM_WORKERS:-8}"
    --num-train-steps "${NUM_TRAIN_STEPS:-50000}"
    --warmup-steps "${WARMUP_STEPS:-500}"
    --peak-lr "${PEAK_LR:-2.5e-5}" --decay-lr "${DECAY_LR:-2.5e-6}"
    --morphact-width "${MORPHACT_WIDTH:-1024}"
    --vlm-morphact-layers "${VLM_MORPHACT_LAYERS:-8}"
    --action-morphact-layers "${ACTION_MORPHACT_LAYERS:-6}"
    --precision "${PRECISION:-bfloat16}"
    --morphact-compute-precision "${MORPHACT_COMPUTE_PRECISION:-bfloat16}"
    --adam-beta1 "${ADAM_BETA1:-0.9}" --adam-beta2 "${ADAM_BETA2:-0.95}"
    --adam-eps "${ADAM_EPS:-1e-8}" --weight-decay "${WEIGHT_DECAY:-1e-10}"
    --max-grad-norm "${MAX_GRAD_NORM:-1.0}"
    --save-interval "${SAVE_INTERVAL:-2500}" --log-interval "${LOG_INTERVAL:-10}"
    --seed "${SEED:-42}" --sampler-seed "${SAMPLER_SEED:-0}"
    --resume-norm-stats-source "$resume_norm_stats_source" --swanlab-mode "${SWANLAB_MODE:-local}"
    --swanlab-project "${SWANLAB_PROJECT:-morphact-mt50}"
  )
  if [[ ${GRADIENT_CHECKPOINTING:-1} == 1 ]]; then command+=(--gradient-checkpointing); else command+=(--no-gradient-checkpointing); fi
  if [[ ${SWANLAB_ENABLED:-1} == 1 ]]; then command+=(--swanlab-enabled); else command+=(--no-swanlab-enabled); fi
  if [[ -n ${SWANLAB_WORKSPACE:-} ]]; then command+=(--swanlab-workspace "$SWANLAB_WORKSPACE"); fi
  if [[ -n ${RESUME_CHECKPOINT:-} ]]; then command+=(--resume-checkpoint "$RESUME_CHECKPOINT"); fi
  if [[ ${COMPILE_MORPHACT_POINTWISE:-0} == 1 ]]; then command+=(--compile-morphact-pointwise); fi
  if ((dry_run)); then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
  [[ -x $python_bin && -s $base_checkpoint ]] || { echo 'Missing OpenPI Python or base checkpoint.' >&2; exit 1; }
  dataset="$HF_LEROBOT_HOME/MINT-SJTU/Evo1_MetaWorld_Dataset"
  [[ -f $dataset/meta/info.json && -f $dataset/meta/tasks.jsonl ]] || { echo 'Missing Evo-1 LeRobot v2.1 dataset metadata.' >&2; exit 1; }
  if [[ -z ${RESUME_CHECKPOINT:-} || $resume_norm_stats_source == assets ]]; then
    [[ -f $assets_base_dir/$config_name/MINT-SJTU/Evo1_MetaWorld_Dataset/norm_stats.json ]] || { echo 'Missing MetaWorld normalization assets.' >&2; exit 1; }
  fi
  if [[ -z ${RESUME_CHECKPOINT:-} && -e $checkpoint_base_dir/$config_name/$exp_name ]]; then
    echo 'Output exists; select a new EXP_NAME or RESUME_CHECKPOINT.' >&2; exit 1
  fi
  exec "${command[@]}"
fi

checkpoint=${MORPHACT_CHECKPOINT:-}
result_dir=${RESULT_DIR:-}
eval_python=${EVAL_PYTHON:-"$artifact_root/envs/metaworld-eval/bin/python"}
if [[ -z $checkpoint || -z $result_dir ]]; then echo 'Eval requires MORPHACT_CHECKPOINT and RESULT_DIR.' >&2; exit 2; fi
port=${PORT:-8000}
eval_seed=${EVAL_SEED:-0}
server=("$python_bin" "$openpi_dir/scripts/serve_morphact.py" --config "$config_name" --morphact-checkpoint "$checkpoint" --base-checkpoint "$base_checkpoint" --device cuda --port "$port" --denoise-steps "${DENOISE_STEPS:-10}" --episode-seeded)
episodes=${EPISODES_PER_TASK:-20}
default_replan=4
[[ $model == pi0 ]] && default_replan=3
client=("$eval_python" "$openpi_dir/examples/metaworld/evaluate.py" --host 127.0.0.1 --port "$port" --output-dir "$result_dir" --episodes-per-task "$episodes" --max-steps 400 --replan-steps "${REPLAN_STEPS:-$default_replan}" --seed "$eval_seed")
if ((dry_run)); then printf '%q ' "${server[@]}"; printf '\n'; printf '%q ' "${client[@]}"; printf '\n'; exit 0; fi
[[ -x $python_bin && -x $eval_python && -s $base_checkpoint && -s $checkpoint/adapter.safetensors && -s $checkpoint/metadata.json ]] || { echo 'Missing executable or model checkpoint.' >&2; exit 1; }
[[ ! -e $result_dir ]] || { echo "Result directory exists: $result_dir" >&2; exit 1; }
mkdir -p "$(dirname "$result_dir")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
"${server[@]}" >"${result_dir}.server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
for ((attempt=0; attempt<180; attempt++)); do
  if grep -q 'Serving MorphAct' "${result_dir}.server.log"; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then cat "${result_dir}.server.log" >&2; exit 1; fi
  sleep 2
done
grep -q 'Serving MorphAct' "${result_dir}.server.log" || { echo 'Policy server did not become ready.' >&2; exit 1; }
"${client[@]}"
"$eval_python" - "$result_dir/summary.json" "$episodes" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary["episodes"] != 50 * int(sys.argv[2]) or summary["errors"] != 0 or len(summary["per_task"]) != 50:
    raise SystemExit("MT50 evaluation is incomplete or has errors")
PY
