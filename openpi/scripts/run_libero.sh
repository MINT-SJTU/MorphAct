#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash scripts/run_libero.sh train pi0|pi05 [--dry-run]
       bash scripts/run_libero.sh eval  pi0|pi05 [--dry-run]

Train: set BASE_CHECKPOINT,
ASSETS_BASE_DIR, HF_LEROBOT_HOME, and CUDA_VISIBLE_DEVICES. Set RESUME_CHECKPOINT
to resume with the same model, task selection, batch and optimizer schedule.
Resume uses checkpoint normalization assets; RESUME_NORM_STATS_SOURCE=assets
explicitly selects the current ASSETS_BASE_DIR instead. SAMPLER_SEED controls
DDP shuffling (default 0), independently of SEED (default 42).
Eval: set MORPHACT_CHECKPOINT, BASE_CHECKPOINT, RESULT_DIR, EVAL_PYTHON,
LIBERO_ROOT, LIBERO_CONFIG_PATH and CUDA_VISIBLE_DEVICES. LIBERO_SUITE selects
one suite (default libero_spatial). Results require a new RESULT_DIR.
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

if [[ $mode == train ]]; then
  case ${LIBERO_DATASET:-all} in
    spatial) config_name=${model}_libero_spatial ;;
    all) config_name=${model}_libero ;;
    *) echo 'LIBERO_DATASET must be spatial or all.' >&2; exit 2 ;;
  esac
  if [[ $config_name == pi05_libero_spatial ]]; then echo 'OpenPI has no pi05_libero_spatial config.' >&2; exit 2; fi
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
  exp_name=${EXP_NAME:-"morphact-${model}-${LIBERO_DATASET:-all}"}
  train_steps=50000
  [[ $model == pi0 ]] && train_steps=100000
  launcher=("$python_bin")
  if ((num_gpus > 1)); then
    launcher+=(-m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$num_gpus")
  fi
  command=(
    "${launcher[@]}" "$openpi_dir/scripts/train_morphact.py"
    --config-name "$config_name" --exp-name "$exp_name"
    --base-checkpoint "$base_checkpoint" --assets-base-dir "$assets_base_dir"
    --checkpoint-base-dir "$checkpoint_base_dir"
    --num-devices "$num_gpus" --per-device-batch-size "$per_gpu_batch_size"
    --gradient-accumulation-steps "$accumulation"
    --num-workers "${NUM_WORKERS:-8}"
    --num-train-steps "${NUM_TRAIN_STEPS:-$train_steps}"
    --warmup-steps "${WARMUP_STEPS:-1000}"
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
    --swanlab-project "${SWANLAB_PROJECT:-morphact}"
  )
  if [[ ${GRADIENT_CHECKPOINTING:-1} == 1 ]]; then command+=(--gradient-checkpointing); else command+=(--no-gradient-checkpointing); fi
  if [[ ${SWANLAB_ENABLED:-1} == 1 ]]; then command+=(--swanlab-enabled); else command+=(--no-swanlab-enabled); fi
  if [[ -n ${SWANLAB_WORKSPACE:-} ]]; then command+=(--swanlab-workspace "$SWANLAB_WORKSPACE"); fi
  if [[ -n ${RESUME_CHECKPOINT:-} ]]; then command+=(--resume-checkpoint "$RESUME_CHECKPOINT"); fi
  if [[ -n ${LOADER_WORKERS_OVERRIDE:-} ]]; then command+=(--loader-workers-override "$LOADER_WORKERS_OVERRIDE"); fi
  if [[ ${COMPILE_MORPHACT_POINTWISE:-0} == 1 ]]; then command+=(--compile-morphact-pointwise); fi
  if [[ -n ${TASK_INDICES:-} ]]; then
    IFS=',' read -r -a task_ids <<<"$TASK_INDICES"
    command+=(--task-indices "${task_ids[@]}")
  fi
  if ((dry_run)); then printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
  [[ -x $python_bin && -s $base_checkpoint ]] || { echo 'Missing OpenPI Python or base checkpoint.' >&2; exit 1; }
  if [[ -z ${RESUME_CHECKPOINT:-} || $resume_norm_stats_source == assets ]]; then
    [[ -f $assets_base_dir/$config_name/physical-intelligence/libero/norm_stats.json ]] || { echo 'Missing LIBERO normalization assets.' >&2; exit 1; }
  fi
  [[ -d $HF_LEROBOT_HOME/physical-intelligence/libero/data ]] || { echo 'Missing LIBERO LeRobot dataset.' >&2; exit 1; }
  if [[ -z ${RESUME_CHECKPOINT:-} && -e $checkpoint_base_dir/$config_name/$exp_name ]]; then
    echo 'Output exists; select a new EXP_NAME or RESUME_CHECKPOINT.' >&2; exit 1
  fi
  exec "${command[@]}"
fi

checkpoint=${MORPHACT_CHECKPOINT:-}
result_dir=${RESULT_DIR:-}
eval_python=${EVAL_PYTHON:-}
libero_root=${LIBERO_ROOT:-}
libero_config_path=${LIBERO_CONFIG_PATH:-}
if [[ -z $checkpoint || -z $result_dir || -z $eval_python || -z $libero_root || -z $libero_config_path ]]; then
  echo 'Eval requires MORPHACT_CHECKPOINT, RESULT_DIR, EVAL_PYTHON, LIBERO_ROOT and LIBERO_CONFIG_PATH.' >&2; exit 2
fi
suite=${LIBERO_SUITE:-libero_spatial}
case $suite in libero_spatial|libero_object|libero_goal|libero_10) ;; *) echo 'Unsupported LIBERO_SUITE.' >&2; exit 2;; esac
trials=${TRIALS_PER_TASK:-50}
default_replan=10
if [[ $model == pi0 && $suite == libero_10 ]]; then default_replan=20; fi
replan=${REPLAN_STEPS:-$default_replan}
port=${PORT:-8000}
eval_seed=${EVAL_SEED:-0}
eval_config=${model}_libero
if [[ ${LIBERO_DATASET:-all} == spatial ]]; then eval_config=${model}_libero_spatial; fi
if [[ $eval_config == pi05_libero_spatial ]]; then echo 'OpenPI has no pi05_libero_spatial config.' >&2; exit 2; fi
server=("$python_bin" "$openpi_dir/scripts/serve_morphact.py" --config "$eval_config" --morphact-checkpoint "$checkpoint" --base-checkpoint "$base_checkpoint" --device cuda --port "$port" --denoise-steps "${DENOISE_STEPS:-10}" --episode-seeded)
client=("$eval_python" "$openpi_dir/examples/libero/main.py" --args.host 127.0.0.1 --args.port "$port" --args.task-suite-name "$suite" --args.num-trials-per-task "$trials" --args.replan-steps "$replan" --args.seed "$eval_seed" --args.video-out-path "$result_dir/videos")
if ((dry_run)); then printf '%q ' "${server[@]}"; printf '\n'; printf '%q ' "${client[@]}"; printf '\n'; exit 0; fi
[[ -x $python_bin && -x $eval_python && -s $base_checkpoint && -s $checkpoint/adapter.safetensors && -s $checkpoint/metadata.json ]] || { echo 'Missing executable or model checkpoint.' >&2; exit 1; }
[[ -d $libero_root && -e $libero_config_path ]] || { echo 'Missing LIBERO source or config.' >&2; exit 1; }
[[ ! -e $result_dir ]] || { echo "Result directory exists: $result_dir" >&2; exit 1; }
mkdir -p "$result_dir/videos"
export PYTHONPATH="$PYTHONPATH:$libero_root"
export LIBERO_CONFIG_PATH="$libero_config_path"
export MUJOCO_GL=${MUJOCO_GL:-egl}
"${server[@]}" >"$result_dir/server.log" 2>&1 &
server_pid=$!
cleanup() { kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
for ((attempt=0; attempt<180; attempt++)); do
  if grep -q 'Serving MorphAct' "$result_dir/server.log"; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then cat "$result_dir/server.log" >&2; exit 1; fi
  sleep 2
done
grep -q 'Serving MorphAct' "$result_dir/server.log" || { echo 'Policy server did not become ready.' >&2; exit 1; }
"${client[@]}" >"$result_dir/eval.log" 2>&1
expected=$((10 * trials))
actual=$(grep -c '# episodes completed so far:' "$result_dir/eval.log" || true)
errors=$(grep -c 'Caught exception:' "$result_dir/eval.log" || true)
[[ $actual -eq $expected && $errors -eq 0 ]] || { echo "LIBERO incomplete: episodes=$actual/$expected errors=$errors" >&2; exit 1; }
