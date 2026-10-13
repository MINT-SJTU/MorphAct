#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash tools/evaluate.sh pi0|pi05|xvla libero|metaworld [--dry-run] [--resume]

Run the existing simulator client against servers started with tools/serve.sh.
Use the same GPU_IDS, SERVERS_PER_GPU and BASE_PORT in both terminals.
Activate the simulator environment first; this script uses its python command.
Install LIBERO in that environment and initialize its normal configuration.
RESULT_DIR defaults to artifacts/evaluations/ in the repository; set it to a new path
for another run. Only X-VLA supports --resume.
EOF
}

model=${1:-}
benchmark=${2:-}
if [[ $model == --help || $model == -h || -z $model ]]; then usage; exit 0; fi
if [[ $model != pi0 && $model != pi05 && $model != xvla ]] ||
   [[ $benchmark != libero && $benchmark != metaworld ]]; then usage >&2; exit 2; fi
shift 2
dry_run=0
resume=0
while (($#)); do
  case $1 in
    --dry-run) dry_run=1 ;;
    --resume) resume=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
if ((resume)) && [[ $model != xvla ]]; then
  echo 'Only the X-VLA evaluator supports --resume.' >&2
  exit 2
fi

repo_root=$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
artifact_root=${MORPHACT_ARTIFACT_ROOT:-"$repo_root/artifacts"}
eval_python=${EVAL_PYTHON:-$(command -v python || true)}
[[ -n $eval_python ]] || { echo 'Activate the simulator environment first.' >&2; exit 1; }
gpu_list=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}
servers_per_gpu=${SERVERS_PER_GPU:-1}
base_port=${BASE_PORT:-8000}
seed=${EVAL_SEED:-0}
[[ $servers_per_gpu =~ ^[1-9][0-9]*$ && $base_port =~ ^[1-9][0-9]*$ && $seed =~ ^[0-9]+$ ]] ||
  { echo 'SERVERS_PER_GPU, BASE_PORT and EVAL_SEED must be nonnegative integers; the first two must be positive.' >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<<"$gpu_list"
for gpu in "${gpu_ids[@]}"; do
  [[ $gpu =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 2; }
done
((base_port + ${#gpu_ids[@]} * servers_per_gpu - 1 <= 65535)) ||
  { echo 'Port range exceeds 65535.' >&2; exit 2; }

result_dir=${RESULT_DIR:-"$artifact_root/evaluations/$model-$benchmark-seed$seed"}
if [[ $benchmark == libero ]]; then
  libero_config=${LIBERO_CONFIG_PATH:-"$artifact_root/libero-config"}
  export LIBERO_CONFIG_PATH=$libero_config
fi
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$repo_root:$repo_root/openpi/src:$repo_root/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

finalize_report() {
  local episodes=$1
  shift
  local -a report_command
  report_command=("$eval_python" -m morphact.eval_report --model "$model" --benchmark "$benchmark"
    --output-dir "$result_dir" --seed "$seed" --episodes-per-task "$episodes" "$@")
  if ((dry_run)); then
    printf '%q ' "${report_command[@]}"; printf '\n'
  else
    "${report_command[@]}"
  fi
}

run_client() {
  local child_pid status=0
  "$@" &
  child_pid=$!
  trap 'kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  wait "$child_pid" || status=$?
  trap - EXIT INT TERM
  return "$status"
}

ports=()
worker_gpus=()
for gpu in "${gpu_ids[@]}"; do
  for ((slot=0; slot<servers_per_gpu; slot++)); do
    ports+=("$((base_port + ${#ports[@]}))")
    worker_gpus+=("$gpu")
  done
done
worker_count=${#ports[@]}

if [[ $model == xvla ]]; then
  if [[ $benchmark == libero ]]; then
    episodes=${EPISODES_PER_TASK:-50}
    replan=${REPLAN_STEPS:-10}
  else
    episodes=${EPISODES_PER_TASK:-20}
    replan=${REPLAN_STEPS:-4}
  fi
  command=("$eval_python" "$repo_root/xvla/examples/evaluate.py" --benchmark "$benchmark" --ports "${ports[@]}" --gpus "${worker_gpus[@]}" --seed "$seed" --episodes-per-task "$episodes" --replan-steps "$replan" --output-dir "$result_dir")
  [[ $benchmark == metaworld ]] && command+=(--max-steps 400)
  ((resume)) && command+=(--resume)
  if ((dry_run)); then
    printf '%q ' "${command[@]}"; printf '\n'
    finalize_report "$episodes"
    exit 0
  fi
  [[ -x $eval_python ]] || { echo "Missing simulator Python: $eval_python" >&2; exit 1; }
  if [[ $benchmark == libero ]]; then
    [[ -f $libero_config/config.yaml ]] || { echo "Missing LIBERO config: $libero_config/config.yaml" >&2; exit 1; }
  fi
  if ((!resume)) && [[ -e $result_dir ]]; then
    echo "Result directory exists: $result_dir" >&2; exit 1
  fi
  run_client "${command[@]}"
  finalize_report "$episodes"
  exit 0
fi

if [[ $benchmark == metaworld ]]; then
  eval_script=$repo_root/openpi/examples/metaworld/evaluate.py
  episodes=${EPISODES_PER_TASK:-20}
  if [[ $model == pi0 ]]; then
    replan=${REPLAN_STEPS:-3}
  else
    replan=${REPLAN_STEPS:-4}
  fi
  [[ $episodes =~ ^[1-9][0-9]*$ ]] || { echo 'EPISODES_PER_TASK must be positive.' >&2; exit 2; }
  ((worker_count <= 50 * episodes)) || { echo 'More policy servers than MetaWorld episodes.' >&2; exit 2; }
  common_args=(--episodes-per-task "$episodes" --max-steps 400 --replan-steps "$replan" --seed "$seed")
  if ((worker_count == 1)); then
    command=("$eval_python" "$eval_script" --host 127.0.0.1 --port "$base_port" --output-dir "$result_dir" "${common_args[@]}")
    if ((dry_run)); then
      printf '%q ' "${command[@]}"; printf '\n'
      finalize_report "$episodes"
      exit 0
    fi
    [[ -x $eval_python ]] || { echo "Missing simulator Python: $eval_python" >&2; exit 1; }
    [[ ! -e $result_dir ]] || { echo "Result directory exists: $result_dir" >&2; exit 1; }
    export CUDA_VISIBLE_DEVICES=${gpu_ids[0]} MUJOCO_EGL_DEVICE_ID=${gpu_ids[0]}
    run_client "${command[@]}"
    finalize_report "$episodes"
    exit 0
  fi
  if ((dry_run)); then
    for ((worker=0; worker<worker_count; worker++)); do
      printf 'CUDA_VISIBLE_DEVICES=%q MUJOCO_EGL_DEVICE_ID=%q ' "${worker_gpus[worker]}" "${worker_gpus[worker]}"
      printf '%q ' "$eval_python" "$eval_script" --host 127.0.0.1 --port "${ports[worker]}" --output-dir "$result_dir/worker-$worker" "${common_args[@]}" --worker-index "$worker" --num-workers "$worker_count"
      printf ' --queue-path %q' "$result_dir/episode-queue.cursor"
      printf '\n'
    done
    printf '%q ' "$eval_python" "$eval_script" --output-dir "$result_dir" "${common_args[@]}" --num-workers "$worker_count" --merge-workers
    printf ' --queue-path %q' "$result_dir/episode-queue.cursor"
    printf '\n'
    finalize_report "$episodes"
    exit 0
  fi
  [[ -x $eval_python ]] || { echo "Missing simulator Python: $eval_python" >&2; exit 1; }
  [[ ! -e $result_dir ]] || { echo "Result directory exists: $result_dir" >&2; exit 1; }
  mkdir -p "$result_dir"
  client_pids=()
  cleanup_clients() {
    for pid in "${client_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${client_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  }
  trap cleanup_clients EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  for ((worker=0; worker<worker_count; worker++)); do
    CUDA_VISIBLE_DEVICES=${worker_gpus[worker]} MUJOCO_EGL_DEVICE_ID=${worker_gpus[worker]} \
      "$eval_python" "$eval_script" --host 127.0.0.1 --port "${ports[worker]}" \
      --output-dir "$result_dir/worker-$worker" "${common_args[@]}" \
      --worker-index "$worker" --num-workers "$worker_count" \
      --queue-path "$result_dir/episode-queue.cursor" \
      >"$result_dir/worker-$worker.log" 2>&1 &
    client_pids+=("$!")
  done
  echo "Evaluating $model MetaWorld: 50 tasks x $episodes episodes, workers=$worker_count, output=$result_dir"
  remaining=$worker_count
  while ((remaining)); do
    if ! wait -n; then
      echo "MetaWorld worker failed: inspect $result_dir/worker-*.log" >&2
      exit 1
    fi
    remaining=$((remaining - 1))
  done
  client_pids=()
  "$eval_python" "$eval_script" --output-dir "$result_dir" "${common_args[@]}" \
    --num-workers "$worker_count" --merge-workers \
    --queue-path "$result_dir/episode-queue.cursor"
  finalize_report "$episodes"
  exit
fi

suite_selection=${LIBERO_SUITE:-all}
case $suite_selection in
  all) suites=(libero_spatial libero_object libero_goal libero_10) ;;
  libero_spatial|libero_object|libero_goal|libero_10) suites=("$suite_selection") ;;
  *) echo "Invalid LIBERO_SUITE: $suite_selection" >&2; exit 2 ;;
esac
trials=${EPISODES_PER_TASK:-50}
[[ $trials =~ ^[1-9][0-9]*$ ]] || { echo 'EPISODES_PER_TASK must be positive.' >&2; exit 2; }
((worker_count <= 10 * trials)) ||
  { echo "Too many policy servers for a 10-task suite with $trials episodes per task: $worker_count" >&2; exit 2; }
if ((!dry_run)); then
  [[ -x $eval_python && -f $libero_config/config.yaml ]] ||
    { echo 'Missing simulator Python or LIBERO config.' >&2; exit 1; }
  [[ ! -e $result_dir ]] || { echo "Result directory exists: $result_dir" >&2; exit 1; }
  mkdir -p "$result_dir"
fi
libero_worker() {
  local worker=$1 suite replan video_dir log child_pid= status
  local -a command
  if ((!dry_run)); then
    trap - EXIT
    trap 'if [[ -n $child_pid ]]; then kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; fi; exit 143' INT TERM
  fi
  for suite in "${suites[@]}"; do
    replan=${REPLAN_STEPS:-10}
    if ((worker_count == 1)); then
      video_dir=$result_dir/$suite/videos
      log=$result_dir/$suite/eval.log
    else
      video_dir=$result_dir/$suite/worker-$worker/videos
      log=$result_dir/$suite/worker-$worker.log
    fi
    command=("$eval_python" "$repo_root/openpi/examples/libero/main.py" --args.host 127.0.0.1 --args.port "${ports[worker]}" --args.task-suite-name "$suite" --args.num-trials-per-task "$trials" --args.replan-steps "$replan" --args.seed "$seed" --args.worker-index "$worker" --args.num-workers "$worker_count" --args.video-out-path "$video_dir")
    command+=(--args.queue-path "$result_dir/$suite/episode-queue.cursor"
      --args.episode-out-path "$(dirname "$video_dir")/episodes.jsonl")
    if ((dry_run)); then
      printf 'CUDA_VISIBLE_DEVICES=%q MUJOCO_EGL_DEVICE_ID=%q ' "${worker_gpus[worker]}" "${worker_gpus[worker]}"
      printf '%q ' "${command[@]}"
      printf '\n'
      continue
    fi
    mkdir -p "$(dirname "$log")"
    CUDA_VISIBLE_DEVICES=${worker_gpus[worker]} MUJOCO_EGL_DEVICE_ID=${worker_gpus[worker]} \
      "${command[@]}" >"$log" 2>&1 &
    child_pid=$!
    if wait "$child_pid"; then
      child_pid=
    else
      status=$?
      echo "LIBERO worker failed: suite=$suite worker=$worker status=$status log=$log" >&2
      return "$status"
    fi
  done
}
if ((dry_run)); then
  for ((worker=0; worker<worker_count; worker++)); do libero_worker "$worker"; done
  finalize_report "$trials" --suites "${suites[@]}"
  exit 0
fi
client_pids=()
cleanup_clients() {
  if ((${#client_pids[@]})); then
    for pid in "${client_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${client_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}
trap cleanup_clients EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((worker=0; worker<worker_count; worker++)); do
  libero_worker "$worker" &
  client_pids+=("$!")
done
echo "Evaluating $model LIBERO: ${#suites[@]} suites x 10 tasks x $trials episodes, workers=$worker_count, output=$result_dir"
remaining=$worker_count
while ((remaining)); do
  if ! wait -n; then
    echo "LIBERO worker failed: inspect $result_dir/*/*.log" >&2
    exit 1
  fi
  remaining=$((remaining - 1))
done
client_pids=()
finalize_report "$trials" --suites "${suites[@]}"
