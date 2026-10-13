#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash tools/serve.sh pi0|pi05|xvla libero|metaworld [--dry-run]

Start policy servers only. Run tools/evaluate.sh separately in another terminal.
GPU_IDS defaults to CUDA_VISIBLE_DEVICES or 0; SERVERS_PER_GPU defaults to 1;
BASE_PORT defaults to 8000. For example, GPU_IDS=4 SERVERS_PER_GPU=3
BASE_PORT=18420 starts three LIBERO servers on ports 18420-18422.
Servers on different GPUs load in parallel; servers on one GPU load in order.
MORPHACT_ADAPTER optionally selects a directory under ckpt/adapters/.
DENOISE_STEPS sets Pi0/Pi0.5 sampling steps (default 10); changing it requires
restarting the policy servers. It does not set the executed action count.
Inference acceleration is enabled by default; INFERENCE_ACCELERATION=0 disables it.
Activate the model environment first; this script uses its python command.
Stop this script to stop its servers.
EOF
}

model=${1:-}
benchmark=${2:-}
if [[ $model == --help || $model == -h || -z $model ]]; then usage; exit 0; fi
if [[ $model != pi0 && $model != pi05 && $model != xvla ]] ||
   [[ $benchmark != libero && $benchmark != metaworld ]]; then usage >&2; exit 2; fi
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

repo_root=$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
artifact_root=${MORPHACT_ARTIFACT_ROOT:-"$repo_root/artifacts"}
ckpt_root=${MORPHACT_CKPT_ROOT:-"$repo_root/ckpt"}
python_bin=${PYTHON_BIN:-$(command -v python || true)}
[[ -n $python_bin ]] || { echo 'Activate the model environment first.' >&2; exit 1; }
gpu_list=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}
servers_per_gpu=${SERVERS_PER_GPU:-1}
base_port=${BASE_PORT:-8000}
[[ $servers_per_gpu =~ ^[1-9][0-9]*$ && $base_port =~ ^[1-9][0-9]*$ ]] ||
  { echo 'SERVERS_PER_GPU and BASE_PORT must be positive integers.' >&2; exit 2; }
IFS=',' read -r -a gpu_ids <<<"$gpu_list"
seen_gpus=,
for gpu in "${gpu_ids[@]}"; do
  [[ $gpu =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 2; }
  [[ $seen_gpus != *",$gpu,"* ]] || { echo "Duplicate GPU ID: $gpu" >&2; exit 2; }
  seen_gpus+="$gpu,"
done
((base_port + ${#gpu_ids[@]} * servers_per_gpu - 1 <= 65535)) ||
  { echo 'Port range exceeds 65535.' >&2; exit 2; }

if [[ $model == xvla ]]; then
  [[ -z ${DENOISE_STEPS+x} ]] ||
    { echo 'DENOISE_STEPS is supported by Pi0/Pi0.5 only; X-VLA LIBERO sets steps in each client request.' >&2; exit 2; }
  base=$ckpt_root/base/xvla-pt
  module=morphact_xvla.serve
  if [[ $benchmark == libero ]]; then
    adapter=xvla-libero
  else
    adapter=xvla-metaworld
    module=morphact_xvla.serve_metaworld
  fi
  checkpoint=$ckpt_root/adapters/${MORPHACT_ADAPTER:-$adapter}
  export PYTHONPATH="$repo_root:$repo_root/xvla/src${PYTHONPATH:+:$PYTHONPATH}"
  command=("$python_bin" -m "$module" --base-model "$base" --checkpoint "$checkpoint" --device cuda --steps 10 --episode-seeded)
else
  denoise_steps=${DENOISE_STEPS:-10}
  [[ $denoise_steps =~ ^[1-9][0-9]*$ ]] ||
    { echo 'DENOISE_STEPS must be a positive integer.' >&2; exit 2; }
  if [[ $model == pi0 ]]; then
    base=$ckpt_root/base/pi0/pi0_base.safetensors
  else
    base=$ckpt_root/base/pi05/model.safetensors
  fi
  case "$model:$benchmark" in
    pi0:libero) adapter=pi0-libero ;;
    pi0:metaworld) adapter=pi0-metaworld ;;
    pi05:libero) adapter=pi05-libero ;;
    pi05:metaworld) adapter=pi05-metaworld ;;
  esac
  checkpoint=$ckpt_root/adapters/${MORPHACT_ADAPTER:-$adapter}
  export PYTHONPATH="$repo_root:$repo_root/openpi/src:$repo_root/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
  export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} USE_TF=${USE_TF:-0} USE_FLAX=${USE_FLAX:-1}
  export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"$artifact_root/openpi/cache"}
  command=("$python_bin" "$repo_root/openpi/scripts/serve_morphact.py" --config "${model}_${benchmark}" --morphact-checkpoint "$checkpoint" --base-checkpoint "$base" --device cuda --denoise-steps "$denoise_steps" --episode-seeded)
fi
case ${INFERENCE_ACCELERATION:-1} in
  1) ;;
  0) command+=(--no-inference-acceleration) ;;
  *) echo 'INFERENCE_ACCELERATION must be 0 or 1.' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} TOKENIZERS_PARALLELISM=false

if ((dry_run)); then
  index=0
  for gpu in "${gpu_ids[@]}"; do
    for ((slot=0; slot<servers_per_gpu; slot++)); do
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
      printf '%q ' "${command[@]}" --port "$((base_port + index))"
      printf '\n'
      ((index+=1))
    done
  done
  exit 0
fi
[[ -x $python_bin && -s $checkpoint/adapter.safetensors && -s $checkpoint/metadata.json ]] ||
  { echo "Missing model Python or adapter: $python_bin $checkpoint" >&2; exit 1; }
if [[ $model == xvla ]]; then
  [[ -s $base/model.safetensors || -s $base/model.safetensors.index.json || -s $base/pytorch_model.bin || -s $base/pytorch_model.bin.index.json ]] ||
    { echo "Missing base model: $base" >&2; exit 1; }
  command -v curl >/dev/null || { echo 'curl is required to check server readiness.' >&2; exit 1; }
else
  [[ -s $base && -d $checkpoint/assets ]] || { echo "Missing Pi base or normalization assets: $base $checkpoint/assets" >&2; exit 1; }
fi

log_dir=${SERVER_LOG_DIR:-"$artifact_root/logs/servers/$model-$benchmark"}
mkdir -p "$log_dir"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((slot=0; slot<servers_per_gpu; slot++)); do
  round_pids=()
  round_ports=()
  round_logs=()
  round_start_lines=()
  for ((gpu_index=0; gpu_index<${#gpu_ids[@]}; gpu_index++)); do
    gpu=${gpu_ids[gpu_index]}
    port=$((base_port + gpu_index * servers_per_gpu + slot))
    if command -v ss >/dev/null && ss -H -ltn "sport = :$port" | grep -q .; then
      echo "Port is already in use: $port" >&2
      exit 1
    fi
    log=$log_dir/$port.log
    printf '\n[%s] GPU %s port %s\n' "$(date -Is)" "$gpu" "$port" >>"$log"
    start_line=$(wc -l <"$log")
    CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" --port "$port" >>"$log" 2>&1 &
    pid=$!
    pids+=("$pid")
    round_pids+=("$pid")
    round_ports+=("$port")
    round_logs+=("$log")
    round_start_lines+=("$start_line")
    echo "$model $benchmark GPU=$gpu port=$port PID=$pid log=$log"
  done
  for ((gpu_index=0; gpu_index<${#round_pids[@]}; gpu_index++)); do
    pid=${round_pids[gpu_index]}
    port=${round_ports[gpu_index]}
    log=${round_logs[gpu_index]}
    start_line=${round_start_lines[gpu_index]}
    ready=0
    for ((attempt=0; attempt<180; attempt++)); do
      if ! kill -0 "$pid" 2>/dev/null; then tail -n 30 "$log" >&2; exit 1; fi
      if [[ $model == xvla ]]; then
        if curl --silent --fail --max-time 1 "http://127.0.0.1:$port/openapi.json" >/dev/null; then ready=1; break; fi
      elif tail -n +"$((start_line + 1))" "$log" | grep -Fq "WebSocket server ready pid=$pid "; then
        ready=1; break
      fi
      sleep 2
    done
    ((ready)) || { echo "Server did not become ready: $log" >&2; exit 1; }
  done
done
echo "All ${#pids[@]} servers are ready. Run tools/evaluate.sh in another terminal."
wait -n "${pids[@]}"
echo 'A policy server exited; stopping the remaining servers.' >&2
exit 1
