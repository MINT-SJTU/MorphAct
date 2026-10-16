# Evaluation

**English** | [简体中文](evaluation.zh-CN.md)

Run MorphAct policies in Meta-World or LIBERO. Commands below run from the repository root. Download pretrained adapters for both benchmarks from [MINT-SJTU/MorphAct](https://huggingface.co/MINT-SJTU/MorphAct).

## Simulator setup

### Meta-World

```bash
bash tools/install_metaworld_eval.sh
```

### LIBERO

```bash
export MORPHACT_ARTIFACT_ROOT="$PWD/artifacts"
conda create --prefix "$MORPHACT_ARTIFACT_ROOT/envs/libero" python=3.8 pip -y
conda activate artifacts/envs/libero
python -m pip install -r openpi/examples/libero/requirements.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu113
mkdir -p "$MORPHACT_ARTIFACT_ROOT/sources"
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  "$MORPHACT_ARTIFACT_ROOT/sources/LIBERO"
python -m pip install -e "$MORPHACT_ARTIFACT_ROOT/sources/LIBERO" \
  -e openpi/packages/openpi-client
LIBERO_CONFIG_PATH="$MORPHACT_ARTIFACT_ROOT/libero-config" \
  python -c 'import libero.libero'
```

Complete LIBERO's path configuration when prompted.

## Meta-World

With a prepared π0 or π0.5 checkpoint, start the policy server in the OpenPI environment:

```bash
source artifacts/envs/openpi/bin/activate
bash tools/serve.sh pi05 metaworld
```

Run the simulator in another terminal:

```bash
source artifacts/envs/metaworld-eval/bin/activate
bash tools/evaluate.sh pi05 metaworld
```

Replace `pi05` with `pi0` for π0. For X-VLA, activate `artifacts/envs/xvla/bin/activate` in the server terminal and use `xvla` in both commands. OpenPI Meta-World evaluation does not record videos by default; X-VLA evaluation does.

## LIBERO

With a prepared π0 or π0.5 checkpoint, start the policy server in the OpenPI environment:

```bash
source artifacts/envs/openpi/bin/activate
bash tools/serve.sh pi05 libero
```

Run the simulator in another terminal:

```bash
conda activate artifacts/envs/libero
bash tools/evaluate.sh pi05 libero
```

Replace `pi05` with `pi0` for π0. For X-VLA, activate `artifacts/envs/xvla/bin/activate` in the server terminal and use `xvla` in both commands. LIBERO evaluation records videos by default.

## Outputs

Each successful run of `tools/evaluate.sh` writes two result files at the result-directory root:

- `episodes.jsonl`: one normalized record per episode.
- `summary.json`: final score, task and group metrics, episode counts, errors, and completion status.

Worker logs, videos, and X-VLA's `evaluation.json` run/resume metadata are retained in the same directory.

The terminal also prints group scores and the path to the summary. `success_rate` is the micro-average over all episodes, from 0 to 1. `mean_success_rate` is the equal-weight average of the four benchmark groups: Spatial, Object, Goal, and Long in LIBERO; Easy, Medium, Hard, and Very Hard in MT50. It is available only when the full benchmark is complete, has no duplicate episodes, and has no errors. For π0 or π0.5, set `LIBERO_SUITE` (for example, `LIBERO_SUITE=libero_spatial`) to run one LIBERO suite; then `complete` is false and `mean_success_rate` is null, while available per-task and group rates are still reported. X-VLA runs all four LIBERO suites.

By default, results go to the repository's `artifacts/evaluations/<model>-<benchmark>-seed0`. `EVAL_SEED` changes the seed suffix, `MORPHACT_ARTIFACT_ROOT` changes the artifact root, and `RESULT_DIR` sets the output directory for one run. Give each new run a new directory.

To continue an existing X-VLA evaluation, pass `--resume` with the same policy checkpoint, evaluation settings, and result directory. Completed episodes, including unsuccessful tasks, are kept; missing episodes and episodes interrupted by runtime errors are rerun. Previous error records are moved to `rollout-errors.jsonl` and excluded from the final score. OpenPI evaluation does not support resume.

```bash
RESULT_DIR=artifacts/evaluations/pi05-metaworld-run2 \
  bash tools/evaluate.sh pi05 metaworld
RESULT_DIR=artifacts/evaluations/xvla-metaworld-run1 \
  bash tools/evaluate.sh xvla metaworld --resume
LIBERO_SUITE=libero_spatial \
RESULT_DIR=artifacts/evaluations/pi05-libero-spatial-check \
  bash tools/evaluate.sh pi05 libero
```
