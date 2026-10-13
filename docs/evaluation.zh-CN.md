# 评测

[English](evaluation.md) | **简体中文**

在 Meta-World 或 LIBERO 中运行 MorphAct 策略。以下命令均从仓库根目录执行。Meta-World 预训练适配器可从 [MINT-SJTU/MorphAct-Metaworld](https://huggingface.co/MINT-SJTU/MorphAct-Metaworld) 下载。LIBERO 预训练检查点 Coming soon。

## 仿真环境

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

首次导入 LIBERO 时，按照提示完成路径配置。

## Meta-World

准备好 π0 或 π0.5 检查点后，在 OpenPI 环境中启动策略服务：

```bash
source artifacts/envs/openpi/bin/activate
bash tools/serve.sh pi05 metaworld
```

在另一个终端运行仿真器：

```bash
source artifacts/envs/metaworld-eval/bin/activate
bash tools/evaluate.sh pi05 metaworld
```

评测 π0 时将 `pi05` 换为 `pi0`。评测 X-VLA 时，在服务终端激活 `artifacts/envs/xvla/bin/activate`，并将两条命令中的模型参数换为 `xvla`。OpenPI 的 Meta-World 默认评测不录制视频；X-VLA 评测会录制。

## LIBERO

准备好 π0 或 π0.5 检查点后，在 OpenPI 环境中启动策略服务：

```bash
source artifacts/envs/openpi/bin/activate
bash tools/serve.sh pi05 libero
```

在另一个终端运行仿真器：

```bash
conda activate artifacts/envs/libero
bash tools/evaluate.sh pi05 libero
```

评测 π0 时将 `pi05` 换为 `pi0`。评测 X-VLA 时，在服务终端激活 `artifacts/envs/xvla/bin/activate`，并将两条命令中的模型参数换为 `xvla`。LIBERO 默认评测会录制视频。

## 输出

每次成功运行 `tools/evaluate.sh` 后，运行目录根部都会生成两个结果文件：

- `episodes.jsonl`：统一格式的逐回合记录。
- `summary.json`：最终得分、逐任务和分组指标、回合数量、错误数及完成状态。

同一目录也保留 worker 日志、视频，以及 X-VLA 用于运行和恢复的 `evaluation.json` 元数据。

终端也会打印分组得分和汇总文件路径。`success_rate` 是全部回合的微平均，取值为 0 到 1。`mean_success_rate` 是四个正式分组的等权平均：LIBERO 的 Spatial、Object、Goal、Long；MT50 的 Easy、Medium、Hard、Very Hard。只有完整评测、没有重复回合且没有错误时才会提供该均值。π0 和 π0.5 可设置 `LIBERO_SUITE`（例如 `LIBERO_SUITE=libero_spatial`）只运行一个 LIBERO suite；此时 `complete` 为 false、`mean_success_rate` 为 null，但仍会报告可用的逐任务和分组得分。X-VLA 运行全部四个 LIBERO suite。

默认结果目录为仓库内的 `artifacts/evaluations/<model>-<benchmark>-seed0`。设置 `EVAL_SEED` 会更改目录名中的 seed 后缀；设置 `MORPHACT_ARTIFACT_ROOT` 可更换产物根目录；设置 `RESULT_DIR` 可指定本轮目录。每轮新评测使用新目录。

恢复已有 X-VLA 评测时，保持策略检查点、评测设置和结果目录一致，并传入 `--resume`。已完成回合（包括正常运行但任务未成功的回合）会保留；尚未完成和因运行异常中断的回合会重新执行。此前的错误记录移入 `rollout-errors.jsonl`，不计入最终得分。OpenPI 评测不支持恢复。

```bash
RESULT_DIR=artifacts/evaluations/pi05-metaworld-run2 \
  bash tools/evaluate.sh pi05 metaworld
RESULT_DIR=artifacts/evaluations/xvla-metaworld-run1 \
  bash tools/evaluate.sh xvla metaworld --resume
LIBERO_SUITE=libero_spatial \
RESULT_DIR=artifacts/evaluations/pi05-libero-spatial-check \
  bash tools/evaluate.sh pi05 libero
```
