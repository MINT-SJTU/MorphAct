# MorphAct for X-VLA

**English** | [简体中文](README.zh-CN.md)

Run X-VLA with pretrained MorphAct checkpoints.

## Installation

Run from the repository root on Linux with an NVIDIA GPU and `uv`:

```bash
bash tools/install_xvla.sh
source artifacts/envs/xvla/bin/activate
```

## Pretrained checkpoints

Place the [X-VLA-Pt base model](https://huggingface.co/2toINF/X-VLA-Pt) in `ckpt/base/xvla-pt/`. Download the X-VLA adapters for Meta-World and LIBERO from [MINT-SJTU/MorphAct](https://huggingface.co/MINT-SJTU/MorphAct).

Place the downloaded adapter under `ckpt/adapters/`:

| Benchmark | Adapter directory |
|---|---|
| Meta-World | `xvla-metaworld/` |
| LIBERO | `xvla-libero/` |

Each adapter directory contains `adapter.safetensors` and `metadata.json`.

## Inference

Start the policy server with the Meta-World checkpoint:

```bash
bash tools/serve.sh xvla metaworld
```

To run the simulator, follow the [evaluation guide](../docs/evaluation.md). X-VLA evaluation records videos for both benchmarks by default.

<details>
<summary>Direct HTTP serving</summary>

```bash
xvla-morphact-serve-metaworld \
  --base-model ckpt/base/xvla-pt \
  --checkpoint ckpt/adapters/xvla-metaworld
```

Send observations to `/act` to receive an action chunk. The command above uses `xvla-morphact-serve-metaworld`; see the [HTTP client](examples/evaluate.py) for a request example.

Both servers enable inference acceleration by default: fused PG RMSNorm/RoPE, scripted FP32 low-rank operations, and CUDA graphs for the denoising loop. Noise and observation-conditioned factors are refreshed for every request. Use `INFERENCE_ACCELERATION=0` with `tools/serve.sh`, or add `--no-inference-acceleration` to either direct server command, to use eager execution. CPU denoising falls back to eager execution; training starts with acceleration disabled.

</details>

## Training

Activate the X-VLA environment above and set an output root:

```bash
export MORPHACT_ARTIFACT_ROOT="$PWD/artifacts"
export HF_HOME="$MORPHACT_ARTIFACT_ROOT/cache/huggingface"
```

### LIBERO

Prepare the dataset with the [X-VLA preprocessing instructions](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md), then pass the resulting training metadata to `--train-metas`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  -m morphact_xvla.train \
  --base-model ckpt/base/xvla-pt \
  --train-metas /path/to/libero_train_meta.json \
  --morphact-config xvla/configs/generator-68m.json \
  --output-dir "$MORPHACT_ARTIFACT_ROOT/runs/xvla-libero" \
  --batch-size 8 --gradient-accumulation-steps 1
```

### Meta-World

Use [Evo-1 MetaWorld in LeRobot v2.1 format](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset/tree/612c4e83d0e3b4e0be5227970bd8f5f07fc0c488):

```bash
python -m morphact_xvla.prepare_metaworld \
  --dataset-dir "$MORPHACT_ARTIFACT_ROOT/data/Evo1_MetaWorld_Dataset"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m accelerate.commands.launch \
  --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 \
  -m morphact_xvla.train_metaworld \
  --base-model ckpt/base/xvla-pt \
  --train-metas "$MORPHACT_ARTIFACT_ROOT/data/Evo1_MetaWorld_Dataset" \
  --morphact-config xvla/configs/generator-68m.json \
  --processing-config xvla/configs/metaworld-processing.json \
  --output-dir "$MORPHACT_ARTIFACT_ROOT/runs/xvla-metaworld" \
  --batch-size 8 --gradient-accumulation-steps 1
```

To resume training, add `--resume /path/to/checkpoint` to the corresponding command.
