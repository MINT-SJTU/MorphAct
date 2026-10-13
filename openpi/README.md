# MorphAct for π0 and π0.5

**English** | [简体中文](README.zh-CN.md)

Run π0 and π0.5 with pretrained MorphAct checkpoints.

## Installation

Run from the repository root on Linux with an NVIDIA GPU and `uv`:

```bash
bash tools/install_openpi.sh
source artifacts/envs/openpi/bin/activate
```

## Pretrained checkpoints

Prepare the corresponding [OpenPI base weights](https://github.com/Physical-Intelligence/openpi#base-models) in PyTorch format:

| Backbone | Base-model path |
|---|---|
| π0 | `ckpt/base/pi0/pi0_base.safetensors` |
| π0.5 | `ckpt/base/pi05/model.safetensors` |

<details>
<summary>Prepare π0.5 base weights from the official JAX checkpoint</summary>

In the OpenPI environment, download the official base and convert it with the upstream script:

```bash
export OPENPI_DATA_HOME="$PWD/artifacts/cache/openpi"
export JAX_PLATFORMS=cpu USE_TF=0 USE_FLAX=1
mkdir -p artifacts/tools
curl -fsSL https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/convert_jax_model_to_pytorch.py \
  -o artifacts/tools/convert_jax_model_to_pytorch.py
JAX_BASE="$(python -c 'from openpi.shared.download import maybe_download; print(maybe_download("gs://openpi-assets/checkpoints/pi05_base"))')"
CUDA_VISIBLE_DEVICES="" python artifacts/tools/convert_jax_model_to_pytorch.py \
  --checkpoint-dir "$JAX_BASE" --config-name pi05_libero \
  --output-path ckpt/base/pi05 --precision bfloat16
```

For π0, use `pi0_base`, `--config-name pi0_libero`, and `--output-path ckpt/base/pi0`, then rename the resulting `model.safetensors` to `pi0_base.safetensors`.

</details>

Download the Meta-World adapters below from [MINT-SJTU/MorphAct-Metaworld](https://huggingface.co/MINT-SJTU/MorphAct-Metaworld) and place them under `ckpt/adapters/`. LIBERO pretrained checkpoints are coming soon.

| Backbone | Meta-World | LIBERO |
|---|---|---|
| π0 | `pi0-metaworld/` | Coming soon |
| π0.5 | `pi05-metaworld/` | Coming soon |

Each adapter directory contains `adapter.safetensors`, `metadata.json`, and its `assets/` directory.

## Inference

Start the policy server with the π0.5 Meta-World checkpoint:

```bash
bash tools/serve.sh pi05 metaworld
```

Inference acceleration is enabled by default: fused PG RMS/RoPE on CUDA, scripted FP32 low-rank updates, and CUDA Graph replay of denoising. The first request warms up and captures the graph. Use `INFERENCE_ACCELERATION=0 bash tools/serve.sh pi05 metaworld` or `--no-inference-acceleration` for direct serving to use eager inference. The policy factory also accepts `inference_acceleration=False`. Training keeps its native path; checkpoint formats are unchanged.

Use `pi0` for π0. To run the simulator, follow the [evaluation guide](../docs/evaluation.md). For Meta-World, the default evaluator does not record videos; LIBERO evaluation records them.

<details>
<summary>Direct WebSocket serving</summary>

```bash
python openpi/scripts/serve_morphact.py \
  --config pi05_metaworld \
  --morphact-checkpoint ckpt/adapters/pi05-metaworld \
  --base-checkpoint ckpt/base/pi05/model.safetensors
```

The client sends an observation and receives an action chunk. See the [LIBERO client](examples/libero/main.py) for an integration example.

</details>

## Training

Activate the OpenPI environment above and set your data and output paths:

```bash
export MORPHACT_ARTIFACT_ROOT="$PWD/artifacts"
export HF_HOME="$MORPHACT_ARTIFACT_ROOT/cache/huggingface"
export HF_LEROBOT_HOME="$HF_HOME/lerobot"
export ASSETS_BASE_DIR="$MORPHACT_ARTIFACT_ROOT/openpi/assets"
export BASE_CHECKPOINT="$PWD/ckpt/base/pi05/model.safetensors"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export USE_TF=0 USE_FLAX=1 JAX_PLATFORMS=cpu
```

### LIBERO

Download [LIBERO in LeRobot format](https://huggingface.co/datasets/physical-intelligence/libero) to `$HF_LEROBOT_HOME/physical-intelligence/libero`, then run:

```bash
python openpi/scripts/compute_norm_stats.py --config-name pi05_libero \
  --assets-base-dir "$ASSETS_BASE_DIR" --state-action-only --batch-size 256 --num-workers 0
source openpi/scripts/configs/morphact300m_libero.env
bash openpi/scripts/run_libero.sh train pi05
```

### Meta-World

Use [Evo-1 MetaWorld in LeRobot v2.1 format](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset/tree/612c4e83d0e3b4e0be5227970bd8f5f07fc0c488). In a separate shell with the training paths above, run:

```bash
python openpi/examples/metaworld/prepare_data.py \
  --root "$HF_LEROBOT_HOME/MINT-SJTU/Evo1_MetaWorld_Dataset" --download --full-download
python openpi/scripts/compute_norm_stats.py --config-name pi05_metaworld \
  --assets-base-dir "$ASSETS_BASE_DIR" --state-action-only --batch-size 256 --num-workers 0
bash openpi/scripts/run_metaworld.sh train pi05
```

For π0 training, use its base weights and replace `pi05` with `pi0` in the configuration and launch arguments. Outputs go to `$MORPHACT_ARTIFACT_ROOT/checkpoints`. Set `EXP_NAME` to name a run, or `RESUME_CHECKPOINT` to continue from a saved checkpoint.
