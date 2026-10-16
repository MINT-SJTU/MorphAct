# π0 与 π0.5 上的 MorphAct

[English](README.md) | **简体中文**

使用 MorphAct 预训练权重运行 π0 和 π0.5。

## 安装

在带 NVIDIA GPU 的 Linux 环境中安装 `uv`，然后从仓库根目录执行：

```bash
bash tools/install_openpi.sh
source artifacts/envs/openpi/bin/activate
```

## 预训练权重

准备对应的 PyTorch 格式 [OpenPI 基座权重](https://github.com/Physical-Intelligence/openpi#base-models)：

| 骨干 | 基座权重路径 |
|---|---|
| π0 | `ckpt/base/pi0/pi0_base.safetensors` |
| π0.5 | `ckpt/base/pi05/model.safetensors` |

<details>
<summary>将官方 JAX 基座转换为 π0.5 PyTorch 权重</summary>

在 OpenPI 环境中下载官方基座，并使用上游脚本转换：

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

转换 π0 时，使用 `pi0_base`、`--config-name pi0_libero` 和 `--output-path ckpt/base/pi0`，再将生成的 `model.safetensors` 重命名为 `pi0_base.safetensors`。

</details>

从 [MINT-SJTU/MorphAct](https://huggingface.co/MINT-SJTU/MorphAct) 下载 Meta-World 和 LIBERO 适配器，并放入 `ckpt/adapters/`。

| 骨干 | Meta-World | LIBERO |
|---|---|---|
| π0 | `pi0-metaworld/` | `pi0-libero/` |
| π0.5 | `pi05-metaworld/` | `pi05-libero/` |

每个适配器目录包含 `adapter.safetensors`、`metadata.json` 和 `assets/` 目录。

## 推理

使用 π0.5 Meta-World 检查点启动策略服务：

```bash
bash tools/serve.sh pi05 metaworld
```

推理默认启用加速：CUDA 上的 PG RMS/RoPE 融合、FP32 低秩计算脚本化，以及去噪 CUDA Graph 重放。首次请求会预热并录制 Graph。使用 `INFERENCE_ACCELERATION=0 bash tools/serve.sh pi05 metaworld`，或在直接启动服务时传入 `--no-inference-acceleration`，可切回普通推理；策略加载函数也接受 `inference_acceleration=False`。训练保持原生路径，权重格式不变。

使用 π0 时将模型参数换为 `pi0`。仿真评测步骤见[评测指南](../docs/evaluation.zh-CN.md)。OpenPI 的 Meta-World 默认评测不录制视频，LIBERO 评测会录制。

<details>
<summary>直接启动 WebSocket 服务</summary>

```bash
python openpi/scripts/serve_morphact.py \
  --config pi05_metaworld \
  --morphact-checkpoint ckpt/adapters/pi05-metaworld \
  --base-checkpoint ckpt/base/pi05/model.safetensors
```

客户端发送观测并接收动作块。接入示例见 [LIBERO 客户端](examples/libero/main.py)。

</details>

## 训练

激活上面的 OpenPI 环境，并设置数据和输出路径：

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

将 [LeRobot 格式的 LIBERO 数据集](https://huggingface.co/datasets/physical-intelligence/libero)下载到 `$HF_LEROBOT_HOME/physical-intelligence/libero`，然后执行：

```bash
python openpi/scripts/compute_norm_stats.py --config-name pi05_libero \
  --assets-base-dir "$ASSETS_BASE_DIR" --state-action-only --batch-size 256 --num-workers 0
source openpi/scripts/configs/morphact300m_libero.env
bash openpi/scripts/run_libero.sh train pi05
```

### Meta-World

使用 [LeRobot v2.1 格式的 Evo-1 MetaWorld 数据集](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset/tree/612c4e83d0e3b4e0be5227970bd8f5f07fc0c488)。在另一个终端设置上述训练路径后执行：

```bash
python openpi/examples/metaworld/prepare_data.py \
  --root "$HF_LEROBOT_HOME/MINT-SJTU/Evo1_MetaWorld_Dataset" --download --full-download
python openpi/scripts/compute_norm_stats.py --config-name pi05_metaworld \
  --assets-base-dir "$ASSETS_BASE_DIR" --state-action-only --batch-size 256 --num-workers 0
bash openpi/scripts/run_metaworld.sh train pi05
```

训练 π0 时，使用对应基座权重，并将配置名和启动参数中的 `pi05` 换为 `pi0`。输出保存在 `$MORPHACT_ARTIFACT_ROOT/checkpoints`。通过 `EXP_NAME` 指定实验名称，或通过 `RESUME_CHECKPOINT` 从已有检查点继续训练。
