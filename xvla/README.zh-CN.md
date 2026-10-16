# X-VLA 上的 MorphAct

[English](README.md) | **简体中文**

使用 MorphAct 预训练权重运行 X-VLA。

## 安装

在带 NVIDIA GPU 的 Linux 环境中安装 `uv`，然后从仓库根目录执行：

```bash
bash tools/install_xvla.sh
source artifacts/envs/xvla/bin/activate
```

## 预训练权重

将 [X-VLA-Pt 基座模型](https://huggingface.co/2toINF/X-VLA-Pt)放入 `ckpt/base/xvla-pt/`。从 [MINT-SJTU/MorphAct](https://huggingface.co/MINT-SJTU/MorphAct) 下载 X-VLA 的 Meta-World 和 LIBERO 适配器。

将下载的适配器放入 `ckpt/adapters/`：

| 基准 | 适配器目录 |
|---|---|
| Meta-World | `xvla-metaworld/` |
| LIBERO | `xvla-libero/` |

每个适配器目录包含 `adapter.safetensors` 和 `metadata.json`。

## 推理

使用 Meta-World 检查点启动策略服务：

```bash
bash tools/serve.sh xvla metaworld
```

仿真评测步骤见[评测指南](../docs/evaluation.zh-CN.md)。X-VLA 默认会为两个基准录制评测视频。

<details>
<summary>直接启动 HTTP 服务</summary>

```bash
xvla-morphact-serve-metaworld \
  --base-model ckpt/base/xvla-pt \
  --checkpoint ckpt/adapters/xvla-metaworld
```

向 `/act` 发送观测即可获得动作块。Meta-World 使用 `xvla-morphact-serve-metaworld`，权重路径为 `ckpt/adapters/xvla-metaworld`。请求示例见 [HTTP 客户端](examples/evaluate.py)。

两个服务入口默认开启推理加速：融合 PG 的 RMSNorm/RoPE、脚本化 FP32 低秩运算，并为去噪循环使用 CUDA Graph。每次请求都会更新噪声和观测对应的动态因子。统一启动脚本 `tools/serve.sh` 可设置 `INFERENCE_ACCELERATION=0`，直接启动服务则添加 `--no-inference-acceleration`，即可恢复 eager 执行；CPU 去噪自动走 eager 路径，训练构造默认关闭加速。

</details>

## 训练

激活上面的 X-VLA 环境，并设置输出根目录：

```bash
export MORPHACT_ARTIFACT_ROOT="$PWD/artifacts"
export HF_HOME="$MORPHACT_ARTIFACT_ROOT/cache/huggingface"
```

### LIBERO

按照 [X-VLA 预处理说明](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md)准备数据，将得到的训练元数据文件传给 `--train-metas`：

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

使用 [LeRobot v2.1 格式的 Evo-1 MetaWorld 数据集](https://huggingface.co/datasets/MINT-SJTU/Evo1_MetaWorld_Dataset/tree/612c4e83d0e3b4e0be5227970bd8f5f07fc0c488)：

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

继续训练时，在对应命令后添加 `--resume /path/to/checkpoint`。
