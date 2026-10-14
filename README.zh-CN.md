<h1 align="center"><img src="assets/morphact-title-zh.svg" width="749" alt="MorphAct：让机器人策略随其所处世界而变化"></h1>

<p align="center">
  <a href="https://ziyanliux.github.io/">Ziyan Liu</a><sup>*</sup> · <a href="https://jitingliu.com/">Jiting Liu</a><sup>*</sup> · Weihong Zhong<sup>*</sup> · Wenda Yu · Yeqiu Chen<br>
  Jason Li · Xiaojiang Peng · Shuyuan Hu · <a href="https://kaiwang960112.github.io/">Kai Wang</a><sup>†</sup> · <a href="https://www.bozhao.me/">Bo Zhao</a><sup>†</sup>
</p>

<p align="center"><sub>* 共同贡献。† 通讯作者。</sub></p>

<p align="center">
  <a href="https://mint-sjtu.github.io/MorphAct.io/"><img src="https://img.shields.io/badge/Project-Page-4C7A67?style=flat-square" height="22" alt="项目主页"></a>
  <img src="https://img.shields.io/badge/Paper-Coming_soon-8C8C8C?style=flat-square&amp;logo=arxiv&amp;logoColor=white" height="22" alt="论文即将发布">
  <a href="https://huggingface.co/MINT-SJTU/MorphAct-Metaworld"><img src="https://img.shields.io/badge/Models-Hugging_Face-C99747?style=flat-square&amp;logo=huggingface&amp;logoColor=white" height="22" alt="Hugging Face 模型"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-68758A?style=flat-square" height="22" alt="自研代码采用 MIT 许可证"></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

将机器人学习扩展到多样的环境与交互，意味着同一个共享策略需要应对不同的感知与控制需求。这些需求既因任务而异，也会在一次交互中不断变化，而常规策略在不同上下文中始终使用同一套有效权重。我们的操作级诊断揭示了相互竞争的参数更新方向：降低一个任务批次损失的更新，可能增加其他任务的损失。这促使我们共享适配机制，同时让它产生的参数更新随上下文变化。

为此，我们提出 **MorphAct**，一个使机器人策略能够随交互上下文变化调整有效权重的上下文参数化框架。观察与指令指导视觉语言侧的参数生成，随后得到的适配后表征与机器人状态共同指导动作侧的参数生成。两个生成器通过策略原生的动作目标联合训练，无需操作标签或测试时优化。

![MorphAct 让感知与行动随当前交互协同适配。](assets/morphact-method.png)

## 📊 实验结果（%）

### LIBERO

| 骨干 | 方法 | Spatial | Object | Goal | Long | Mean | 方差 (pp²) |
|---|---|---:|---:|---:|---:|---:|---:|
| X-VLA | Baseline | 95.40 | 96.60 | 96.00 | 84.20 | 93.05 | — |
| X-VLA | MorphAct | 98.67 | 99.87 | 97.07 | 93.20 | **97.20** | 0.0300 |
| π0 | Baseline | 96.80 | 98.80 | 95.80 | 85.20 | 94.15 | — |
| π0 | MorphAct | 98.00 | 99.60 | 96.60 | 92.40 | **96.65** | 0.0517 |
| π0.5 | Baseline | 98.80 | 98.20 | 98.00 | 92.40 | 96.85 | — |
| π0.5 | MorphAct | 98.33 | 99.67 | 98.87 | 97.53 | **98.60** | 0.0525 |

### Meta-World

| 骨干 | 方法 | Easy | Medium | Hard | Very Hard | Avg. | 方差 (pp²) |
|---|---|---:|---:|---:|---:|---:|---:|
| X-VLA | Baseline | 72.50 | 46.36 | 48.30 | 55.00 | 55.54 | — |
| X-VLA | MorphAct | 83.99 | 58.18 | 55.28 | 55.00 | **63.11** | 1.8400 |
| π0 | Baseline | 77.90 | 51.80 | 53.30 | 20.00 | 50.75 | — |
| π0 | MorphAct | 82.44 | 71.36 | 56.39 | 58.00 | **67.05** | 0.6342 |
| π0.5 | Baseline | 70.50 | 45.00 | 38.30 | 41.00 | 48.70 | — |
| π0.5 | MorphAct | 85.77 | 59.85 | 66.39 | 45.67 | **64.42** | 0.3970 |

## 🤗 预训练模型

三种骨干的 Meta-World 预训练适配器可从 [MINT-SJTU/MorphAct-Metaworld](https://huggingface.co/MINT-SJTU/MorphAct-Metaworld) 下载。LIBERO 预训练权重 Coming soon。

将下载的适配器放入 `ckpt/adapters/`，目录名称如下：

| 骨干 | Meta-World | LIBERO |
|---|---|---|
| π0 | `pi0-metaworld` | Coming soon |
| π0.5 | `pi05-metaworld` | Coming soon |
| X-VLA | `xvla-metaworld` | Coming soon |

基座模型路径与推理命令：[π0 / π0.5](openpi/README.zh-CN.md) · [X-VLA](xvla/README.zh-CN.md)。

## 🚀 快速开始

使用带 NVIDIA GPU 的 Linux 环境，并安装 `uv`。以下命令均在仓库根目录执行，并准备好 π0.5 基座模型和 `pi05-metaworld` 检查点：

```bash
bash tools/install_openpi.sh
source artifacts/envs/openpi/bin/activate
bash tools/serve.sh pi05 metaworld
```

在 Meta-World 中评测时，先完成[仿真环境安装](docs/evaluation.zh-CN.md#仿真环境)，再打开第二个终端运行：

```bash
source artifacts/envs/metaworld-eval/bin/activate
bash tools/evaluate.sh pi05 metaworld
```

使用 π0 时将模型参数换为 `pi0`。X-VLA 的安装和推理见 [X-VLA 指南](xvla/README.zh-CN.md)。评测产物和得分定义见[评测指南](docs/evaluation.zh-CN.md#输出)。

## 🛠️ 训练

如需在自己的数据上训练 MorphAct，请参考 [π0 / π0.5 训练指南](openpi/README.zh-CN.md#训练)或 [X-VLA 训练指南](xvla/README.zh-CN.md#训练)。

## 致谢

骨干集成使用了 OpenPI 和 X-VLA。

## 联系我们

- **Ziyan Liu：**[liuziyan@mail.ustc.edu.cn](mailto:liuziyan@mail.ustc.edu.cn)
- **Jiting Liu：**[liujiting616@gmail.com](mailto:liujiting616@gmail.com)

## 许可证

MorphAct 自研代码采用 [MIT 许可证](LICENSE)。第三方组件沿用各自的许可证，详见 [NOTICE](NOTICE)。
