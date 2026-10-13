<h1 align="center">MorphAct</h1>

<p align="center">
  <strong>让机器人策略随其所处世界而变化</strong>
</p>

<p align="center">
  Ziyan Liu<sup>*</sup> · <a href="https://jitingliu.com/">Jiting Liu</a><sup>*</sup> · Weihong Zhong<sup>*</sup> · Wenda Yu · Yeqiu Chen<br>
  Jason Li · Xiaojiang Peng · Shuyuan Hu · <a href="https://kaiwang960112.github.io/">Kai Wang</a><sup>†</sup> · <a href="https://www.bozhao.me/">Bo Zhao</a><sup>†</sup>
</p>

<p align="center"><sub>* 共同贡献。† 通讯作者。</sub></p>

<table align="center">
  <tr>
    <th width="200" align="center">🌐 项目主页</th>
    <th width="200" align="center">📄 arXiv</th>
  </tr>
  <tr>
    <td align="center"><a href="https://mint-sjtu.github.io/MorphAct.io/">Website</a></td>
    <td align="center"><a href="#" title="论文即将发布">Coming soon</a></td>
  </tr>
</table>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

**MorphAct 让机器人策略根据当前交互调整有效权重。** 感知与控制需求会随环境、任务以及同一任务的不同阶段而变化。传统视觉–语言–动作（VLA）策略通过随情境变化的激活和动作作出响应，执行时的有效权重则保持固定。MorphAct 学习共享的参数生成规则，让生成的更新随当前情境而变化。

对 RoboCasa365 的 65 个原子任务进行的局部诊断显示，视觉–语言与动作参数都存在相互竞争的更新方向，同一模块内不同参数组的相反贡献也会抵消。这些发现促使我们让参数适配取决于当前情境。

![MorphAct：通过适配后的表示耦合感知与控制的参数生成。](assets/morphact-method.png)

## 方法

1. **适配感知。** 视觉–语言生成器读取观测与指令的嵌入，为 VLM 生成参数更新，由适配后的 VLM 产生表示。
2. **适配控制。** 动作生成器读取这些适配后的表示和机器人状态，为动作模块生成参数更新。
3. **联合训练。** 两个生成器通过骨干模型原有的动作目标联合学习，无需操作类别标签或专家适配器权重监督。

低秩因子让生成的更新能够高效计算。每次获得用于重新规划的观测时，策略都会刷新参数更新，并在动作块预测过程中复用。

## 实验结果（%）

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

## 预训练模型

三种骨干的 Meta-World 预训练适配器可从 [MINT-SJTU/MorphAct-Metaworld](https://huggingface.co/MINT-SJTU/MorphAct-Metaworld) 下载。LIBERO 预训练权重 Coming soon。

将下载的适配器放入 `ckpt/adapters/`，目录名称如下：

| 骨干 | Meta-World | LIBERO |
|---|---|---|
| π0 | `pi0-metaworld` | Coming soon |
| π0.5 | `pi05-metaworld` | Coming soon |
| X-VLA | `xvla-metaworld` | Coming soon |

基座模型路径与推理命令：[π0 / π0.5](openpi/README.zh-CN.md) · [X-VLA](xvla/README.zh-CN.md)。

## 快速开始

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

## 训练

如需在自己的数据上训练 MorphAct，请参考 [π0 / π0.5 训练指南](openpi/README.zh-CN.md#训练)或 [X-VLA 训练指南](xvla/README.zh-CN.md#训练)。

## 致谢

骨干集成使用了 OpenPI 和 X-VLA。

## 联系我们

- **Ziyan Liu：**[liuziyan@mail.ustc.edu.cn](mailto:liuziyan@mail.ustc.edu.cn)
- **Jiting Liu：**[liujiting616@gmail.com](mailto:liujiting616@gmail.com)

## 许可证

MorphAct 自研代码采用 [MIT 许可证](LICENSE)。第三方组件沿用各自的许可证，详见 [NOTICE](NOTICE)。
