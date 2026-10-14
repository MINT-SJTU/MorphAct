<h1 align="center"><img src="assets/morphact-title-en.svg" width="960" alt="MorphAct: Robot Policies Morphing with the Worlds They Act In"></h1>

<p align="center">
  <a href="https://ziyanliux.github.io/">Ziyan Liu</a><sup>*</sup> · <a href="https://jitingliu.com/">Jiting Liu</a><sup>*</sup> · Weihong Zhong<sup>*</sup> · Wenda Yu · Yeqiu Chen<br>
  Jason Li · Xiaojiang Peng · Shuyuan Hu · <a href="https://kaiwang960112.github.io/">Kai Wang</a><sup>†</sup> · <a href="https://www.bozhao.me/">Bo Zhao</a><sup>†</sup>
</p>

<p align="center"><sub>* Equal contribution. † Corresponding authors.</sub></p>

<p align="center">
  <a href="https://mint-sjtu.github.io/MorphAct.io/"><img src="https://img.shields.io/badge/Project-Page-green?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" height="22" alt="Project Page"></a>
  <img src="https://img.shields.io/badge/Paper-Coming_soon-red?style=flat-square&amp;logo=arxiv&amp;logoColor=white" height="22" alt="Paper: coming soon">
  <a href="https://huggingface.co/MINT-SJTU/MorphAct-Metaworld"><img src="https://img.shields.io/badge/Models-Hugging_Face-blue?style=flat-square&amp;logo=huggingface&amp;logoColor=FFD21E" height="22" alt="Hugging Face models"></a>
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

Scaling robot learning brings diverse perceptual and control demands into one shared policy, across tasks and within a single interaction. Conventional policies keep their effective weights fixed across contexts. **MorphAct** learns a shared adaptation rule that generates **context-dependent parameter updates**, coupling vision–language and action adaptation through the representations used for control.

![MorphAct adapts perception and action together as the interaction changes.](assets/morphact-method.png)

- **Operation-level evidence.** Different operations can favor competing parameter updates; our diagnostics show that reducing one task’s batch loss can increase loss on others.
- **Representation-coupled generation.** Observations and instructions guide vision–language updates; the adapted representations and robot state then guide action-side parameter generation.
- **Joint learning.** Both generators learn through the native action objective, without operation labels or test-time optimization. Their weights are shared across contexts.

*See results and robot demos on the [project page](https://mint-sjtu.github.io/MorphAct.io/).*

## 📊 Results (%)

### LIBERO

| Backbone | Method | Spatial | Object | Goal | Long | Mean | Var. (pp²) |
|---|---|---:|---:|---:|---:|---:|---:|
| X-VLA | Baseline | 95.40 | 96.60 | 96.00 | 84.20 | 93.05 | — |
| X-VLA | MorphAct | 98.67 | 99.87 | 97.07 | 93.20 | **97.20** | 0.0300 |
| π0 | Baseline | 96.80 | 98.80 | 95.80 | 85.20 | 94.15 | — |
| π0 | MorphAct | 98.00 | 99.60 | 96.60 | 92.40 | **96.65** | 0.0517 |
| π0.5 | Baseline | 98.80 | 98.20 | 98.00 | 92.40 | 96.85 | — |
| π0.5 | MorphAct | 98.33 | 99.67 | 98.87 | 97.53 | **98.60** | 0.0525 |

### Meta-World

| Backbone | Method | Easy | Medium | Hard | Very Hard | Avg. | Var. (pp²) |
|---|---|---:|---:|---:|---:|---:|---:|
| X-VLA | Baseline | 72.50 | 46.36 | 48.30 | 55.00 | 55.54 | — |
| X-VLA | MorphAct | 83.99 | 58.18 | 55.28 | 55.00 | **63.11** | 1.8400 |
| π0 | Baseline | 77.90 | 51.80 | 53.30 | 20.00 | 50.75 | — |
| π0 | MorphAct | 82.44 | 71.36 | 56.39 | 58.00 | **67.05** | 0.6342 |
| π0.5 | Baseline | 70.50 | 45.00 | 38.30 | 41.00 | 48.70 | — |
| π0.5 | MorphAct | 85.77 | 59.85 | 66.39 | 45.67 | **64.42** | 0.3970 |

## 🤗 Pretrained models

Download the Meta-World pretrained adapters for all three backbones from [MINT-SJTU/MorphAct-Metaworld](https://huggingface.co/MINT-SJTU/MorphAct-Metaworld). LIBERO pretrained adapters are coming soon.

Place each downloaded adapter under `ckpt/adapters/` using this directory name:

| Backbone | Meta-World | LIBERO |
|---|---|---|
| π0 | `pi0-metaworld` | Coming soon |
| π0.5 | `pi05-metaworld` | Coming soon |
| X-VLA | `xvla-metaworld` | Coming soon |

Base-model paths and inference commands: [π0 / π0.5](openpi/README.md) · [X-VLA](xvla/README.md).

## 🚀 Quick start

On Linux with an NVIDIA GPU and `uv`, run these commands from the repository root with the π0.5 base model and `pi05-metaworld` checkpoint:

```bash
bash tools/install_openpi.sh
source artifacts/envs/openpi/bin/activate
bash tools/serve.sh pi05 metaworld
```

To evaluate it in Meta-World, follow the [simulator setup](docs/evaluation.md#simulator-setup), then run in a second terminal:

```bash
source artifacts/envs/metaworld-eval/bin/activate
bash tools/evaluate.sh pi05 metaworld
```

Use `pi0` for π0. X-VLA installation and inference are described in the [X-VLA guide](xvla/README.md). Evaluation outputs and score definitions are documented in the [evaluation guide](docs/evaluation.md#outputs).

## 🛠️ Training

To train MorphAct on your own data, see [π0 / π0.5 training](openpi/README.md#training) or [X-VLA training](xvla/README.md#training).

## Acknowledgements

The backbone integrations use OpenPI and X-VLA.

## Contact

- **Ziyan Liu:** [liuziyan@mail.ustc.edu.cn](mailto:liuziyan@mail.ustc.edu.cn)
- **Jiting Liu:** [liujiting616@gmail.com](mailto:liujiting616@gmail.com)

## License

Original MorphAct code is licensed under the [MIT License](LICENSE). Third-party components retain their respective licenses; see [NOTICE](NOTICE).
