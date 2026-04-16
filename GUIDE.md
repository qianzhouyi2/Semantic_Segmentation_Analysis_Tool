# Semantic Segmentation Analysis Guide

这份指南面向当前仓库已经落地的功能，目标不是只解释几个前端热图，而是把“数据检查 -> clean 评测 -> 对抗评测 -> 稀疏防御 -> 批量协议 -> 前端解释”串成一条可执行的分析流程。

## 1. 仓库当前定位

这个项目现在适合做：

- 数据集体检，检查图像 / mask 配对、尺寸、空 mask、非法标签和类别分布。
- 离线预测结果评估，直接比较预测 mask 和 GT mask。
- Pascal VOC checkpoint 的 clean 验证集评测。
- 单模型白盒攻击评测，输出指标、扰动统计和逐类结果。
- source -> target 的黑盒迁移攻击评测。
- 稀疏防御 sidecar 校准、阈值搜索和批量对比。
- 基于 Slurm 的 18 模型 attack suite、strict transfer protocol 和 full all-attacks 套件。
- Streamlit 交互分析，包括三联图、对抗特征、CAM 和响应区域。

这个项目当前不适合直接做：

- 只靠前端热图证明“某防御已经稳健”。
- 只跑一次单攻击就声称“攻击已经打透”。
- 不做批量统计就从个别样本概括整个模型。
- 替代 multi-restart、EOT、BPDA、AutoAttack 风格的强验证基线。

建议把它理解成一个“研究分析工具箱”，而不是单纯的可视化 demo。

## 2. 功能入口速查

| 场景 | 推荐入口 | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| 数据集扫描 | `streamlit run app.py` 的 `Dataset Scan`，或 `python scripts/check_dataset.py ...` | 数据集配置、标签配置、图像目录、mask 目录 | `summary.json`、`class_distribution.csv`、`report.md` |
| 三联图预览 | `Triplet Preview` | 原图、GT、预测，或 VOC val 样本 | 叠加可视化 |
| 离线预测评估 | `python scripts/evaluate_predictions.py ...` | `gt_dir`、`pred_dir`、标签配置 | `summary.json`、`per_class_metrics.csv`、`report.md` |
| VOC clean 评测 | `python scripts/evaluate_voc_clean.py ...` | family、checkpoint、数据集根目录 | `summary.json`、`per_class_metrics.csv`、`report.md`、`evaluate.log` |
| 单模型对抗评测 | `python scripts/run_attack.py ...` | attack YAML、family、checkpoint | 攻击后指标、扰动统计、逐类指标、可选特征可视化 |
| clean / adv 对比 | `python scripts/evaluate_robustness.py ...` | 两份 `summary.json` | `summary.json`、`report.md` |
| 两次运行对比 | `python scripts/compare_runs.py ...` / `python scripts/compare_robustness.py ...` | baseline / candidate summary | 差异报告 |
| 稀疏防御校准 | `python scripts/prepare_sparse_defense.py ...` | checkpoint、防御 YAML、VOC split | sidecar `.pt`、校准 summary |
| 稀疏阈值搜索 | `python scripts/search_sparse_thresholds.py ...` | family、checkpoint、stats、threshold 列表 | 每阈值 clean / PGD 结果、`search_summary.json` |
| 稀疏搜索汇总 | `python scripts/summarize_sparse_threshold_search.py ...` | search root | `selected_thresholds.json`、`selected_thresholds.md` |
| 18 模型 attack suite | `materialize_voc_attack_suite_manifest.py` -> `launch_voc_attack_suite.py` -> `summarize_voc_attack_suite.py` | 阈值搜索结果、Slurm | suite summary |
| strict transfer protocol | `materialize_voc_transfer_protocol_manifest.py` -> `launch_voc_transfer_protocol.py` -> `summarize_voc_transfer_protocol.py` | 阈值搜索结果、Slurm | transfer protocol summary |
| full all-attacks suite | `launch_voc_all_attacks.py` -> `summarize_voc_all_attacks.py` | manifest、攻击配置目录、Slurm | 所有攻击的总表 |

## 3. 当前已落地的模型、攻击和防御

### 3.1 模型 family

当前代码里可直接构建的 family 有三个：

- `upernet_convnext`
- `upernet_resnet50`
- `segmenter_vit_s`

它们都可以走：

- clean VOC 评测
- 对抗攻击评测
- Streamlit 单样本攻击预览
- 稀疏防御加载

### 3.2 已提供的攻击配置

`configs/attacks/` 里已经有以下攻击 YAML：

- `fgsm`
- `fgsm_4_255`
- `pgd`
- `bim`
- `cospgd`
- `segpgd`
- `sea`
- `dag`
- `fspgd`
- `rppgd`
- `transegpgd`
- `mi_fgsm`
- `ni_fgsm`
- `di2_fgsm`
- `ti_fgsm`
- `ni_di_ti`
- `tass`

可以把它们粗分成三类：

- 白盒基础 / 加强攻击：`fgsm`、`pgd`、`bim`、`cospgd`、`segpgd`、`sea`
- 分割结构专用攻击：`dag`、`fspgd`、`rppgd`、`transegpgd`、`tass`
- 迁移增强攻击：`mi_fgsm`、`ni_fgsm`、`di2_fgsm`、`ti_fgsm`、`ni_di_ti`

### 3.3 稀疏防御能力

当前 runtime 可以加载这些 sparse 变体：

- `meansparse`
- `extrasparse`
- `cc_extra_sparse`
- `dir_extra_sparse`
- `margin_extra_sparse`

当前自动化阈值搜索脚本直接支持：

- `meansparse`
- `extrasparse`

配置示例位于 `configs/defenses/`，统计 sidecar 默认放在 `models/defenses/`。

## 4. 推荐使用方式

### 4.1 环境准备

```bash
conda activate segtool
python -m pip install -r requirements.txt
```

所有命令都应从仓库根目录运行。

Pascal VOC 评测默认期望目录为：

```text
datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/val.txt
```

### 4.2 先做数据和结果体检

#### 数据集检查

```bash
python scripts/check_dataset.py \
  --dataset-config configs/datasets/example.yaml \
  --label-config configs/labels/example.yaml
```

这个步骤适合先回答：

- 图像和 mask 是否一一匹配。
- 是否有 orphan mask、尺寸不一致、空 mask。
- 是否存在标签越界或类别分布极端失衡。

#### 离线预测结果评估

```bash
python scripts/evaluate_predictions.py \
  --gt-dir datasets/masks \
  --pred-dir results/predictions \
  --label-config configs/labels/example.yaml \
  --output-dir results/reports/evaluation/manual_eval
```

这个脚本不依赖模型，只关心预测 mask 和 GT mask 本身，适合先验证外部生成的结果文件是否正常。

### 4.3 做 VOC clean 基线

```bash
python scripts/evaluate_voc_clean.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --dataset-root datasets \
  --output-dir results/reports/voc_clean_eval/UperNet_ConvNext_T_VOC_clean
```

如果你要评估带防御的模型，可以直接加：

```bash
--defense-config configs/defenses/meansparse_example.yaml
```

当前 clean 评测会输出：

- `reference_percent`: `mIoU`、`mAcc`、`aAcc`
- 扩展指标：`pixel_accuracy`、`mean_dice`、`mean_precision`、`mean_recall`、`mean_f1`
- `per_class_metrics.csv`

### 4.4 做单次对抗评测

```bash
python scripts/run_attack.py \
  --attack-config configs/attacks/fgsm.yaml \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --dataset-root datasets
```

常用增强参数：

- `--defense-config ...`：评估带稀疏防御的模型。
- `--epsilon-scale 1.25`：在 YAML 基础上整体放大攻击半径和显式步长。
- `--feature-vis-samples 8`：导出前 8 个样本的逐层特征可视化。
- `--feature-vis-layers 6`：限制每个样本导出的层数。
- `--no-strict`：兼容 checkpoint key 不完全匹配的情况。

默认输出目录：

```text
results/reports/voc_adv_eval/<checkpoint_stem>_<attack_name>/
```

对抗评测结果除了常规指标外，还会记录：

- `mean_linf`
- `max_linf`
- `mean_l2`
- 可选的 `feature_visualizations`

### 4.5 clean / adv 结果对比

```bash
python scripts/evaluate_robustness.py \
  --clean-summary results/reports/voc_clean_eval/UperNet_ConvNext_T_VOC_clean/summary.json \
  --adv-summary results/reports/voc_adv_eval/UperNet_ConvNext_T_VOC_clean_fgsm/summary.json \
  --output-dir results/reports/robustness/UperNet_ConvNext_T_fgsm
```

它更适合做“批量指标层面”的结论，包括：

- clean / adv `pixel_accuracy`
- clean / adv `mean_iou`
- clean / adv `mean_dice`
- 对应的 drop

如果你要比较两次 clean 运行或两次 robustness 运行，可以再用：

```bash
python scripts/compare_runs.py --baseline ... --candidate ... --output-dir ...
python scripts/compare_robustness.py --baseline ... --candidate ... --output-dir ...
```

### 4.6 做迁移攻击评测

如果你要显式区分 source 和 target，可以直接用：

```bash
python scripts/run_transfer_attack.py \
  --attack-config configs/attacks/mi_fgsm.yaml \
  --source-family upernet_convnext \
  --source-checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --target-family segmenter_vit_s \
  --target-checkpoint models/Segmenter_ViT_S_VOC_clean.pth \
  --dataset-root datasets \
  --output-dir results/reports/voc_transfer_eval/convnext_to_vit_mi_fgsm
```

这个脚本会输出：

- target 侧 `mIoU`、`mAcc`、`aAcc`
- source / target checkpoint 信息
- 扰动统计
- `per_class_metrics.csv`

### 4.7 做稀疏防御校准与阈值搜索

推荐流程是三步。

#### 第一步：校准统计 sidecar

```bash
python scripts/prepare_sparse_defense.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --defense-config configs/defenses/meansparse_example.yaml \
  --dataset-root datasets \
  --dataset-split val
```

这一步会生成：

- 稀疏统计 sidecar `.pt`
- 校准 summary `.summary.json`

#### 第二步：把防御插回 clean / attack 评测

```bash
python scripts/evaluate_voc_clean.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --defense-config configs/defenses/meansparse_example.yaml \
  --dataset-root datasets
```

```bash
python scripts/run_attack.py \
  --attack-config configs/attacks/pgd.yaml \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --defense-config configs/defenses/meansparse_example.yaml \
  --dataset-root datasets
```

#### 第三步：自动搜索阈值

```bash
python scripts/search_sparse_thresholds.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --variant meansparse \
  --stats-path models/defenses/meansparse_example_stats.pt \
  --attack-config configs/attacks/pgd.yaml \
  --dataset-root datasets \
  --dataset-split train \
  --output-dir results/reports/sparse_search/UperNet_ConvNext_T_VOC_clean/meansparse
```

这个脚本会：

- 对一组阈值分别跑 clean 和 PGD。
- 生成每个阈值自己的 `results.json`。
- 生成 `search_summary.json` 和 `search_summary.md`。
- 在 clean / adv 双目标上给出 Pareto frontier 和 best threshold。

如果要汇总多个 case 的搜索结果，可以再跑：

```bash
python scripts/summarize_sparse_threshold_search.py \
  --search-root results/reports/sparse_search
```

### 4.8 用 Slurm 跑批量协议

GPU 评测不要直接在登录节点跑，优先走 Slurm。

#### clean VOC 批量评测

```bash
sbatch scripts/submit_voc_clean_eval.sbatch
```

也可以覆盖默认变量：

```bash
RESULTS_ROOT=results/reports/voc_clean_eval_debug \
DATASET_ROOT=datasets \
sbatch scripts/submit_voc_clean_eval.sbatch
```

#### 18 模型 attack suite

这个协议默认把 6 个 base checkpoint 扩成 18 个模型：

- 6 个 baseline
- 6 个 `meansparse`
- 6 个 `extrasparse`

然后统一跑：

- white-box: `pgd`、`segpgd`
- black-box transfer: `mi_fgsm`、`ni_di_ti`

典型流程：

```bash
python scripts/materialize_voc_attack_suite_manifest.py \
  --search-root results/reports/sparse_search \
  --output-dir results/reports/voc_attack_suite_manifest
```

```bash
python scripts/launch_voc_attack_suite.py \
  --manifest results/reports/voc_attack_suite_manifest/attack_suite_manifest.json \
  --suite-root results/reports/voc_attack_suite \
  --dataset-root datasets
```

```bash
python scripts/summarize_voc_attack_suite.py \
  --manifest results/reports/voc_attack_suite_manifest/attack_suite_manifest.json \
  --suite-root results/reports/voc_attack_suite
```

#### strict transfer protocol

这个协议会把 source 端生成的同一组对抗输入共享给一组 target，从而保证 transfer 对比更严格。

```bash
python scripts/materialize_voc_transfer_protocol_manifest.py \
  --search-root results/reports/sparse_search \
  --output-dir results/reports/voc_transfer_protocol_manifest
```

```bash
python scripts/launch_voc_transfer_protocol.py \
  --manifest results/reports/voc_transfer_protocol_manifest/transfer_protocol_manifest.json \
  --suite-root results/reports/voc_transfer_protocol \
  --dataset-root datasets
```

```bash
python scripts/summarize_voc_transfer_protocol.py \
  --manifest results/reports/voc_transfer_protocol_manifest/transfer_protocol_manifest.json \
  --suite-root results/reports/voc_transfer_protocol
```

#### full all-attacks 套件

如果你想让 18 个模型把 `configs/attacks/*.yaml` 里的所有攻击都跑一遍，可以在 manifest 基础上再启动：

```bash
python scripts/launch_voc_all_attacks.py \
  --manifest results/reports/voc_attack_suite_manifest/attack_suite_manifest.json \
  --suite-root results/reports/voc_all_attacks \
  --dataset-root datasets
```

```bash
python scripts/summarize_voc_all_attacks.py \
  --manifest results/reports/voc_attack_suite_manifest/attack_suite_manifest.json \
  --suite-root results/reports/voc_all_attacks
```

### 4.9 用 Streamlit 做交互分析

启动方式：

```bash
streamlit run app.py
```

当前页面包含五个 tab。

#### `Dataset Scan`

适合快速查看：

- 图像 / mask 配对数量
- missing mask、orphan mask、size mismatch
- 空 mask 和非法标签
- 类别分布

#### `Triplet Preview`

当前有三种模式：

- `Dataset browser`：按数据集配置浏览样本。
- `Pascal VOC demo`：直接浏览 VOC val 样本，并可叠加外部预测目录。
- `Manual paths`：手工输入原图、GT、预测路径。

这是最值得优先看的模块，因为它能先回答“到底错在哪里”。

#### `Adversarial Feature Preview`

当前可以在前端直接选择：

- VOC val 样本
- model family
- 自动发现的 checkpoint
- 攻击类型和具体 YAML
- 扰动半径 `radius/255`
- device 和 strict load

输出包括：

- clean / adversarial 输入
- perturbation
- sample delta heatmap
- GT / clean prediction / adversarial prediction 叠加图
- 逐层特征的 clean / adv / abs diff 热图

#### `CAM Preview`

在单样本攻击结果基础上，进一步对某个类别画 CAM。默认会选择最深的可用 4D 特征层。

适合回答：

- 类相关注意区域是否从主体漂移到背景。
- clean 和 adversarial 的热点是否还覆盖目标物体。

#### `Response Region Analysis`

这个模块会基于目标类别在输入空间上的响应，展示：

- clean / adversarial response heatmap
- clean / adversarial high-response region
- stable overlap region
- `Clean Mean`、`Adv Mean`、`Diff Mean`
- `Clean Area`、`Adv Area`
- `Overlap IoU`

它更适合回答“热点位置有没有迁移”，不适合把面积当成主结论。

## 5. 输出目录约定

常见输出位置如下：

- `results/reports/dataset_check/`
- `results/reports/evaluation/`
- `results/reports/voc_clean_eval/`
- `results/reports/voc_adv_eval/`
- `results/reports/robustness/`
- `results/reports/comparison/`
- `results/reports/robustness_comparison/`
- `results/reports/sparse_search/`
- `results/reports/voc_attack_suite/`
- `results/reports/voc_transfer_protocol/`
- `results/reports/voc_all_attacks/`

多数脚本至少会生成以下几类工件中的一部分：

- `summary.json`
- `per_class_metrics.csv`
- `report.md`
- `evaluate.log` 或 `search.log`

## 6. 前端模块分别适合回答什么

### `Dataset Scan`

适合回答：

- 数据本身是否干净。
- 类别分布是否明显异常。

不适合回答：

- 模型鲁棒性是否足够强。
- 某个防御是否有效。

### `Triplet Preview`

适合回答：

- clean / GT / prediction 的差异是什么。
- 错误是边界侵蚀、局部错分、整物体消失还是类别替换。

这是所有单样本分析的起点。

### `Adversarial Feature Preview`

适合回答：

- 从哪一层开始明显失稳。
- 更早脆弱的是 backbone 还是更晚的 decoder / readout。
- 差异更像局部块状还是全局扩散。

局限：

- 这里显示的是高维特征压缩后的能量图，不是严格归因图。
- 它适合定位“哪一层出问题”，不适合单独回答“为什么最终类别错了”。

### `CAM Preview`

适合回答：

- 语义关注区域是否还在目标主体上。
- 热点是否从主体漂到背景、边界或干扰纹理。

局限：

- 更适合看空间位置变化，不适合比较“谁更亮”。
- 如果你选的类别在当前样本里不相关，图仍然能画出来，但解释价值会明显下降。

### `Response Region Analysis`

适合回答：

- 输入空间里的敏感区域是否发生迁移。
- clean / adv 高响应区域是否仍然重叠。
- 稳定重叠区域是否还落在主体上。

局限：

- 面积高度依赖 percentile 阈值。
- `Clean Area` / `Adv Area` 不应直接当成主结论。
- 如果 `target_pixels = 0`，图更适合作调试，不适合作论文证据。

## 7. 推荐分析顺序

如果你是为了做严肃的对抗分析，推荐按下面的顺序走。

1. 先用 `check_dataset.py` 或 `Dataset Scan` 确认数据没有结构性问题。
2. 先做 clean 基线，拿到整体 `mIoU`、`mAcc`、`aAcc` 和逐类结果。
3. 再跑攻击评测，先看批量指标 drop，而不是先看热图。
4. 如果是防御方案，先做 sidecar 校准，再做阈值搜索，而不是手调阈值。
5. 如果 claim 涉及迁移鲁棒性，再补 `run_transfer_attack.py` 或 strict transfer protocol。
6. 从批量结果里挑代表样本，再回到前端。
7. 在前端里先看 `Triplet Preview`，再看 `Perturbation / Sample Delta`，再看 `Feature Preview`。
8. 最后只对相关类别看 `CAM` 和 `Response Region`。
9. 用“批量数值 + 若干代表样本 + 中间解释图”一起写结论。

## 8. 哪些结论现在不能直接说

当前工具不足以单独支持下面这些强结论：

- “这个防御已经被充分验证为稳健。”
- “这个方法不存在 gradient masking。”
- “因为 CAM 很稳定，所以模型就稳健。”
- “因为响应区域面积没怎么变，所以模型鲁棒。”
- “因为 feature diff 不大，所以防御成功。”

理由很简单：

- 单次攻击不等于最强攻击验证。
- 当前没有统一接入 multi-restart、EOT、BPDA、AutoAttack 风格基线。
- 前端主要是单样本解释工具，不是稳健性证明工具。

## 9. 哪些结论相对稳妥

当前工具比较适合支持这些较保守的结论：

- 哪类样本在攻击下更容易出现哪种错误。
- 模型从哪一层开始明显失稳。
- 语义热点是否从主体迁移到背景。
- 输入高响应区域是否从主体偏移到边界或局部纹理。
- 某个防御是在减小整体崩坏，还是只是推迟失稳层。
- 不同 sparse 阈值在 clean / PGD 之间的 trade-off 是什么。
- 某个 source family 生成的扰动对哪些 target family 更可迁移。

## 10. 当前已知局限

- VOC 评测路径当前围绕 `val` split 构建，默认预处理是 `resize_short=473` 和中心裁剪 `473x473`。
- 前端攻击预览一次只看一个样本，不会自动做最坏情况搜索。
- 大多数热图都有各自独立的归一化，因此更适合看结构变化，不适合看绝对强弱。
- `CAM` 和 `Response Region` 对不相关类别也能出图，所以类别选择必须谨慎。
- full all-attacks 套件的覆盖范围取决于 `configs/attacks/*.yaml`，不是固定写死在前端里。

## 11. 一句话建议

把这个仓库的最佳使用方式理解成：

先用 CLI 和 Slurm 拿到批量 clean / adversarial / transfer / sparse-search 结果，再用 Streamlit 回看代表样本，最后只在数值证据已经站住时把 CAM、Feature Preview 和 Response Region 当作解释工具。
