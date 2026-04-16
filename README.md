# Semantic Segmentation Analysis Tool

一个面向语义分割实验的本地分析与鲁棒性评测工具箱。仓库围绕 Pascal VOC 和通用 mask 数据，覆盖数据集体检、离线预测评估、VOC clean 验证集评测、白盒攻击、黑盒迁移攻击、稀疏防御校准、Slurm 批量协议，以及 Streamlit 交互可视化。

当前仓库已经不是最初的 MVP 骨架，更接近一个可直接复用的研究分析工具链：可以从数据检查一路跑到 clean / adv / transfer / sparse 汇总，并统一导出 `json`、`csv`、`md` 结果。

## 功能概览

- 数据集扫描与体检：检查图像和 mask 配对、缺失文件、尺寸不一致、空 mask、非法标签，并统计类别分布。
- Streamlit 可视化：提供 `Dataset Scan`、`Triplet Preview`、`Adversarial Feature Preview`、`CAM Preview`、`Response Region Analysis` 五个模块。
- 分割结果评估：对预测 mask 与 GT mask 计算 `pixel_accuracy`、`mIoU`、`Dice`、`Precision`、`Recall`、`F1`。
- Pascal VOC clean 评测：加载本地 checkpoint，在 VOC2012 val 上输出标准分割指标和逐类结果。
- 对抗攻击评测：支持多种分割攻击配置，并导出攻击后指标、逐类结果和扰动统计。
- 黑盒迁移攻击：显式区分 `source -> target`，评估迁移型扰动在目标模型上的退化效果。
- 稀疏防御分析：支持 sidecar 校准、阈值搜索、clean / PGD 联合筛选，以及防御配置加载。
- 批量实验自动化：提供 attack suite、transfer protocol、all-attacks 套件的 manifest、Slurm 提交和汇总脚本。
- 结果导出：统一生成 `summary.json`、`per_class_metrics.csv`、`report.md`、`summary_all.csv` 等输出。

## 已支持的模型、攻击与防御

### 模型 family

- `upernet_convnext`
- `upernet_resnet50`
- `segmenter_vit_s`

### 内置攻击

- `fgsm`
- `pgd`
- `bim`
- `cospgd`
- `segpgd`
- `sea`
- `mi-fgsm`
- `ni-fgsm`
- `di2-fgsm`
- `ti-fgsm`
- `ni+di+ti`
- `dag`
- `tass`
- `transegpgd`
- `fspgd`
- `rppgd`

对应 YAML 配置位于 `configs/attacks/`，例如 `fgsm.yaml`、`pgd.yaml`、`segpgd.yaml`、`transegpgd.yaml`、`rppgd.yaml`。

### 稀疏防御变体

- `meansparse`
- `extrasparse`
- `cc_extra_sparse`
- `dir_extra_sparse`
- `margin_extra_sparse`

当前自动化阈值搜索脚本直接支持 `meansparse` 和 `extrasparse`。防御配置位于 `configs/defenses/`，统计 sidecar 默认放在 `models/defenses/`。

## 项目结构

```text
.
├── app.py                               # Streamlit 入口
├── GUIDE.md                             # 当前完整工作流说明
├── README_FRONTEND.md                   # 前端页面与交互说明
├── configs/
│   ├── attacks/                         # 攻击 YAML
│   ├── defenses/                        # 稀疏防御 YAML 与筛选结果
│   ├── datasets/                        # 数据集路径与后缀配置
│   ├── labels/                          # 标签 ID / 名称 / 调色板配置
│   └── reports/                         # 报告配置示例
├── models/                              # 本地 checkpoint 与防御 sidecar
├── scripts/                             # CLI 脚本与 Slurm 提交脚本
├── src/
│   ├── apps/                            # Streamlit 页面逻辑
│   ├── attacks/                         # 攻击实现与运行器
│   ├── datasets/                        # VOC loader、扫描与统计
│   ├── evaluation/                      # clean / adversarial 评测
│   ├── metrics/                         # 分割指标
│   ├── models/                          # 模型构建、注册、稀疏防御
│   ├── reporting/                       # JSON / CSV / Markdown 导出
│   ├── robustness/                      # 对比与鲁棒性分析
│   └── visualization/                   # 三联图、CAM、响应区域
├── tests/                               # scanner / metrics / attacks / sparse 单元测试
└── results/                             # 评测输出目录
```

所有命令都应从仓库根目录运行，这样 `scripts/_bootstrap.py` 才能正确注入项目路径。

## 环境准备

推荐环境：

- Python `3.11`
- Conda 环境名 `segtool`
- PyTorch `2.4.1`

安装方式：

```bash
conda activate segtool
python -m pip install -r requirements.txt
```

如果只在 CPU 上调试，可在评测命令中显式传入 `--device cpu`。

## 数据与权重准备

### 通用扫描 / 预测评估

示例数据集配置：

```yaml
name: example_dataset
image_dir: datasets/images
mask_dir: datasets/masks
image_suffixes: [.jpg, .jpeg, .png]
mask_suffixes: [.png]
```

示例标签配置：

```yaml
ignore_index: 255
background_ids: [0]
classes:
  - id: 0
    name: background
    color: [0, 0, 0]
  - id: 1
    name: foreground
    color: [0, 255, 0]
```

### Pascal VOC clean / adversarial / transfer 评测

VOC loader 期望的数据目录为：

```text
datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/val.txt
```

本地模型权重放在 `models/`。

## 快速开始

### 1. 启动可视化页面

```bash
streamlit run app.py
```

页面当前提供五块功能：

- `Dataset Scan`：扫描图像与 mask，展示总量、空 mask、异常标签和类别分布。
- `Triplet Preview`：渲染原图、GT mask、预测 mask 的三联图。
- `Adversarial Feature Preview`：单样本攻击预览，查看输入扰动、预测变化和逐层特征热图。
- `CAM Preview`：单样本类激活图预览，固定使用最深可用 CAM 层，对比 clean / adversarial 注意区域。
- `Response Region Analysis`：单样本输入响应区域分析，比较 clean / adversarial 的梯度响应热图、高响应区域和重叠区域。

### 2. 检查数据集结构与标签分布

```bash
python scripts/check_dataset.py \
  --dataset-config configs/datasets/example.yaml \
  --label-config configs/labels/example.yaml
```

默认输出到 `results/reports/dataset_check/`，包含：

- `summary.json`
- `class_distribution.csv`
- `report.md`

### 3. 评估已有预测 mask

```bash
python scripts/evaluate_predictions.py \
  --gt-dir datasets/masks \
  --pred-dir results/predictions \
  --label-config configs/labels/example.yaml \
  --output-dir results/reports/evaluation/manual_eval
```

该脚本适合评估已经离线生成好的预测结果，不依赖具体模型实现。

### 4. 评估 VOC clean 验证集上的 checkpoint

```bash
python scripts/evaluate_voc_clean.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --dataset-root datasets \
  --output-dir results/reports/voc_clean_eval/UperNet_ConvNext_T_VOC_clean
```

如果要评估带防御的模型，可额外传入：

```bash
--defense-config configs/defenses/meansparse_example.yaml
```

输出目录包含：

- `summary.json`
- `per_class_metrics.csv`
- `report.md`
- `evaluate.log`

### 5. 运行白盒对抗攻击评测

```bash
python scripts/run_attack.py \
  --attack-config configs/attacks/fgsm.yaml \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --dataset-root datasets
```

如果不显式指定 `--output-dir`，结果默认写入：

```text
results/reports/voc_adv_eval/<checkpoint_stem>_<attack_name>/
```

对抗评测结果除了常规分割指标，还会记录：

- 扰动预算参数
- `mean_linf`
- `max_linf`
- `mean_l2`

### 6. 对比 clean 与 adversarial 结果

```bash
python scripts/evaluate_robustness.py \
  --clean-summary results/reports/voc_clean_eval/UperNet_ConvNext_T_VOC_clean/summary.json \
  --adv-summary results/reports/voc_adv_eval/UperNet_ConvNext_T_VOC_clean_fgsm/summary.json \
  --output-dir results/reports/robustness/UperNet_ConvNext_T_fgsm
```

### 7. 稀疏防御校准与阈值搜索

先为 base checkpoint 生成统计 sidecar：

```bash
python scripts/prepare_sparse_defense.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --defense-config configs/defenses/meansparse_example.yaml \
  --dataset-root datasets
```

再在指定 split 上做 clean / PGD 联合阈值搜索：

```bash
python scripts/search_sparse_thresholds.py \
  --family upernet_convnext \
  --checkpoint models/UperNet_ConvNext_T_VOC_clean.pth \
  --variant meansparse \
  --stats-path models/defenses/meansparse_example_stats.pt \
  --dataset-root datasets \
  --dataset-split train \
  --output-dir results/reports/sparse_search/UperNet_ConvNext_T_VOC_clean_meansparse
```

搜索会输出：

- `search_summary.json`
- `search_summary.md`
- 每个阈值的 clean / PGD 结果子目录

### 8. 运行黑盒迁移攻击评测

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

该脚本会输出目标模型上的：

- `mIoU`
- `mAcc`
- `aAcc`
- 扰动统计
- `per_class_metrics.csv`

### 9. 批量评测 VOC clean 模型

```bash
python scripts/evaluate_all_voc_clean_models.py \
  --dataset-root datasets \
  --output-dir results/reports/voc_clean_eval
```

聚合结果会写入：

- `summary_all.json`
- `summary_all.csv`
- `summary_all.md`

## 进阶批量协议

仓库已经补齐三类 Slurm 批量协议：

- attack suite：`materialize_voc_attack_suite_manifest.py` -> `launch_voc_attack_suite.py` -> `summarize_voc_attack_suite.py`
- strict transfer protocol：`materialize_voc_transfer_protocol_manifest.py` -> `launch_voc_transfer_protocol.py` -> `summarize_voc_transfer_protocol.py`
- full all-attacks suite：`launch_voc_all_attacks.py` -> `summarize_voc_all_attacks.py`

如果只是做单次 clean 评测，也可以直接提交：

```bash
sbatch scripts/submit_voc_clean_eval.sbatch
```

也可以覆盖默认参数：

```bash
RESULTS_ROOT=results/reports/voc_clean_eval_debug \
DATASET_ROOT=datasets \
sbatch scripts/submit_voc_clean_eval.sbatch
```

脚本默认申请：

- `gpu:1`
- `8 CPU`
- `48G` 内存
- `8 小时`

## 输出目录约定

常见输出位置如下：

- `results/reports/dataset_check/`
- `results/reports/evaluation/`
- `results/reports/voc_clean_eval/`
- `results/reports/voc_adv_eval/`
- `results/reports/voc_transfer_eval/`
- `results/reports/robustness/`
- `results/reports/sparse_search/`
- `results/reports/comparison/`
- `results/reports/robustness_comparison/`

当前仓库的导出器会自动创建父目录，因此大多数脚本可以直接写入新的结果路径。

## 测试

运行全部单元测试：

```bash
python -m unittest discover -s tests
```

当前测试覆盖的重点包括：

- 数据扫描
- 分割指标计算
- 攻击配置与预算约束
- 对抗预览与响应区域可视化
- 稀疏防御与模型加载

## 补充文档

- `GUIDE.md`：完整工作流说明，覆盖 clean / adv / sparse / transfer / suite 的推荐用法。
- `README_FRONTEND.md`：前端页面、交互和展示重点。
- `src/attacks/README.md`：各攻击方法的原理与实现约定。
- `configs/defenses/README.md`：稀疏防御 YAML 的字段说明与推荐流程。
- `models/README.md`：本地权重和 sidecar 的存放约定。

## 说明

- `PascalVOCValidationDataset` 当前支持 `val`，部分搜索流程也支持指定 `train` split。
- VOC 评测会对输入图像做 `resize_short=473` 和中心裁剪到 `473x473`。
- `run_attack.py` 和 `run_transfer_attack.py` 支持 `--no-strict` 以兼容部分 checkpoint key 不完全匹配的情况。
- 如果只是想把任意 JSON 摘要转成 Markdown，可使用 `python scripts/generate_report.py --help` 查看参数。
