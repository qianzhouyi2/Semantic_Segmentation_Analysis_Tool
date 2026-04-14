# Semantic Segmentation Analysis Tool

一个面向语义分割实验的本地分析仓库，覆盖数据集体检、预测结果评估、Pascal VOC clean 验证集评测、对抗攻击评估，以及结果导出与可视化。

当前仓库已经不是最初的 MVP 骨架，而是一个可以直接跑通数据检查、指标统计、模型评测和鲁棒性分析的工具集。

## 当前能力

- 数据集扫描与体检：检查图像和 mask 配对、缺失文件、尺寸不一致、空 mask、非法标签，并统计类别分布。
- Streamlit 可视化：提供数据集扫描、三联图预览、对抗特征可视化和独立的 CAM 预览页面。
- 分割结果评估：对预测 mask 与 GT mask 计算 pixel accuracy、mIoU、Dice、Precision、Recall、F1。
- Pascal VOC clean 评测：加载本地 checkpoint，在 VOC2012 val 上输出标准分割指标和逐类结果。
- 对抗攻击评测：支持多种分割攻击配置，并导出攻击后指标与扰动统计。
- 鲁棒性分析：比较 clean 与 adversarial 结果，生成 robustness summary 和 Markdown 报告。
- 结果导出：统一生成 `summary.json`、`per_class_metrics.csv`、`report.md` 等输出。
- 批量评测与 Slurm：支持对一组本地模型批量跑 VOC clean 评测，并提供现成的 `sbatch` 脚本。

## 已支持的模型与攻击

### 模型族

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

对应 YAML 配置位于 `configs/attacks/`，例如 `fgsm.yaml`、`pgd.yaml`、`segpgd.yaml`、`transegpgd.yaml`。

## 项目结构

```text
.
├── app.py                         # Streamlit 入口
├── configs/
│   ├── attacks/                  # 攻击配置
│   ├── datasets/                 # 数据集路径与后缀配置
│   ├── labels/                   # 标签 ID / 名称 / 调色板配置
│   └── reports/                  # 报告配置示例
├── models/                       # 本地 checkpoint 存放目录
├── scripts/                      # CLI 脚本与 Slurm 提交脚本
├── src/
│   ├── attacks/                  # 攻击实现与运行器
│   ├── datasets/                 # VOC loader、扫描与统计
│   ├── evaluation/               # clean / adversarial 评测
│   ├── metrics/                  # 分割指标
│   ├── models/                   # 模型构建、注册、适配
│   ├── reporting/                # JSON / CSV / Markdown 导出
│   ├── robustness/               # 鲁棒性分析
│   └── visualization/            # 三联图可视化
├── tests/                        # scanner / metrics / attacks 单元测试
└── results/                      # 评测输出目录
```

所有脚本都应从仓库根目录运行，这样 `scripts/_bootstrap.py` 才能正确注入项目路径。

## 环境准备

推荐环境：

- Python `3.11`
- Conda 环境名 `segtool`
- PyTorch `2.4.1`，默认依赖文件使用 CUDA `12.1` 官方 wheel index

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

### Pascal VOC clean / adversarial 评测

VOC loader 期望的数据目录为：

```text
datasets/
└── VOCdevkit/
    └── VOC2012/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── ImageSets/Segmentation/val.txt
```

本地模型权重放在 `models/`。仓库默认不提交 `.pt`、`.pth`、`.ckpt` 大文件。

## 快速开始

### 1. 启动可视化页面

```bash
streamlit run app.py
```

页面当前提供四块功能：

- `Dataset Scan`：扫描图像与 mask，展示总量、空 mask、异常标签和类别分布。
- `Triplet Preview`：渲染原图、GT mask、预测 mask 的三联图。
- `Adversarial Feature Preview`：单样本攻击预览，查看输入扰动、预测变化和逐层特征热图。
- `CAM Preview`：单样本类激活图预览，固定使用最深可用 CAM 层，对比 clean / adversarial 注意区域。

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

输出目录包含：

- `summary.json`
- `per_class_metrics.csv`
- `report.md`
- `evaluate.log`

### 5. 运行对抗攻击评测

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

### 7. 比较两次评测结果

比较两份 segmentation summary：

```bash
python scripts/compare_runs.py \
  --baseline results/reports/voc_clean_eval/run_a/summary.json \
  --candidate results/reports/voc_clean_eval/run_b/summary.json \
  --output-dir results/reports/comparison/run_a_vs_run_b
```

比较两份 robustness summary：

```bash
python scripts/compare_robustness.py \
  --baseline results/reports/robustness/run_a/summary.json \
  --candidate results/reports/robustness/run_b/summary.json \
  --output-dir results/reports/robustness_comparison/run_a_vs_run_b
```

### 8. 批量评测 VOC clean 模型

```bash
python scripts/evaluate_all_voc_clean_models.py \
  --dataset-root datasets \
  --output-dir results/reports/voc_clean_eval
```

该脚本当前会遍历以下命名约定的 checkpoint：

- `models/UperNet_ConvNext_T_VOC_adv.pth`
- `models/UperNet_ConvNext_T_VOC_clean.pth`
- `models/UperNet_ResNet50_VOC_adv.pth`
- `models/UperNet_ResNet50_VOC_clean.pth`
- `models/Segmenter_ViT_S_VOC_adv.pth`
- `models/Segmenter_ViT_S_VOC_clean.pth`

聚合结果会写入：

- `summary_all.json`
- `summary_all.csv`
- `summary_all.md`

如果你手动调整了输出目录命名，可以再运行：

```bash
python scripts/refresh_voc_clean_summary.py \
  --output-dir results/reports/voc_clean_eval
```

### 9. 用 Slurm 提交 GPU clean 评测

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
- `results/reports/robustness/`
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
- 攻击配置与攻击预算约束
- 新增攻击注册与基本行为

## 说明

- `PascalVOCValidationDataset` 当前只支持 `val` split。
- VOC 评测会对输入图像做 `resize_short=473` 和中心裁剪到 `473x473`。
- `run_attack.py` 支持 `--no-strict` 以兼容部分 checkpoint key 不完全匹配的情况。
- 如果只是想把任意 JSON 摘要转成 Markdown，可使用 `python scripts/generate_report.py --help` 查看参数。
