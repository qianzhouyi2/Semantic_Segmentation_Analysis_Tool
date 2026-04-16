# Frontend README

本仓库前端是一个基于 Streamlit 的轻量本地界面，入口文件为 `app.py`。它不负责训练或批量评测调度，主要承担以下职责：

- 数据集扫描与类别分布展示
- 原图 / GT / prediction 三联图预览
- Pascal VOC 单样本的对抗攻击预览
- 逐层特征热图与 sample delta 热图可视化
- CAM 预览
- 响应区域分析模块

## 1. 启动方式

在仓库根目录执行：

```bash
conda activate segtool
streamlit run app.py
```

默认页面会使用侧边栏中的：

- `Dataset config`
- `Label config`

对抗预览页还会自动发现：

- `models/` 下的 checkpoint
- `configs/attacks/` 下的攻击配置

## 2. 页面结构

前端当前有 5 个主页签：

### `Dataset Scan`

用途：

- 扫描图像和 mask 是否配对
- 检查空 mask、尺寸不一致、非法标签
- 展示类别分布表

核心代码：

- `app.py`
- `src/apps/dashboard.py`
- `src/datasets/scanner.py`
- `src/datasets/stats.py`

### `Triplet Preview`

用途：

- 浏览数据集中的 image / GT / prediction
- 支持手工路径模式
- 支持 Pascal VOC demo 模式

核心代码：

- `app.py`
- `src/visualization/triplet.py`

### `Adversarial Feature Preview`

用途：

- 选择 Pascal VOC val 单样本
- 选择模型 family 与 checkpoint
- 选择攻击与对应 config
- 设置扰动半径 `0-255`
- 查看 clean / adv 输入、预测结果、perturbation、sample delta 热图
- 用滑条逐层查看特征热图

核心代码：

- `app.py`
- `src/apps/adversarial_preview.py`
- `src/models/base.py`
- `src/models/backbones/convnext.py`
- `src/models/backbones/vit.py`
- `src/robustness/visualization.py`

### `CAM Preview`

用途：

- 选择 Pascal VOC val 单样本
- 选择模型 family 与 checkpoint
- 选择攻击与对应 config
- 设置扰动半径 `0-255`
- 选择目标类别并查看 clean / adv 的类激活图
- 固定使用最深可用 CAM 层，不暴露手动层选择

核心代码：

- `app.py`
- `src/apps/adversarial_preview.py`
- `src/visualization/cam.py`

### `Response Region Analysis`

用途：

- 选择 Pascal VOC val 单样本
- 选择模型 family 与 checkpoint
- 选择攻击与对应 config
- 设置扰动半径 `0-255`
- 选择目标类别，查看 clean / adv 的输入响应热图
- 提取高响应区域并展示 clean / adv 重叠区域

核心代码：

- `app.py`
- `src/visualization/response_region.py`

## 3. 当前对抗预览交互

对抗预览页的流程是：

1. 从 `datasets/VOCdevkit/VOC2012/...` 读取 `val` 样本
2. 从 `models/` 自动发现可用 checkpoint，并按 family 过滤
3. 从 `configs/attacks/*.yaml` 自动发现攻击配置，并按攻击名过滤
4. 使用 `扰动半径 (0-255)` 将预算换算为 `epsilon = radius / 255`
5. 点击按钮后执行一次单样本攻击，结果写入 `st.session_state`
6. 之后拖动层滑条时，只切换显示层，不重复跑攻击

前端当前显示的图包括：

- `Clean Input`
- `Adversarial Input`
- `Perturbation`
- `Sample Delta Heatmap`
- `Ground Truth`
- `Clean Prediction`
- `Adversarial Prediction`
- 当前层的 `Clean Feature`
- 当前层的 `Adversarial Feature`
- 当前层的 `Abs Diff`

## 4. 当前 CAM 交互

`CAM Preview` 模块复用同一套样本、模型和攻击选择逻辑，但交互遵循常规 CAM 用法：

1. 先运行一次单样本 clean / adversarial 预览
2. 前端自动选取当前模型最深的可用 CAM 层
3. 用户只选择 `target class`
4. 页面展示 clean CAM、adv CAM 和 CAM 差分图

这样做的原因是：

- 常规 CAM 一般固定在最后一个语义层
- 浅层更偏纹理与边缘，不适合作为默认解释结果
- 手动切层更适合研究调试，不适合作为默认前端交互

## 5. 特征层命名规则

为了让滑条下面显示稳定、可解释的层名，前端依赖统一的 feature key 命名。

当前约定如下：

- `upernet_convnext`
  - 逐 block: `backbone:stage{stage_idx}:block{block_idx}`
  - 例如 `backbone:stage2:block05`
- `upernet_resnet50`
  - 当前仍以 stage 级为主：`backbone:stage0` 到 `backbone:stage3`
- `segmenter_vit_s`
  - 逐 block: `encoder:block00`、`encoder:block01` ...

前端层排序优先级：

1. `backbone:stage*:block*`
2. `encoder:block*`
3. `backbone:stage*`

这意味着：

- `upernet_convnext` 前端会优先显示逐 block 层
- 如果某个模型只返回 stage 特征，前端会自动退回 stage 级显示

## 6. Sample Delta 热图

`Sample Delta Heatmap` 是 clean sample 和 adversarial sample 的像素级差分热图，不依赖模型特征。

当前实现：

- 输入：clean image、adv image
- 计算：对每个像素取通道绝对差，再对通道做平均
- 输出：2D 热图，经 colormap 映射后显示

相关代码：

- `src/robustness/visualization.py` 中的 `summarize_image_delta`
- `src/apps/adversarial_preview.py` 中的 `build_sample_delta_visualization`

它适合回答两个简单问题：

- 扰动主要集中在哪些区域
- 当前攻击是不是只改了少量局部区域

## 7. 响应区域分析

`Response Region Analysis` 模块与 CAM 模块一样复用单样本攻击预览流程，但方法落在输入空间而不是中间特征层：

1. 先运行一次 clean / adversarial 单样本预览
2. 选择目标类别
3. 对该类别在当前预测区域上的平均 logit 求输入梯度
4. 取梯度绝对值并对通道做平均，生成输入响应热图
5. 按分位阈值提取高响应区域，并展示 clean / adv 的重叠区域

模块会展示：

- `Clean Response Heatmap`
- `Adversarial Response Heatmap`
- `Response Diff Heatmap`
- `Clean High-Response Region`
- `Adversarial High-Response Region`
- `Stable Overlap Region`

## 8. 缓存与状态

前端目前用两类状态：

### `st.cache_resource`

用于缓存：

- VOC validation dataset
- 已加载的模型 adapter

目的：

- 避免重复加载大模型
- 避免每次切换层号都重新构建模型

### `st.session_state`

用于保存最近一次对抗预览结果，例如：

- `adv_preview_result`
- `adv_preview_signature`
- `adv_preview_checkpoint_info`
- `adv_layer_index`
- `cam_preview_cam_result`
- `cam_preview_cam_signature`

目的：

- 修改层滑条时复用上一次结果
- 当样本 / 模型 / 攻击配置变更时提示“当前展示的是上一次运行结果”

## 9. 前端开发建议

如果继续扩展前端，建议遵守下面的边界：

- 页面编排放在 `app.py`
- 复杂交互和数据组织逻辑放在 `src/apps/`
- 图像、热图、三联图渲染逻辑放在 `src/visualization/` 或 `src/robustness/visualization.py`
- 不要在 `app.py` 里直接堆大段模型推理细节
- 新增模型特征导出时，优先保证 feature key 命名稳定

推荐的扩展方向：

- 把层滑条改成“滑条 + 层名下拉框”双模式
- 增加 signed delta 热图
- 增加特征图导出按钮
- 增加 clean / adv logits 或 attention 的对比页
- 增加运行耗时与显存占用提示

## 10. 常见问题

### 页面里看不到模型 checkpoint

检查：

- 文件是否在 `models/`
- 后缀是否为 `.pt` / `.pth` / `.ckpt`
- 文件名是否能推断出 family，或者是否属于内置已知 checkpoint

### `Adversarial Feature Preview` 无法运行

检查：

- `datasets/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt` 是否存在
- checkpoint 是否可加载
- 当前环境是否有对应的 PyTorch / CUDA 支持

### `radius = 0` 为什么还能点运行

这是刻意保留的 clean baseline 模式。此时前端不会真正执行攻击，而是直接返回 clean sample，用于对照可视化。

### 为什么 `CAM Preview` 不再提供层选择

这是有意收敛到常规 CAM 交互。页面固定使用最深可用层，只保留类别选择，避免把研究调试参数暴露成默认操作。

## 10. 相关文件

- `app.py`
- `src/apps/dashboard.py`
- `src/apps/adversarial_preview.py`
- `src/visualization/triplet.py`
- `src/robustness/visualization.py`
- `src/models/base.py`
- `src/models/backbones/convnext.py`
- `src/models/backbones/vit.py`
