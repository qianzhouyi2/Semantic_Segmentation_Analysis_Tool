# models

本目录用于放置本地语义分割模型权重。

默认不将 `.pth` / `.pt` / `.ckpt` 文件提交到 Git 仓库，以避免仓库体积过大或 GitHub 上传失败。

如果需要共享权重，建议在主仓库中只保留下载说明，并将实际文件放在 Git LFS 或外部模型托管平台。

对于 `meansparse` / `extrasparse` / `cc_extra_sparse` / `dir_extra_sparse` / `margin_extra_sparse` 这类防御，不建议再额外导出一份完整模型权重。当前仓库支持保留原始 base checkpoint，并把预计算得到的稀疏统计 sidecar 单独存放在例如 `models/defenses/*.pt` 下，再通过 `--defense-config` 在加载时自动挂载。
