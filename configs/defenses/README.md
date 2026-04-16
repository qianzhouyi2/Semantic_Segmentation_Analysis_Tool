# Sparse Defense Configs

本目录放 `meansparse` / `extrasparse` / postsparse 变体的 YAML 配置。

最小字段：

- `name` 或 `variant`
- `threshold`
- `stats_path`

postsparse 变体还支持：

- `direction_mode`
- `lambda_mix`
- `alpha0`
- `alpha0_mode`
- `beta`
- `beta_scale`
- `tau`

推荐流程：

1. 先用 `python scripts/prepare_sparse_defense.py --family ... --checkpoint ... --defense-config ...` 生成统计 sidecar。
2. 再在 `scripts/evaluate_voc_clean.py` 或 `scripts/run_attack.py` 里传同一份 `--defense-config`。

说明：

- `upernet_convnext` 和 `upernet_resnet50` 走 feature-map sparse。
- `segmenter_vit_s` 走 encoder token-space sparse，统计量对应的是 ViT token 通道。
