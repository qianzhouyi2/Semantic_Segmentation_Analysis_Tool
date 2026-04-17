# Selected Sparse Defense Configs

- search_root: results/reports/voc_train_threshold_search_rerun
- output_dir: configs/defenses
- num_configs: 30

| checkpoint | family | variant | threshold | config |
| --- | --- | --- | ---: | --- |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `cc_extra_sparse` | 0.20 | `Segmenter_ViT_S_VOC_adv_cc_extra_sparse.yaml` |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `dir_extra_sparse` | 0.10 | `Segmenter_ViT_S_VOC_adv_dir_extra_sparse.yaml` |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `extrasparse` | 0.20 | `Segmenter_ViT_S_VOC_adv_extrasparse.yaml` |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `margin_extra_sparse` | 0.05 | `Segmenter_ViT_S_VOC_adv_margin_extra_sparse.yaml` |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `meansparse` | 0.20 | `Segmenter_ViT_S_VOC_adv_meansparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `cc_extra_sparse` | 0.25 | `Segmenter_ViT_S_VOC_clean_cc_extra_sparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `dir_extra_sparse` | 0.10 | `Segmenter_ViT_S_VOC_clean_dir_extra_sparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `extrasparse` | 0.25 | `Segmenter_ViT_S_VOC_clean_extrasparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `margin_extra_sparse` | 0.10 | `Segmenter_ViT_S_VOC_clean_margin_extra_sparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `meansparse` | 0.25 | `Segmenter_ViT_S_VOC_clean_meansparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `cc_extra_sparse` | 0.25 | `UperNet_ConvNext_T_VOC_adv_cc_extra_sparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `dir_extra_sparse` | 0.25 | `UperNet_ConvNext_T_VOC_adv_dir_extra_sparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `extrasparse` | 0.25 | `UperNet_ConvNext_T_VOC_adv_extrasparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `margin_extra_sparse` | 0.20 | `UperNet_ConvNext_T_VOC_adv_margin_extra_sparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `meansparse` | 0.25 | `UperNet_ConvNext_T_VOC_adv_meansparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `cc_extra_sparse` | 0.35 | `UperNet_ConvNext_T_VOC_clean_cc_extra_sparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `dir_extra_sparse` | 0.25 | `UperNet_ConvNext_T_VOC_clean_dir_extra_sparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `extrasparse` | 0.25 | `UperNet_ConvNext_T_VOC_clean_extrasparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `margin_extra_sparse` | 0.25 | `UperNet_ConvNext_T_VOC_clean_margin_extra_sparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `meansparse` | 0.25 | `UperNet_ConvNext_T_VOC_clean_meansparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `cc_extra_sparse` | 0.25 | `UperNet_ResNet50_VOC_adv_cc_extra_sparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `dir_extra_sparse` | 0.20 | `UperNet_ResNet50_VOC_adv_dir_extra_sparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `extrasparse` | 0.20 | `UperNet_ResNet50_VOC_adv_extrasparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `margin_extra_sparse` | 0.15 | `UperNet_ResNet50_VOC_adv_margin_extra_sparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `meansparse` | 0.25 | `UperNet_ResNet50_VOC_adv_meansparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `cc_extra_sparse` | 0.35 | `UperNet_ResNet50_VOC_clean_cc_extra_sparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `dir_extra_sparse` | 0.25 | `UperNet_ResNet50_VOC_clean_dir_extra_sparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `extrasparse` | 0.25 | `UperNet_ResNet50_VOC_clean_extrasparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `margin_extra_sparse` | 0.25 | `UperNet_ResNet50_VOC_clean_margin_extra_sparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `meansparse` | 0.10 | `UperNet_ResNet50_VOC_clean_meansparse.yaml` |
