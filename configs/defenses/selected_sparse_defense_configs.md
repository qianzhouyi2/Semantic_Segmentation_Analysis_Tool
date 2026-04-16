# Selected Sparse Defense Configs

- search_root: results/reports/voc_train_threshold_search_rerun
- output_dir: configs/defenses
- num_configs: 12

| checkpoint | family | variant | threshold | config |
| --- | --- | --- | ---: | --- |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `extrasparse` | 0.20 | `Segmenter_ViT_S_VOC_adv_extrasparse.yaml` |
| `Segmenter_ViT_S_VOC_adv` | `segmenter_vit_s` | `meansparse` | 0.20 | `Segmenter_ViT_S_VOC_adv_meansparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `extrasparse` | 0.25 | `Segmenter_ViT_S_VOC_clean_extrasparse.yaml` |
| `Segmenter_ViT_S_VOC_clean` | `segmenter_vit_s` | `meansparse` | 0.25 | `Segmenter_ViT_S_VOC_clean_meansparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `extrasparse` | 0.25 | `UperNet_ConvNext_T_VOC_adv_extrasparse.yaml` |
| `UperNet_ConvNext_T_VOC_adv` | `upernet_convnext` | `meansparse` | 0.25 | `UperNet_ConvNext_T_VOC_adv_meansparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `extrasparse` | 0.25 | `UperNet_ConvNext_T_VOC_clean_extrasparse.yaml` |
| `UperNet_ConvNext_T_VOC_clean` | `upernet_convnext` | `meansparse` | 0.25 | `UperNet_ConvNext_T_VOC_clean_meansparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `extrasparse` | 0.20 | `UperNet_ResNet50_VOC_adv_extrasparse.yaml` |
| `UperNet_ResNet50_VOC_adv` | `upernet_resnet50` | `meansparse` | 0.25 | `UperNet_ResNet50_VOC_adv_meansparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `extrasparse` | 0.25 | `UperNet_ResNet50_VOC_clean_extrasparse.yaml` |
| `UperNet_ResNet50_VOC_clean` | `upernet_resnet50` | `meansparse` | 0.10 | `UperNet_ResNet50_VOC_clean_meansparse.yaml` |
