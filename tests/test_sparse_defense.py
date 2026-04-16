from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_model, build_model_from_checkpoint
from src.models.sparse import (
    SparseDefenseConfig,
    apply_sparse_defense,
    export_sparse_sidecar,
    iter_meansparse_modules,
    iter_postsparse_modules,
)


class SparseDefenseIntegrationTest(unittest.TestCase):
    def test_apply_sparse_defense_supports_segmenter_meansparse(self) -> None:
        model = build_model("segmenter_vit_s", num_classes=3)
        apply_sparse_defense(
            model,
            family="segmenter_vit_s",
            config=SparseDefenseConfig(variant="meansparse", stats_path=Path("/tmp/dummy.pt"), threshold=0.25),
            load_stats=False,
        )
        sparse_modules = list(iter_meansparse_modules(model))
        self.assertTrue(sparse_modules)
        self.assertAlmostEqual(float(sparse_modules[0].threshold.item()), 0.25)

    def test_build_model_from_checkpoint_loads_segmenter_meansparse_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "segmenter_base_model.pth"
            stats_path = root / "segmenter_meansparse_stats.pt"

            base_model = build_model("segmenter_vit_s", num_classes=3)
            torch.save({"state_dict": base_model.state_dict()}, checkpoint_path)

            sparse_model = build_model("segmenter_vit_s", num_classes=3)
            config = SparseDefenseConfig(variant="meansparse", stats_path=stats_path, threshold=0.15)
            apply_sparse_defense(sparse_model, family="segmenter_vit_s", config=config, load_stats=False)
            first_module = next(iter(iter_meansparse_modules(sparse_model)))
            first_module.running_mean.fill_(0.5)
            first_module.running_var.fill_(1.75)
            export_sparse_sidecar(
                sparse_model,
                family="segmenter_vit_s",
                config=config,
                output_path=stats_path,
            )

            loaded_model, missing_keys, unexpected_keys = build_model_from_checkpoint(
                family="segmenter_vit_s",
                checkpoint_path=checkpoint_path,
                num_classes=3,
                defense_config={
                    "variant": "meansparse",
                    "stats_path": str(stats_path),
                    "threshold": 0.15,
                },
            )

            self.assertEqual(missing_keys, [])
            self.assertEqual(unexpected_keys, [])
            loaded_module = next(iter(iter_meansparse_modules(loaded_model)))
            self.assertAlmostEqual(float(loaded_module.running_mean[0]), 0.5)
            self.assertAlmostEqual(float(loaded_module.running_var[0]), 1.75)
            self.assertAlmostEqual(float(loaded_module.threshold.item()), 0.15)

    def test_build_model_from_checkpoint_loads_meansparse_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "base_model.pth"
            stats_path = root / "meansparse_stats.pt"

            base_model = build_model("upernet_convnext", num_classes=3)
            torch.save({"state_dict": base_model.state_dict()}, checkpoint_path)

            sparse_model = build_model("upernet_convnext", num_classes=3)
            config = SparseDefenseConfig(variant="meansparse", stats_path=stats_path, threshold=0.35)
            apply_sparse_defense(sparse_model, family="upernet_convnext", config=config, load_stats=False)
            first_module = next(iter(iter_meansparse_modules(sparse_model)))
            first_module.running_mean.fill_(1.25)
            first_module.running_var.fill_(2.5)
            export_sparse_sidecar(
                sparse_model,
                family="upernet_convnext",
                config=config,
                output_path=stats_path,
            )

            loaded_model, missing_keys, unexpected_keys = build_model_from_checkpoint(
                family="upernet_convnext",
                checkpoint_path=checkpoint_path,
                num_classes=3,
                defense_config={
                    "variant": "meansparse",
                    "stats_path": str(stats_path),
                    "threshold": 0.35,
                },
            )

            self.assertEqual(missing_keys, [])
            self.assertEqual(unexpected_keys, [])
            loaded_module = next(iter(iter_meansparse_modules(loaded_model)))
            self.assertAlmostEqual(float(loaded_module.running_mean[0]), 1.25)
            self.assertAlmostEqual(float(loaded_module.running_var[0]), 2.5)
            self.assertAlmostEqual(float(loaded_module.threshold.item()), 0.35)
            self.assertEqual(getattr(loaded_model, "_sparse_defense_info")["variant"], "meansparse")

    def test_build_model_from_checkpoint_loads_postsparse_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_path = root / "base_model.pth"
            stats_path = root / "postsparse_stats.pt"

            base_model = build_model("upernet_resnet50", num_classes=3)
            torch.save({"state_dict": base_model.state_dict()}, checkpoint_path)

            sparse_model = build_model("upernet_resnet50", num_classes=3)
            config = SparseDefenseConfig(variant="dir_extra_sparse", stats_path=stats_path, threshold=0.2)
            apply_sparse_defense(sparse_model, family="upernet_resnet50", config=config, load_stats=False)
            first_module = next(iter(iter_postsparse_modules(sparse_model)))
            first_module.running_mean.fill_(0.75)
            first_module.running_var.fill_(1.5)
            first_module.set_class_statistics(
                torch.full_like(first_module.class_conditional_mean, 1.0),
                torch.full_like(first_module.class_conditional_std, 2.0),
                torch.arange(first_module.num_classes, dtype=torch.long),
            )
            export_sparse_sidecar(
                sparse_model,
                family="upernet_resnet50",
                config=config,
                output_path=stats_path,
            )

            loaded_model, missing_keys, unexpected_keys = build_model_from_checkpoint(
                family="upernet_resnet50",
                checkpoint_path=checkpoint_path,
                num_classes=3,
                defense_config={
                    "variant": "dir_extra_sparse",
                    "stats_path": str(stats_path),
                    "threshold": 0.2,
                },
            )

            self.assertEqual(missing_keys, [])
            self.assertEqual(unexpected_keys, [])
            loaded_modules = list(iter_postsparse_modules(loaded_model))
            loaded_module = loaded_modules[0]
            self.assertAlmostEqual(float(loaded_module.running_mean[0]), 0.75)
            self.assertAlmostEqual(float(loaded_module.running_var[0]), 1.5)
            self.assertAlmostEqual(float(loaded_module.class_conditional_mean[0, 0]), 1.0)
            self.assertAlmostEqual(float(loaded_module.class_conditional_std[0, 0]), 2.0)
            self.assertEqual(int(loaded_module.class_count[2]), 2)
            self.assertTrue(any(float(module.classifier_weight.abs().sum()) > 0.0 for module in loaded_modules))


if __name__ == "__main__":
    unittest.main()
