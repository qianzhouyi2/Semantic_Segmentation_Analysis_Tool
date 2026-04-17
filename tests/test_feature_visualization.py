from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.backbones.vit import VisionTransformer
from src.robustness.visualization import resolve_heatmap_display_bounds, save_layerwise_feature_visualizations


class VisionTransformerIntermediateFeatureTest(unittest.TestCase):
    def test_forward_tokens_collects_hidden_states_per_block(self) -> None:
        model = VisionTransformer(
            image_size=(16, 16),
            patch_size=8,
            n_layers=2,
            d_model=16,
            d_ff=32,
            n_heads=4,
            n_cls=3,
            dropout=0.0,
            drop_path_rate=0.0,
        )
        inputs = torch.randn(1, 3, 16, 16)

        tokens, hidden_states = model.forward_tokens(inputs, collect_hidden_states=True)

        self.assertEqual(tokens.shape, (1, 5, 16))
        self.assertEqual(len(hidden_states), 2)
        self.assertEqual(hidden_states[0].shape, (1, 5, 16))
        self.assertEqual(hidden_states[1].shape, (1, 5, 16))


class FeatureVisualizationExportTest(unittest.TestCase):
    def test_save_layerwise_feature_visualizations_writes_artifacts(self) -> None:
        clean_image = torch.linspace(0.0, 1.0, steps=3 * 8 * 8, dtype=torch.float32).reshape(1, 3, 8, 8)
        adversarial_image = (clean_image + 0.05).clamp(0.0, 1.0)
        perturbation = adversarial_image - clean_image
        clean_features = {
            "encoder:block00": torch.randn(1, 4, 4, 4),
            "encoder:block01": torch.randn(1, 8, 2, 2),
        }
        adversarial_features = {
            "encoder:block00": clean_features["encoder:block00"] + 0.1,
            "encoder:block01": clean_features["encoder:block01"] - 0.2,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata = save_layerwise_feature_visualizations(
                output_dir=Path(tmp_dir),
                sample_key="sample/a",
                clean_image=clean_image,
                adversarial_image=adversarial_image,
                perturbation=perturbation,
                clean_features=clean_features,
                adversarial_features=adversarial_features,
                max_layers=1,
            )

            sample_dir = Path(metadata["sample_dir"])
            self.assertTrue((sample_dir / "input.png").exists())
            self.assertTrue((sample_dir / "adversarial.png").exists())
            self.assertTrue((sample_dir / "perturbation.png").exists())
            self.assertEqual(len(metadata["layers"]), 1)
            self.assertTrue(Path(metadata["layers"][0]["figure_path"]).exists())

            json_payload = json.loads((sample_dir / "feature_maps.json").read_text(encoding="utf-8"))
            self.assertEqual(json_payload["sample_key"], "sample/a")
            self.assertEqual(len(json_payload["layers"]), 1)

    def test_resolve_heatmap_display_bounds_supports_shared_and_fixed_modes(self) -> None:
        heatmaps = [
            torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32).numpy(),
            torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32).numpy(),
        ]

        shared_bounds = resolve_heatmap_display_bounds(heatmaps, scale_mode="shared", percentile_clip_upper=100.0)
        fixed_bounds = resolve_heatmap_display_bounds(heatmaps, scale_mode="fixed", fixed_range=(0.0, 1.0))

        self.assertEqual(shared_bounds, [(0.0, 7.0), (0.0, 7.0)])
        self.assertEqual(fixed_bounds, [(0.0, 1.0), (0.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
