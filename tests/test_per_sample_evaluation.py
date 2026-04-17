from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.segmentation import evaluate_segmentation_model


class _ToyDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            (
                torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
                torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
                "sample_a.png",
            ),
            (
                torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
                torch.tensor([[1, 0], [0, 1]], dtype=torch.int64),
                "sample_b.png",
            ),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class _ToyModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        channel = images[:, 0]
        background = 1.0 - channel
        foreground = channel
        return torch.stack([background, foreground], dim=1)


class PerSampleEvaluationTest(unittest.TestCase):
    def test_evaluate_segmentation_model_collects_per_sample_metrics(self) -> None:
        dataset = _ToyDataset()
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        payload = evaluate_segmentation_model(
            model=_ToyModel(),
            dataloader=dataloader,
            num_classes=2,
            device="cpu",
            class_names={0: "background", 1: "foreground"},
            collect_per_sample=True,
        )

        self.assertEqual(payload["processed_samples"], 2)
        self.assertEqual(payload["filenames"], ["sample_a.png", "sample_b.png"])
        self.assertIn("per_sample_metrics", payload)
        self.assertEqual(len(payload["per_sample_metrics"]), 2)
        self.assertEqual(payload["per_sample_metrics"][0]["filename"], "sample_a.png")
        self.assertAlmostEqual(float(payload["per_sample_metrics"][0]["sample_miou"]), 1.0)
        self.assertAlmostEqual(float(payload["per_sample_metrics"][1]["sample_dice"]), 1.0)


if __name__ == "__main__":
    unittest.main()
