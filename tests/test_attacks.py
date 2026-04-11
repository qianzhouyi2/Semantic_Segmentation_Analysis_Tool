from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import AttackConfig, AttackRunner, CosPGDAttack, FGSMAttack, PGDAttack
from src.attacks.losses import segmentation_cospgd_loss, segmentation_segpgd_loss
from src.models.base import TorchSegmentationModelAdapter


def _build_toy_adapter() -> TorchSegmentationModelAdapter:
    model = nn.Conv2d(1, 2, kernel_size=1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[[[4.0]]], [[[-4.0]]]]))
        model.bias.copy_(torch.tensor([-1.0, 1.0]))
    return TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")


class AttackConfigTest(unittest.TestCase):
    def test_from_dict_normalizes_values_and_keeps_extra_fields(self) -> None:
        config = AttackConfig.from_dict(
            {
                "name": "PGD",
                "epsilon": "0.25",
                "step_size": "0.1",
                "steps": "4",
                "custom_flag": True,
            }
        )

        self.assertEqual(config.name, "pgd")
        self.assertAlmostEqual(config.epsilon, 0.25)
        self.assertAlmostEqual(config.step_size or 0.0, 0.1)
        self.assertEqual(config.steps, 4)
        self.assertEqual(config.extra["custom_flag"], True)


class PGDAttackTest(unittest.TestCase):
    def test_pgd_respects_linf_budget_and_keeps_model_grads_clean(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((2, 1, 2, 2), 0.9, dtype=torch.float32)
        targets = torch.zeros((2, 2, 2), dtype=torch.long)
        config = AttackConfig(name="pgd", epsilon=0.2, step_size=0.05, steps=4, random_start=True)

        output = PGDAttack(adapter, config).run(images, targets)

        self.assertLessEqual(float(output.perturbation.abs().max().item()), 0.2 + 1e-6)
        self.assertGreaterEqual(float(output.adversarial_images.min().item()), 0.0)
        self.assertLessEqual(float(output.adversarial_images.max().item()), 1.0)
        self.assertTrue(all(parameter.grad is None for parameter in adapter.model.parameters()))

    def test_untargeted_pgd_increases_clean_label_loss(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float32)
        targets = torch.zeros((1, 1, 1), dtype=torch.long)
        config = AttackConfig(name="pgd", epsilon=0.8, step_size=0.2, steps=4, random_start=False)

        clean_loss = F.cross_entropy(adapter.logits(images), targets)
        output = PGDAttack(adapter, config).run(images, targets)
        adversarial_loss = F.cross_entropy(adapter.logits(output.adversarial_images), targets)
        prediction = int(adapter.predict(output.adversarial_images).item())

        self.assertGreater(float(adversarial_loss.item()), float(clean_loss.item()))
        self.assertEqual(prediction, 1)

    def test_targeted_pgd_decreases_target_label_loss(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float32)
        target_labels = torch.ones((1, 1, 1), dtype=torch.long)
        config = AttackConfig(name="pgd", epsilon=0.8, step_size=0.2, steps=4, random_start=False, targeted=True)

        clean_loss = F.cross_entropy(adapter.logits(images), target_labels)
        output = PGDAttack(adapter, config).run(images, target_labels)
        adversarial_loss = F.cross_entropy(adapter.logits(output.adversarial_images), target_labels)
        prediction = int(adapter.predict(output.adversarial_images).item())

        self.assertLess(float(adversarial_loss.item()), float(clean_loss.item()))
        self.assertEqual(prediction, 1)


class FGSMAttackTest(unittest.TestCase):
    def test_fgsm_untargeted_increases_clean_label_loss(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float32)
        targets = torch.zeros((1, 1, 1), dtype=torch.long)
        config = AttackConfig(name="fgsm", epsilon=0.8)

        clean_loss = F.cross_entropy(adapter.logits(images), targets)
        output = FGSMAttack(adapter, config).run(images, targets)
        adversarial_loss = F.cross_entropy(adapter.logits(output.adversarial_images), targets)

        self.assertGreater(float(adversarial_loss.item()), float(clean_loss.item()))
        self.assertEqual(int(adapter.predict(output.adversarial_images).item()), 1)

    def test_fgsm_targeted_decreases_target_label_loss(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float32)
        target_labels = torch.ones((1, 1, 1), dtype=torch.long)
        config = AttackConfig(name="fgsm", epsilon=0.8, targeted=True)

        clean_loss = F.cross_entropy(adapter.logits(images), target_labels)
        output = FGSMAttack(adapter, config).run(images, target_labels)
        adversarial_loss = F.cross_entropy(adapter.logits(output.adversarial_images), target_labels)

        self.assertLess(float(adversarial_loss.item()), float(clean_loss.item()))
        self.assertEqual(int(adapter.predict(output.adversarial_images).item()), 1)

    def test_fgsm_ignores_invalid_pixels_like_reference_masking(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((1, 1, 2, 2), 0.9, dtype=torch.float32)
        targets = torch.full((1, 2, 2), 255, dtype=torch.long)
        config = AttackConfig(name="fgsm", epsilon=0.2, ignore_index=255)

        output = FGSMAttack(adapter, config).run(images, targets)

        self.assertTrue(torch.allclose(output.adversarial_images, images))
        self.assertEqual(output.metadata["valid_pixels"], 0)
        self.assertAlmostEqual(float(output.metadata["loss"]), 0.0)


class CosPGDAttackTest(unittest.TestCase):
    def test_cospgd_loss_matches_manual_cosine_scaling(self) -> None:
        logits = torch.tensor(
            [[[[2.0, -1.0]], [[-2.0, 1.0]]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[[0, 0]]], dtype=torch.long)

        loss = segmentation_cospgd_loss(logits=logits, targets=targets)

        probabilities = F.softmax(logits, dim=1)
        one_hot_targets = F.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).to(dtype=logits.dtype)
        cosine_similarity = F.cosine_similarity(probabilities, one_hot_targets, dim=1)
        loss_map = F.cross_entropy(logits, targets, reduction="none")
        expected = (cosine_similarity * loss_map).mean()
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)

    def test_cospgd_respects_linf_budget_and_keeps_model_grads_clean(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((2, 1, 2, 2), 0.9, dtype=torch.float32)
        targets = torch.zeros((2, 2, 2), dtype=torch.long)
        config = AttackConfig(name="cospgd", epsilon=0.2, step_size=0.05, steps=4, random_start=True)

        output = CosPGDAttack(adapter, config).run(images, targets)

        self.assertLessEqual(float(output.perturbation.abs().max().item()), 0.2 + 1e-6)
        self.assertGreaterEqual(float(output.adversarial_images.min().item()), 0.0)
        self.assertLessEqual(float(output.adversarial_images.max().item()), 1.0)
        self.assertTrue(all(parameter.grad is None for parameter in adapter.model.parameters()))


class SegPGDAttackTest(unittest.TestCase):
    def test_segpgd_loss_matches_manual_branch_weighting(self) -> None:
        logits = torch.tensor(
            [[[[2.0, -1.0]], [[-2.0, 1.0]]]],
            dtype=torch.float32,
        )
        targets = torch.tensor([[[0, 0]]], dtype=torch.long)

        loss = segmentation_segpgd_loss(logits=logits, targets=targets, iteration=2, iterations=4)

        lambda_t = 2.0 / 8.0
        loss_map = F.cross_entropy(logits, targets, reduction="none")
        weights = torch.tensor([[[1.0 - lambda_t, lambda_t]]], dtype=logits.dtype)
        expected = (weights * loss_map).mean()
        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)


class AttackRegistrationTest(unittest.TestCase):
    def test_runner_supports_new_attack_variants(self) -> None:
        adapter = _build_toy_adapter()
        runner = AttackRunner(adapter)
        images = torch.full((1, 1, 2, 2), 0.9, dtype=torch.float32)
        targets = torch.zeros((1, 2, 2), dtype=torch.long)
        attack_names = (
            "segpgd",
            "sea",
            "mi-fgsm",
            "ni-fgsm",
            "di2-fgsm",
            "ti-fgsm",
            "ni+di+ti",
            "dag",
            "tass",
            "transegpgd",
            "fspgd",
        )

        for attack_name in attack_names:
            with self.subTest(attack=attack_name):
                config = AttackConfig(
                    name=attack_name,
                    epsilon=0.2,
                    step_size=0.05,
                    steps=2,
                    random_start=False,
                    ignore_index=255,
                )
                output = runner.run(config, images, targets)
                self.assertLessEqual(float(output.perturbation.abs().max().item()), 0.2 + 1e-6)
                self.assertGreaterEqual(float(output.adversarial_images.min().item()), 0.0)
                self.assertLessEqual(float(output.adversarial_images.max().item()), 1.0)


if __name__ == "__main__":
    unittest.main()
