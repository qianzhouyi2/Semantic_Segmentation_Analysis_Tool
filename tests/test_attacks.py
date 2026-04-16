from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import (
    AttackConfig,
    AttackRunner,
    CosPGDAttack,
    FGSMAttack,
    PGDAttack,
    finalize_attack_runtime_aggregate,
    init_attack_runtime_aggregate,
    update_attack_runtime_aggregate,
)
from src.attacks.losses import segmentation_cospgd_loss, segmentation_segpgd_loss
from src.evaluation.adversarial import evaluate_adversarial_segmentation_model
from src.models.base import TorchSegmentationModelAdapter
from src.models.sparse import MeanSparse2d


def _build_toy_adapter() -> TorchSegmentationModelAdapter:
    model = nn.Conv2d(1, 2, kernel_size=1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[[[4.0]]], [[[-4.0]]]]))
        model.bias.copy_(torch.tensor([-1.0, 1.0]))
    return TorchSegmentationModelAdapter(model=model, num_classes=2, device="cpu")


class _SparseToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sparse = MeanSparse2d(1)
        self.sparse.set_threshold(0.0)
        self.head = nn.Conv2d(1, 2, kernel_size=1, bias=True)
        self.last_seen_backward_mode: str | None = None
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[[[4.0]]], [[[-4.0]]]]))
            self.head.bias.copy_(torch.tensor([-1.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_seen_backward_mode = self.sparse.attack_backward_mode
        return self.head(self.sparse(x))


def _build_sparse_toy_adapter() -> TorchSegmentationModelAdapter:
    return TorchSegmentationModelAdapter(model=_SparseToyModel(), num_classes=2, device="cpu")


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

    def test_scaled_multiplies_budget_and_preserves_original_values_in_extra(self) -> None:
        config = AttackConfig(name="fgsm", epsilon=0.1, step_size=0.02, steps=1)

        scaled = config.scaled(1.5)

        self.assertAlmostEqual(scaled.epsilon, 0.15)
        self.assertAlmostEqual(scaled.step_size or 0.0, 0.03)
        self.assertAlmostEqual(float(scaled.extra["base_epsilon"]), 0.1)
        self.assertAlmostEqual(float(scaled.extra["base_step_size"]), 0.02)
        self.assertAlmostEqual(float(scaled.extra["epsilon_scale"]), 1.5)

    def test_with_radius_255_sets_absolute_budget(self) -> None:
        config = AttackConfig(name="pgd", epsilon=8.0 / 255.0, step_size=2.0 / 255.0, steps=4)

        adjusted = config.with_radius_255(4)

        self.assertAlmostEqual(adjusted.epsilon, 4.0 / 255.0)
        self.assertAlmostEqual(adjusted.step_size or 0.0, 1.0 / 255.0)
        self.assertAlmostEqual(float(adjusted.extra["epsilon_radius_255"]), 4.0)

    def test_with_runtime_overrides_prefers_absolute_radius(self) -> None:
        config = AttackConfig(name="pgd", epsilon=8.0 / 255.0, step_size=2.0 / 255.0, steps=4)

        adjusted = config.with_runtime_overrides(
            epsilon_scale=1.5,
            epsilon_radius_255=2.0,
            attack_backward_mode="bpda_ste",
            num_restarts=3,
            eot_iters=5,
        )

        self.assertAlmostEqual(adjusted.epsilon, 2.0 / 255.0)
        self.assertAlmostEqual(adjusted.step_size or 0.0, 0.5 / 255.0)
        self.assertEqual(adjusted.attack_backward_mode, "bpda_ste")
        self.assertEqual(adjusted.num_restarts, 3)
        self.assertEqual(adjusted.eot_iters, 5)


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


class RPPGDAttackTest(unittest.TestCase):
    def test_rppgd_runs_with_generic_feature_map_fallback(self) -> None:
        adapter = _build_toy_adapter()
        runner = AttackRunner(adapter)
        images = torch.full((1, 1, 2, 2), 0.9, dtype=torch.float32)
        targets = torch.zeros((1, 2, 2), dtype=torch.long)
        config = AttackConfig(
            name="rppgd",
            epsilon=0.2,
            step_size=0.05,
            steps=3,
            random_start=True,
            ignore_index=255,
        )

        output = runner.run(config, images, targets)

        self.assertLessEqual(float(output.perturbation.abs().max().item()), 0.2 + 1e-6)
        self.assertGreaterEqual(float(output.adversarial_images.min().item()), 0.0)
        self.assertLessEqual(float(output.adversarial_images.max().item()), 1.0)
        self.assertIn("region", output.metadata)
        self.assertIn("prototype", output.metadata)
        self.assertEqual(output.metadata["attack"], "rppgd")


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
            "rppgd",
            "rp-pgd",
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


class AttackProtocolTest(unittest.TestCase):
    def test_runtime_aggregate_sums_restart_statistics_across_batches(self) -> None:
        aggregate = init_attack_runtime_aggregate(AttackConfig(name="fgsm", epsilon=0.1, num_restarts=2))

        update_attack_runtime_aggregate(
            aggregate,
            {
                "sparse_modules_configured": 1,
                "best_mean_score": 0.25,
                "selected_restart_histogram": [1, 1],
                "restart_summaries": [
                    {"restart_index": 0, "mean_score": 0.60},
                    {"restart_index": 1, "mean_score": 0.25},
                ],
            },
            batch_size=2,
        )
        update_attack_runtime_aggregate(
            aggregate,
            {
                "sparse_modules_configured": 1,
                "best_mean_score": 0.10,
                "selected_restart_histogram": [0, 1],
                "restart_summaries": [
                    {"restart_index": 0, "mean_score": 0.40},
                    {"restart_index": 1, "mean_score": 0.10},
                ],
            },
            batch_size=1,
        )

        summary = finalize_attack_runtime_aggregate(aggregate)

        self.assertEqual(summary["runtime_batches_aggregated"], 2)
        self.assertEqual(summary["runtime_samples_aggregated"], 3)
        self.assertEqual(summary["selected_restart_histogram"], [1, 2])
        self.assertEqual(summary["selected_restart_fraction"], [1.0 / 3.0, 2.0 / 3.0])
        self.assertAlmostEqual(float(summary["best_mean_score"]), (0.25 * 2.0 + 0.10) / 3.0)
        self.assertEqual(summary["restart_mean_score_by_restart"], [((0.60 * 2.0) + 0.40) / 3.0, ((0.25 * 2.0) + 0.10) / 3.0])

    def test_runner_applies_sparse_backward_mode_temporarily(self) -> None:
        adapter = _build_sparse_toy_adapter()
        runner = AttackRunner(adapter)
        images = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float32)
        targets = torch.zeros((1, 1, 1), dtype=torch.long)
        config = AttackConfig(name="fgsm", epsilon=0.2, attack_backward_mode="bpda_ste")

        self.assertEqual(adapter.model.sparse.attack_backward_mode, "default")

        output = runner.run(config, images, targets)

        self.assertEqual(adapter.model.last_seen_backward_mode, "bpda_ste")
        self.assertEqual(adapter.model.sparse.attack_backward_mode, "default")
        self.assertEqual(output.metadata["attack_backward_mode"], "bpda_ste")
        self.assertEqual(output.metadata["sparse_modules_configured"], 1)

    def test_runner_single_restart_matches_direct_attack(self) -> None:
        adapter = _build_toy_adapter()
        runner = AttackRunner(adapter)
        images = torch.full((1, 1, 1, 1), 0.9, dtype=torch.float32)
        targets = torch.zeros((1, 1, 1), dtype=torch.long)
        config = AttackConfig(name="pgd", epsilon=0.8, step_size=0.2, steps=4, random_start=False, num_restarts=1)

        direct_output = PGDAttack(adapter, config).run(images, targets)
        runner_output = runner.run(config, images, targets)

        self.assertTrue(torch.allclose(runner_output.adversarial_images, direct_output.adversarial_images))
        self.assertTrue(torch.allclose(runner_output.perturbation, direct_output.perturbation))
        self.assertEqual(runner_output.metadata["num_restarts"], 1)
        self.assertFalse(runner_output.metadata["sample_wise_worst_case_over_restarts"])

    def test_evaluation_summary_includes_attack_protocol_fields(self) -> None:
        adapter = _build_toy_adapter()
        images = torch.full((2, 1, 1, 1), 0.9, dtype=torch.float32)
        targets = torch.zeros((2, 1, 1), dtype=torch.long)
        dataloader = DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)
        attack_config = AttackConfig(name="fgsm", epsilon=8.0 / 255.0).with_runtime_overrides(
            epsilon_radius_255=4.0,
            attack_backward_mode="default",
            num_restarts=2,
            eot_iters=3,
        )

        summary = evaluate_adversarial_segmentation_model(
            model=adapter,
            attack_config=attack_config,
            dataloader=dataloader,
        )

        self.assertEqual(summary["attack"]["attack_backward_mode"], "default")
        self.assertEqual(summary["attack"]["num_restarts"], 2)
        self.assertEqual(summary["attack"]["eot_iters"], 3)
        self.assertAlmostEqual(float(summary["attack"]["epsilon_radius_255"]), 4.0)
        self.assertAlmostEqual(float(summary["attack"]["epsilon"]), 4.0 / 255.0)
        self.assertTrue(summary["attack"]["sample_wise_worst_case_over_restarts"])
        self.assertEqual(summary["attack"]["restart_selection"], "per_image_accuracy")
        self.assertEqual(summary["attack"]["runtime_samples_aggregated"], summary["processed_samples"])
        self.assertEqual(sum(summary["attack"]["selected_restart_histogram"]), summary["processed_samples"])
        self.assertEqual(len(summary["attack"]["restart_mean_score_by_restart"]), 2)


if __name__ == "__main__":
    unittest.main()
