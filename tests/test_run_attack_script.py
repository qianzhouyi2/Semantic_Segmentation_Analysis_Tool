from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_attack import build_attack_run_specs, format_radius_label, resolve_output_dir
from src.attacks import AttackConfig


def _build_args(**overrides) -> argparse.Namespace:
    payload = {
        "attack_config": "configs/attacks/pgd.yaml",
        "family": "upernet_convnext",
        "checkpoint": "models/example.pth",
        "defense_config": "",
        "dataset_root": "datasets",
        "output_dir": "",
        "batch_size": 4,
        "num_workers": 4,
        "device": "cpu",
        "num_classes": 21,
        "max_batches": -1,
        "epsilon_scale": 1.0,
        "epsilon_radius_255": None,
        "epsilon_radius_255_sweep": None,
        "attack_backward_mode": "default",
        "num_restarts": 1,
        "eot_iters": 1,
        "feature_vis_samples": 0,
        "feature_vis_layers": -1,
        "feature_vis_dir": "",
        "strict": True,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


class RunAttackScriptHelpersTest(unittest.TestCase):
    def test_format_radius_label_normalizes_fractional_values(self) -> None:
        self.assertEqual(format_radius_label(2), "eps_2_255")
        self.assertEqual(format_radius_label(2.5), "eps_2p5_255")

    def test_build_attack_run_specs_returns_single_run_by_default(self) -> None:
        base_config = AttackConfig(name="pgd", epsilon=8.0 / 255.0, step_size=2.0 / 255.0, steps=4)
        args = _build_args(epsilon_radius_255=4.0, attack_backward_mode="bpda_ste", num_restarts=3, eot_iters=2)

        run_specs = build_attack_run_specs(base_config, args)

        self.assertEqual(len(run_specs), 1)
        run_label, config = run_specs[0]
        self.assertIsNone(run_label)
        self.assertAlmostEqual(config.epsilon, 4.0 / 255.0)
        self.assertAlmostEqual(config.step_size or 0.0, 1.0 / 255.0)
        self.assertEqual(config.attack_backward_mode, "bpda_ste")
        self.assertEqual(config.num_restarts, 3)
        self.assertEqual(config.eot_iters, 2)

    def test_build_attack_run_specs_expands_budget_sweep(self) -> None:
        base_config = AttackConfig(name="pgd", epsilon=8.0 / 255.0, step_size=2.0 / 255.0, steps=4)
        args = _build_args(epsilon_radius_255_sweep=[2.0, 4.0, 8.0], num_restarts=5)

        run_specs = build_attack_run_specs(base_config, args)

        self.assertEqual([label for label, _ in run_specs], ["eps_2_255", "eps_4_255", "eps_8_255"])
        self.assertEqual([config.epsilon_radius_255() for _, config in run_specs], [2.0, 4.0, 8.0])
        self.assertTrue(all(config.num_restarts == 5 for _, config in run_specs))

    def test_resolve_output_dir_uses_budget_sweep_suffix(self) -> None:
        args = _build_args(checkpoint="models/demo.pth", epsilon_radius_255_sweep=[2.0, 4.0])

        output_dir = resolve_output_dir(args, "pgd")

        self.assertEqual(output_dir, Path("results/reports/voc_adv_eval/demo_pgd_budget_sweep"))


if __name__ == "__main__":
    unittest.main()
