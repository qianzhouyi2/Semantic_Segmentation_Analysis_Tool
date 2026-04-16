from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import materialize_sparse_defense_configs
import materialize_voc_attack_suite_manifest
import materialize_voc_transfer_protocol_manifest
import search_sparse_thresholds
from src.common.sparse_workflow import parse_sparse_variants, resolve_sparse_defense_config
from src.common.voc_protocol import VOC_BASE_MODELS


class SparseWorkflowScriptsTest(unittest.TestCase):
    def test_search_parse_args_accepts_postsparse_variant(self) -> None:
        argv = [
            "search_sparse_thresholds.py",
            "--family",
            "upernet_convnext",
            "--checkpoint",
            "models/UperNet_ConvNext_T_VOC_clean.pth",
            "--variant",
            "dir_extra_sparse",
            "--stats-path",
            "models/defenses/dir_extra_sparse_stats.pt",
            "--output-dir",
            "results/reports/sparse_search/demo",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = search_sparse_thresholds.parse_args()
        self.assertEqual(args.variant, "dir_extra_sparse")

    def test_resolve_sparse_defense_config_inherits_postsparse_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            template_path = root / "dir_extra_sparse_template.yaml"
            template_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "dir_extra_sparse",
                        "stats_path": "./placeholder.pt",
                        "threshold": 0.4,
                        "direction_mode": "class_mean",
                        "lambda_mix": 0.8,
                        "alpha0": 0.25,
                        "alpha0_mode": "adaptive",
                        "beta": 0.2,
                        "beta_scale": 1.25,
                        "tau": 2.3,
                        "strict_stats": False,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            stats_override = Path("models/defenses/test_dir_extra_sparse_stats.pt")

            config = resolve_sparse_defense_config(
                variant="dir_extra_sparse",
                family="upernet_convnext",
                threshold=0.15,
                stats_path=stats_override,
                template_config_path=template_path,
            )

            self.assertEqual(config.variant, "dir_extra_sparse")
            self.assertEqual(config.direction_mode, "class_mean")
            self.assertAlmostEqual(config.lambda_mix, 0.8)
            self.assertAlmostEqual(config.alpha0 or 0.0, 0.25)
            self.assertEqual(config.alpha0_mode, "adaptive")
            self.assertAlmostEqual(config.beta, 0.2)
            self.assertAlmostEqual(config.beta_scale or 0.0, 1.25)
            self.assertAlmostEqual(config.tau, 2.3)
            self.assertFalse(config.strict_stats)
            self.assertAlmostEqual(config.threshold, 0.15)
            self.assertEqual(config.stats_path, stats_override.resolve())

    def test_materialize_sparse_defense_configs_preserves_postsparse_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            search_root = root / "search"
            output_dir = root / "configs"
            checkpoint_path = root / "models" / "UperNet_ConvNext_T_VOC_clean.pth"
            stats_path = root / "models" / "dir_extra_sparse_stats.pt"
            template_path = root / "dir_extra_sparse_template.yaml"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text("placeholder", encoding="utf-8")
            template_path.write_text(
                yaml.safe_dump(
                    {
                        "name": "dir_extra_sparse",
                        "direction_mode": "class_mean",
                        "lambda_mix": 0.7,
                        "alpha0_mode": "adaptive",
                        "beta": 0.22,
                        "tau": 1.9,
                        "strict_stats": False,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            self._write_search_summary(
                search_root=search_root,
                checkpoint_name=checkpoint_path.stem,
                family="upernet_convnext",
                checkpoint_path=checkpoint_path,
                variant="dir_extra_sparse",
                stats_path=stats_path,
                threshold=0.18,
                defense_template_config=template_path,
                effective_defense_config={
                    "variant": "dir_extra_sparse",
                    "name": "dir_extra_sparse",
                    "family": "upernet_convnext",
                    "threshold": 0.18,
                    "stats_path": str(stats_path),
                    "strict_stats": False,
                    "direction_mode": "class_mean",
                    "lambda_mix": 0.7,
                    "alpha0_mode": "adaptive",
                    "beta": 0.22,
                    "tau": 1.9,
                },
            )

            argv = [
                "materialize_sparse_defense_configs.py",
                "--search-root",
                str(search_root),
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                materialize_sparse_defense_configs.main()

            config_path = output_dir / f"{checkpoint_path.stem}_dir_extra_sparse.yaml"
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["name"], "dir_extra_sparse")
            self.assertEqual(payload["direction_mode"], "class_mean")
            self.assertAlmostEqual(float(payload["lambda_mix"]), 0.7)
            self.assertEqual(payload["alpha0_mode"], "adaptive")
            self.assertAlmostEqual(float(payload["beta"]), 0.22)
            self.assertAlmostEqual(float(payload["tau"]), 1.9)
            self.assertEqual(payload["strict_stats"], False)

    def test_attack_suite_manifest_supports_new_sparse_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            search_root = root / "search"
            output_dir = root / "attack_manifest"
            self._populate_search_root(search_root, ["dir_extra_sparse"])

            manifest = materialize_voc_attack_suite_manifest.build_manifest(
                search_root,
                output_dir,
                ["dir_extra_sparse"],
            )

            self.assertEqual(manifest["requested_sparse_variants"], ["dir_extra_sparse"])
            self.assertEqual(manifest["num_models"], len(VOC_BASE_MODELS) * 2)
            variant_models = [row for row in manifest["models"] if row["variant"] == "dir_extra_sparse"]
            self.assertEqual(len(variant_models), len(VOC_BASE_MODELS))
            self.assertTrue(all(row["model_id"].startswith("dir_extra_sparse__") for row in variant_models))

    def test_transfer_manifest_supports_new_sparse_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            search_root = root / "search"
            output_dir = root / "transfer_manifest"
            self._populate_search_root(search_root, ["margin_extra_sparse"])

            models, written_configs = materialize_voc_transfer_protocol_manifest.build_models(
                search_root,
                output_dir,
                ["margin_extra_sparse"],
            )
            cases = materialize_voc_transfer_protocol_manifest.build_cases(models, ["margin_extra_sparse"])

            self.assertEqual(len(written_configs), len(VOC_BASE_MODELS))
            self.assertTrue(any(row["variant"] == "margin_extra_sparse" for row in models))
            self.assertTrue(all(len(case["targets"]) == 2 for case in cases))
            self.assertTrue(all(case["targets"][1]["variant"] == "margin_extra_sparse" for case in cases))

    def test_parse_sparse_variants_default_keeps_legacy_subset(self) -> None:
        self.assertEqual(parse_sparse_variants(None), ["meansparse", "extrasparse"])

    def test_attack_suite_manifest_meansparse_extrasparse_keeps_legacy_18_model_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            search_root = root / "search"
            output_dir = root / "attack_manifest"
            self._populate_search_root(search_root, ["meansparse", "extrasparse"])

            manifest = materialize_voc_attack_suite_manifest.build_manifest(
                search_root,
                output_dir,
                ["meansparse", "extrasparse"],
            )

            self.assertEqual(manifest["num_models"], len(VOC_BASE_MODELS) * 3)
            baseline = [row for row in manifest["models"] if row["variant"] == "baseline"]
            meansparse = [row for row in manifest["models"] if row["variant"] == "meansparse"]
            extrasparse = [row for row in manifest["models"] if row["variant"] == "extrasparse"]
            self.assertEqual(len(baseline), len(VOC_BASE_MODELS))
            self.assertEqual(len(meansparse), len(VOC_BASE_MODELS))
            self.assertEqual(len(extrasparse), len(VOC_BASE_MODELS))

    def _populate_search_root(self, search_root: Path, variants: list[str]) -> None:
        for base_model in VOC_BASE_MODELS:
            checkpoint_path = search_root / "checkpoints" / Path(base_model["checkpoint"]).name
            stats_root = search_root / "stats"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text("placeholder", encoding="utf-8")
            for variant in variants:
                stats_path = stats_root / variant / f"{base_model['name']}_{variant}.pt"
                stats_path.parent.mkdir(parents=True, exist_ok=True)
                stats_path.write_text("placeholder", encoding="utf-8")
                effective_defense_config = self._build_effective_defense_config(
                    family=base_model["family"],
                    variant=variant,
                    threshold=0.2,
                    stats_path=stats_path,
                )
                self._write_search_summary(
                    search_root=search_root,
                    checkpoint_name=base_model["name"],
                    family=base_model["family"],
                    checkpoint_path=checkpoint_path,
                    variant=variant,
                    stats_path=stats_path,
                    threshold=0.2,
                    defense_template_config=None,
                    effective_defense_config=effective_defense_config,
                )

    def _build_effective_defense_config(
        self,
        *,
        family: str,
        variant: str,
        threshold: float,
        stats_path: Path,
    ) -> dict:
        if variant in {"meansparse", "extrasparse"}:
            return {
                "variant": variant,
                "name": variant,
                "family": family,
                "threshold": threshold,
                "stats_path": str(stats_path),
                "strict_stats": True,
            }
        return {
            "variant": variant,
            "name": variant,
            "family": family,
            "threshold": threshold,
            "stats_path": str(stats_path),
            "strict_stats": True,
            "direction_mode": "weight_sign",
            "lambda_mix": 0.5,
            "alpha0_mode": "fixed",
            "beta": 0.15,
            "tau": 1.5,
        }

    def _write_search_summary(
        self,
        *,
        search_root: Path,
        checkpoint_name: str,
        family: str,
        checkpoint_path: Path,
        variant: str,
        stats_path: Path,
        threshold: float,
        defense_template_config: Path | None,
        effective_defense_config: dict,
    ) -> None:
        case_dir = search_root / checkpoint_name / variant
        clean_results = case_dir / "thr_0_20_clean" / "results.json"
        adv_results = case_dir / "thr_0_20_pgd" / "results.json"
        clean_results.parent.mkdir(parents=True, exist_ok=True)
        adv_results.parent.mkdir(parents=True, exist_ok=True)
        clean_results.write_text("{}", encoding="utf-8")
        adv_results.write_text("{}", encoding="utf-8")
        summary_payload = {
            "family": family,
            "checkpoint": str(checkpoint_path.resolve()),
            "variant": variant,
            "stats_path": str(stats_path.resolve()),
            "defense_template_config": None if defense_template_config is None else str(defense_template_config.resolve()),
            "effective_defense_config": effective_defense_config,
            "variant_hyperparameters": {
                key: value
                for key, value in effective_defense_config.items()
                if key not in {"variant", "name", "family", "threshold", "stats_path"}
            },
            "dataset": {"root": str((search_root / "datasets").resolve()), "split": "train"},
            "attack_config": str((search_root / "configs" / "attacks" / "pgd.yaml").resolve()),
            "skip_clean": False,
            "thresholds": [
                {
                    "threshold": threshold,
                    "clean_miou": 61.0,
                    "clean_macc": 72.0,
                    "clean_aacc": 80.0,
                    "adv_miou": 29.0,
                    "adv_macc": 40.0,
                    "adv_aacc": 50.0,
                    "clean_results": str(clean_results.resolve()),
                    "adv_results": str(adv_results.resolve()),
                    "effective_defense_config": effective_defense_config,
                    "variant_hyperparameters": {
                        key: value
                        for key, value in effective_defense_config.items()
                        if key not in {"variant", "name", "family", "threshold", "stats_path"}
                    },
                }
            ],
            "pareto_frontier": [],
            "best_threshold": {
                "threshold": threshold,
                "clean_miou": 61.0,
                "clean_macc": 72.0,
                "clean_aacc": 80.0,
                "adv_miou": 29.0,
                "adv_macc": 40.0,
                "adv_aacc": 50.0,
                "clean_results": str(clean_results.resolve()),
                "adv_results": str(adv_results.resolve()),
                "effective_defense_config": effective_defense_config,
                "variant_hyperparameters": {
                    key: value
                    for key, value in effective_defense_config.items()
                    if key not in {"variant", "name", "family", "threshold", "stats_path"}
                },
            },
            "selection_rule": "test",
        }
        (case_dir / "search_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
