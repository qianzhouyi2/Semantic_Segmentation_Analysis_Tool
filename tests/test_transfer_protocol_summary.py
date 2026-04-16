from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import summarize_voc_transfer_protocol


class TransferProtocolSummaryTest(unittest.TestCase):
    def test_summary_generates_paper_friendly_tables_and_worstcase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            suite_root = root / "suite"
            manifest_path = root / "transfer_protocol_manifest.json"
            manifest = {
                "requested_sparse_variants": ["meansparse"],
                "requested_attack_stems": ["mi_fgsm", "transegpgd"],
                "transfer_attacks": [
                    {"stem": "mi_fgsm", "name": "mi-fgsm", "config": "configs/attacks/mi_fgsm.yaml"},
                    {"stem": "transegpgd", "name": "transegpgd", "config": "configs/attacks/transegpgd.yaml"},
                ],
                "num_cases": 3,
                "cases": [
                    self._build_case(
                        attack_stem="mi_fgsm",
                        attack_name="mi-fgsm",
                        source_model_id="baseline__UperNet_ConvNext_T_VOC_clean",
                        source_display_name="UperNet_ConvNext_T_VOC_clean",
                        source_family="upernet_convnext",
                        relation="cross_family",
                    ),
                    self._build_case(
                        attack_stem="transegpgd",
                        attack_name="transegpgd",
                        source_model_id="baseline__UperNet_ConvNext_T_VOC_clean",
                        source_display_name="UperNet_ConvNext_T_VOC_clean",
                        source_family="upernet_convnext",
                        relation="cross_family",
                    ),
                    self._build_case(
                        attack_stem="mi_fgsm",
                        attack_name="mi-fgsm",
                        source_model_id="baseline__Segmenter_ViT_S_VOC_clean",
                        source_display_name="Segmenter_ViT_S_VOC_clean",
                        source_family="segmenter_vit_s",
                        relation="same_family",
                    ),
                ],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            self._write_case_summary(
                suite_root=suite_root,
                case=manifest["cases"][0],
                source_transfer_miou=35.0,
                source_transfer_drop=40.0,
                baseline_target_transfer_miou=25.0,
                baseline_target_transfer_drop=45.0,
                meansparse_target_transfer_miou=30.0,
                meansparse_target_transfer_drop=41.0,
            )
            self._write_case_summary(
                suite_root=suite_root,
                case=manifest["cases"][1],
                source_transfer_miou=30.0,
                source_transfer_drop=45.0,
                baseline_target_transfer_miou=20.0,
                baseline_target_transfer_drop=50.0,
                meansparse_target_transfer_miou=27.0,
                meansparse_target_transfer_drop=44.0,
            )
            self._write_case_summary(
                suite_root=suite_root,
                case=manifest["cases"][2],
                source_transfer_miou=40.0,
                source_transfer_drop=36.0,
                baseline_target_transfer_miou=28.0,
                baseline_target_transfer_drop=42.0,
                meansparse_target_transfer_miou=34.0,
                meansparse_target_transfer_drop=37.0,
            )

            argv = [
                "summarize_voc_transfer_protocol.py",
                "--manifest",
                str(manifest_path),
                "--suite-root",
                str(suite_root),
            ]
            with mock.patch.object(sys, "argv", argv):
                summarize_voc_transfer_protocol.main()

            summary_dir = suite_root / "summary"
            summary_payload = json.loads((summary_dir / "transfer_protocol_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["requested_attack_stems"], ["mi_fgsm", "transegpgd"])
            self.assertEqual(summary_payload["num_cases"], 3)
            self.assertFalse(summary_payload["missing_case_ids"])

            gain_rows = summary_payload["tables"]["gain_over_baseline"]
            transeg_gain_row = next(
                row
                for row in gain_rows
                if row["attack_stem"] == "transegpgd"
                and row["target_model_id"] == "meansparse__Segmenter_ViT_S_VOC_clean"
            )
            self.assertAlmostEqual(float(transeg_gain_row["gain_over_baseline_miou"]), 7.0)
            self.assertAlmostEqual(float(transeg_gain_row["drop_reduction_vs_baseline"]), 6.0)

            same_family_rows = summary_payload["tables"]["fixed_target_same_family_comparison"]
            self.assertTrue(
                any(
                    row["source_model_id"] == "baseline__Segmenter_ViT_S_VOC_clean"
                    and row["target_model_id"] == "meansparse__Segmenter_ViT_S_VOC_clean"
                    for row in same_family_rows
                )
            )

            family_matrix_rows = summary_payload["tables"]["source_target_family_matrix_on_baseline_targets"]
            self.assertTrue(
                any(
                    row["attack_stem"] == "transegpgd"
                    and row["source_family"] == "upernet_convnext"
                    and row["target_family"] == "segmenter_vit_s"
                    and float(row["baseline_target_transfer_miou"]) == 20.0
                    for row in family_matrix_rows
                )
            )

            worstcase_payload = json.loads((summary_dir / "transfer_protocol_worstcase.json").read_text(encoding="utf-8"))
            worst_meansparse = next(
                row
                for row in worstcase_payload["per_target_worst_overall"]
                if row["target_model_id"] == "meansparse__Segmenter_ViT_S_VOC_clean"
            )
            self.assertEqual(worst_meansparse["worst_attack_stem"], "transegpgd")
            self.assertEqual(worst_meansparse["worst_source_model_id"], "baseline__UperNet_ConvNext_T_VOC_clean")
            self.assertAlmostEqual(float(worst_meansparse["worst_transfer_miou"]), 27.0)

            tables_md = (summary_dir / "transfer_protocol_tables.md").read_text(encoding="utf-8")
            self.assertIn("## Source Self Transfer Strength", tables_md)
            self.assertIn("## Worst-Case Target Summary", tables_md)

            targets_csv = (summary_dir / "transfer_protocol_targets.csv").read_text(encoding="utf-8")
            self.assertIn("gain_over_baseline_miou", targets_csv)

    def _build_case(
        self,
        *,
        attack_stem: str,
        attack_name: str,
        source_model_id: str,
        source_display_name: str,
        source_family: str,
        relation: str,
    ) -> dict:
        target_display_name = "Segmenter_ViT_S_VOC_clean"
        return {
            "case_id": f"{attack_stem}__{source_model_id}__to__{target_display_name}",
            "attack_stem": attack_stem,
            "attack_name": attack_name,
            "attack_config": f"configs/attacks/{attack_stem}.yaml",
            "regime": "clean",
            "relation": relation,
            "source": {
                "model_id": source_model_id,
                "display_name": source_display_name,
                "family": source_family,
                "regime": "clean",
                "variant": "baseline",
                "checkpoint": f"/tmp/{source_display_name}.pth",
                "defense_config": None,
                "threshold": None,
            },
            "targets": [
                {
                    "model_id": "baseline__Segmenter_ViT_S_VOC_clean",
                    "display_name": target_display_name,
                    "family": "segmenter_vit_s",
                    "regime": "clean",
                    "variant": "baseline",
                    "checkpoint": "/tmp/Segmenter_ViT_S_VOC_clean.pth",
                    "defense_config": None,
                    "threshold": None,
                },
                {
                    "model_id": "meansparse__Segmenter_ViT_S_VOC_clean",
                    "display_name": target_display_name,
                    "family": "segmenter_vit_s",
                    "regime": "clean",
                    "variant": "meansparse",
                    "checkpoint": "/tmp/Segmenter_ViT_S_VOC_clean.pth",
                    "defense_config": "/tmp/meansparse_segmenter.yaml",
                    "threshold": 0.2,
                },
            ],
        }

    def _write_case_summary(
        self,
        *,
        suite_root: Path,
        case: dict,
        source_transfer_miou: float,
        source_transfer_drop: float,
        baseline_target_transfer_miou: float,
        baseline_target_transfer_drop: float,
        meansparse_target_transfer_miou: float,
        meansparse_target_transfer_drop: float,
    ) -> None:
        case_dir = (
            suite_root
            / "cases"
            / case["attack_stem"]
            / case["source"]["model_id"]
            / case["targets"][0]["display_name"]
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": case["case_id"],
            "regime": case["regime"],
            "relation": case["relation"],
            "attack": {
                "stem": case["attack_stem"],
                "name": case["attack_name"],
            },
            "source_self": {
                "model": {
                    "model_id": case["source"]["model_id"],
                    "family": case["source"]["family"],
                    "checkpoint": case["source"]["checkpoint"],
                    "defense_config": case["source"]["defense_config"],
                    "sparse_defense": None,
                },
                "clean": {"reference_percent": {"mIoU": 75.0}},
                "transfer": {"reference_percent": {"mIoU": source_transfer_miou}},
                "transfer_miou_drop": source_transfer_drop,
            },
            "targets": [
                {
                    "model": {
                        "model_id": "baseline__Segmenter_ViT_S_VOC_clean",
                        "family": "segmenter_vit_s",
                        "checkpoint": "/tmp/Segmenter_ViT_S_VOC_clean.pth",
                        "defense_config": None,
                        "sparse_defense": None,
                    },
                    "clean": {"reference_percent": {"mIoU": 70.0}},
                    "transfer": {"reference_percent": {"mIoU": baseline_target_transfer_miou}},
                    "transfer_miou_drop": baseline_target_transfer_drop,
                },
                {
                    "model": {
                        "model_id": "meansparse__Segmenter_ViT_S_VOC_clean",
                        "family": "segmenter_vit_s",
                        "checkpoint": "/tmp/Segmenter_ViT_S_VOC_clean.pth",
                        "defense_config": "/tmp/meansparse_segmenter.yaml",
                        "sparse_defense": {
                            "variant": "meansparse",
                            "threshold": 0.2,
                        },
                    },
                    "clean": {"reference_percent": {"mIoU": 71.0}},
                    "transfer": {"reference_percent": {"mIoU": meansparse_target_transfer_miou}},
                    "transfer_miou_drop": meansparse_target_transfer_drop,
                },
            ],
        }
        (case_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
