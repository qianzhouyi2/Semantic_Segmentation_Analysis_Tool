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

import summarize_voc_all_attacks


class VocAllAttacksSummaryTest(unittest.TestCase):
    def test_summary_adds_imagewise_worstcase_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            suite_root = root / "suite"
            config_dir = root / "attacks"
            manifest_path = root / "attack_suite_manifest.json"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "fgsm.yaml").write_text("name: fgsm\n", encoding="utf-8")
            (config_dir / "pgd.yaml").write_text("name: pgd\n", encoding="utf-8")

            manifest = {
                "models": [
                    {
                        "model_id": "baseline__demo_model",
                        "display_name": "demo_model",
                        "family": "upernet_convnext",
                        "variant": "baseline",
                        "threshold": None,
                    }
                ]
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            self._write_summary(
                suite_root / "clean" / "baseline__demo_model",
                miou_percent=80.0,
            )
            self._write_summary(
                suite_root / "attacks" / "fgsm" / "baseline__demo_model",
                miou_percent=50.0,
                per_sample_rows=[
                    {"sample_index": 0, "filename": "img_001.png", "pixel_accuracy": 0.70, "sample_miou": 0.60, "sample_dice": 0.72, "valid_class_count": 2},
                    {"sample_index": 1, "filename": "img_002.png", "pixel_accuracy": 0.60, "sample_miou": 0.40, "sample_dice": 0.55, "valid_class_count": 2},
                    {"sample_index": 2, "filename": "img_003.png", "pixel_accuracy": 0.65, "sample_miou": 0.50, "sample_dice": 0.60, "valid_class_count": 2},
                ],
            )
            self._write_summary(
                suite_root / "attacks" / "pgd" / "baseline__demo_model",
                miou_percent=45.0,
                per_sample_rows=[
                    {"sample_index": 0, "filename": "img_001.png", "pixel_accuracy": 0.55, "sample_miou": 0.30, "sample_dice": 0.42, "valid_class_count": 2},
                    {"sample_index": 1, "filename": "img_002.png", "pixel_accuracy": 0.58, "sample_miou": 0.45, "sample_dice": 0.50, "valid_class_count": 2},
                    {"sample_index": 2, "filename": "img_003.png", "pixel_accuracy": 0.66, "sample_miou": 0.55, "sample_dice": 0.64, "valid_class_count": 2},
                ],
            )

            argv = [
                "summarize_voc_all_attacks.py",
                "--manifest",
                str(manifest_path),
                "--suite-root",
                str(suite_root),
                "--attack-config-dir",
                str(config_dir),
                "--per-sample-policy",
                "require",
            ]
            with mock.patch.object(sys, "argv", argv):
                summarize_voc_all_attacks.main()

            summary_dir = suite_root / "summary"
            payload = json.loads((summary_dir / "all_attacks_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["rows_with_computed_worstcase"], 1)
            row = payload["rows"][0]
            self.assertAlmostEqual(float(row["mean_attack_miou"]), 47.5)
            self.assertAlmostEqual(float(row["worst_imagewise_attack_miou"]), 40.0)
            self.assertEqual(row["worst_attack_stem_by_frequency"], "fgsm")
            self.assertEqual(int(row["num_attacks_counted"]), 2)
            self.assertEqual(int(row["num_attacks_with_per_sample"]), 2)
            self.assertEqual(row["worstcase_status"], "computed")

            paper_payload = json.loads((summary_dir / "all_attacks_worstcase_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(paper_payload["rows"]), 1)
            self.assertIn("worst_imagewise_attack_miou", paper_payload["rows"][0])

            worstcase_md = (summary_dir / "all_attacks_worstcase_summary.md").read_text(encoding="utf-8")
            self.assertIn("imagewise_worst", worstcase_md)

    def _write_summary(
        self,
        output_dir: Path,
        *,
        miou_percent: float,
        per_sample_rows: list[dict[str, object]] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "reference_percent": {
                "mIoU": miou_percent,
                "mAcc": miou_percent,
                "aAcc": miou_percent,
            },
            "artifacts": {
                "per_sample_metrics_csv": None,
            },
        }
        if per_sample_rows is not None:
            csv_path = output_dir / "per_sample_metrics.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "sample_index,filename,pixel_accuracy,sample_miou,sample_dice,valid_class_count",
                        *[
                            (
                                f"{row['sample_index']},{row['filename']},{row['pixel_accuracy']},"
                                f"{row['sample_miou']},{row['sample_dice']},{row['valid_class_count']}"
                            )
                            for row in per_sample_rows
                        ],
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            payload["artifacts"]["per_sample_metrics_csv"] = str(csv_path.resolve())
        (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
