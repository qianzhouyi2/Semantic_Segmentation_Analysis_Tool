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

import compare_per_sample_robustness


class ComparePerSampleRobustnessTest(unittest.TestCase):
    def test_compare_script_outputs_paired_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline_csv = root / "baseline.csv"
            candidate_csv = root / "candidate.csv"
            output_dir = root / "comparison"

            baseline_csv.write_text(
                "\n".join(
                    [
                        "sample_index,filename,pixel_accuracy,sample_miou,sample_dice,valid_class_count",
                        "0,img_001.png,0.80,0.30,0.40,2",
                        "1,img_002.png,0.82,0.40,0.50,2",
                        "2,img_003.png,0.84,0.50,0.60,2",
                        "3,img_004.png,0.86,0.60,0.70,2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            candidate_csv.write_text(
                "\n".join(
                    [
                        "sample_index,filename,pixel_accuracy,sample_miou,sample_dice,valid_class_count",
                        "0,img_001.png,0.83,0.35,0.45,2",
                        "1,img_002.png,0.80,0.38,0.48,2",
                        "2,img_003.png,0.85,0.52,0.62,2",
                        "3,img_004.png,0.86,0.60,0.70,2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                "compare_per_sample_robustness.py",
                "--baseline",
                str(baseline_csv),
                "--candidate",
                str(candidate_csv),
                "--output-dir",
                str(output_dir),
                "--metrics",
                "sample_miou pixel_accuracy",
                "--bootstrap-samples",
                "200",
                "--seed",
                "7",
            ]
            with mock.patch.object(sys, "argv", argv):
                compare_per_sample_robustness.main()

            payload = json.loads((output_dir / "paired_robustness_stats.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["n_paired_samples"], 4)
            self.assertEqual(payload["selected_metrics"], ["sample_miou", "pixel_accuracy"])

            sample_miou_row = next(row for row in payload["rows"] if row["metric"] == "sample_miou")
            self.assertAlmostEqual(float(sample_miou_row["mean_delta"]), 0.0125)
            self.assertAlmostEqual(float(sample_miou_row["median_delta"]), 0.01)
            self.assertEqual(int(sample_miou_row["improved_count"]), 2)
            self.assertEqual(int(sample_miou_row["worsened_count"]), 1)
            self.assertEqual(int(sample_miou_row["tied_count"]), 1)
            self.assertIsNotNone(sample_miou_row["bootstrap_ci_lower"])
            self.assertIsNotNone(sample_miou_row["sign_test_pvalue"])

            markdown = (output_dir / "paired_robustness_stats.md").read_text(encoding="utf-8")
            self.assertIn("sample_miou", markdown)
            self.assertIn("pixel_accuracy", markdown)

            deltas_csv = (output_dir / "paired_sample_deltas.csv").read_text(encoding="utf-8")
            self.assertIn("baseline_value", deltas_csv)
            self.assertIn("candidate_value", deltas_csv)


if __name__ == "__main__":
    unittest.main()
