from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "pandas" not in sys.modules:
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = object
    fake_pandas.__spec__ = ModuleSpec("pandas", loader=None)
    sys.modules["pandas"] = fake_pandas

if "streamlit" not in sys.modules:
    fake_streamlit = types.ModuleType("streamlit")

    def _cache_resource(*_args, **_kwargs):
        def decorator(function):
            return function

        return decorator

    fake_streamlit.cache_resource = _cache_resource
    fake_streamlit.__spec__ = ModuleSpec("streamlit", loader=None)
    sys.modules["streamlit"] = fake_streamlit

import app
from src.common.sample_manifest import filter_voc_sample_ids, load_voc_sample_id_manifest, normalize_voc_sample_id
from scripts.find_rescued_voc_samples import find_rescued_rows, resolve_output_json


class SampleManifestTest(unittest.TestCase):
    def test_normalize_voc_sample_id_strips_suffix(self) -> None:
        self.assertEqual(normalize_voc_sample_id("2007_000001.jpg"), "2007_000001")
        self.assertEqual(normalize_voc_sample_id("2007_000001.png"), "2007_000001")

    def test_load_voc_sample_id_manifest_accepts_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "samples.json"
            manifest_path.write_text(
                json.dumps({"name": "demo", "sample_ids": ["2007_000001.jpg", "2007_000002"]}),
                encoding="utf-8",
            )
            manifest = load_voc_sample_id_manifest(manifest_path)
        self.assertEqual(manifest["sample_ids"], ["2007_000001", "2007_000002"])

    def test_filter_voc_sample_ids_preserves_dataset_order(self) -> None:
        filtered = filter_voc_sample_ids(["2007_000003", "2007_000001", "2007_000002"], ["2007_000002", "2007_000003"])
        self.assertEqual(filtered, ["2007_000003", "2007_000002"])

    def test_load_optional_voc_sample_manifest_returns_none_for_empty_input(self) -> None:
        self.assertIsNone(app._load_optional_voc_sample_manifest(""))

    def test_find_rescued_rows_uses_delta_threshold(self) -> None:
        baseline = {
            "2007_000001": {"filename": "2007_000001.jpg", "sample_miou": 0.20, "pixel_accuracy": 0.3, "sample_dice": 0.25},
            "2007_000002": {"filename": "2007_000002.jpg", "sample_miou": 0.60, "pixel_accuracy": 0.7, "sample_dice": 0.65},
        }
        defended = {
            "2007_000001": {"filename": "2007_000001.jpg", "sample_miou": 0.31, "pixel_accuracy": 0.5, "sample_dice": 0.35},
            "2007_000002": {"filename": "2007_000002.jpg", "sample_miou": 0.62, "pixel_accuracy": 0.72, "sample_dice": 0.66},
        }
        rescued, improved = find_rescued_rows(baseline, defended, metric="sample_miou", min_delta=0.05)
        self.assertEqual([row["sample_id"] for row in rescued], ["2007_000001"])
        self.assertEqual([row["sample_id"] for row in improved], ["2007_000001", "2007_000002"])

    def test_resolve_output_json_includes_attack_stem_when_present(self) -> None:
        args = SimpleNamespace(
            output_json="",
            checkpoint="models/UperNet_ConvNext_T_VOC_clean.pth",
            defense_config="configs/defenses/UperNet_ConvNext_T_VOC_clean_extrasparse.yaml",
            attack_config="configs/attacks/pgd.yaml",
        )
        output_path = resolve_output_json(args)
        self.assertEqual(
            output_path,
            Path("samples/UperNet_ConvNext_T_VOC_clean_extrasparse_pgd_rescued_samples.json"),
        )


if __name__ == "__main__":
    unittest.main()
