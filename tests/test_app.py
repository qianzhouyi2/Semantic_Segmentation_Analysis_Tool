from __future__ import annotations

import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


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


class AppConfigResolutionTest(unittest.TestCase):
    def test_resolve_pascal_voc_label_config_path_prefers_repo_voc_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            default_label_path = root / "labels" / "custom.yaml"
            voc_label_path = root / "configs" / "labels" / "pascal_voc.yaml"
            default_label_path.parent.mkdir(parents=True, exist_ok=True)
            voc_label_path.parent.mkdir(parents=True, exist_ok=True)
            default_label_path.write_text("classes: []\n", encoding="utf-8")
            voc_label_path.write_text("classes: []\n", encoding="utf-8")

            with patch.object(app, "_default_pascal_voc_config_paths", return_value=(root / "configs" / "datasets" / "pascal_voc.yaml", voc_label_path)):
                resolved = app._resolve_pascal_voc_label_config_path(str(default_label_path))

        self.assertEqual(resolved, str(voc_label_path))

    def test_resolve_pascal_voc_label_config_path_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            default_label_path = root / "labels" / "custom.yaml"
            default_label_path.parent.mkdir(parents=True, exist_ok=True)
            default_label_path.write_text("classes: []\n", encoding="utf-8")
            missing_voc_label_path = root / "configs" / "labels" / "pascal_voc.yaml"

            with patch.object(app, "_default_pascal_voc_config_paths", return_value=(root / "configs" / "datasets" / "pascal_voc.yaml", missing_voc_label_path)):
                resolved = app._resolve_pascal_voc_label_config_path(str(default_label_path))

        self.assertEqual(resolved, str(default_label_path))

    def test_parse_sweep_radii_returns_sorted_unique_values(self) -> None:
        self.assertEqual(app._parse_sweep_radii("4, 0, 2, 4"), [0, 2, 4])

    def test_parse_sweep_radii_rejects_out_of_range_value(self) -> None:
        with self.assertRaises(ValueError):
            app._parse_sweep_radii("1, 512")

    def test_discover_target_class_ids_defaults_to_present_classes(self) -> None:
        preview_result = SimpleNamespace(
            ground_truth=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            clean_prediction=np.asarray([[0, 1], [1, 1]], dtype=np.int64),
            adversarial_prediction=np.asarray([[0, 2], [2, 2]], dtype=np.int64),
        )

        with patch.object(app, "_optional_label_config", return_value=None):
            class_ids, class_names, background_ids, present_class_ids = app._discover_target_class_ids(
                preview_result,
                "configs/labels/example.yaml",
                show_all_classes=False,
            )

        self.assertEqual(class_ids, [0, 1, 2])
        self.assertEqual(class_names, {})
        self.assertEqual(background_ids, (0,))
        self.assertEqual(present_class_ids, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
