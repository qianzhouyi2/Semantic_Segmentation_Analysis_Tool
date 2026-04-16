from __future__ import annotations

import sys
import tempfile
import types
import unittest
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
