from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CORE_SCRIPT_MODULES = (
    "scripts.evaluate_voc_clean",
    "scripts.launch_voc_attack_suite",
    "scripts.launch_voc_transfer_protocol",
    "scripts.run_attack",
    "scripts.run_transfer_attack",
    "scripts.run_transfer_group_attack",
    "scripts.search_sparse_thresholds",
    "scripts.prepare_sparse_defense",
)


class ScriptImportTest(unittest.TestCase):
    def test_core_script_modules_are_importable(self) -> None:
        for module_name in CORE_SCRIPT_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
