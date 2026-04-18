from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logger import setup_logger


class SetupLoggerTest(unittest.TestCase):
    def test_setup_logger_reconfigures_existing_logger_to_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            first_log = temp_root / "first.log"
            second_log = temp_root / "second.log"

            logger = setup_logger("tests.logger.reconfigure", first_log)
            logger.info("first-message")

            logger = setup_logger("tests.logger.reconfigure", second_log)
            logger.info("second-message")

            for handler in logger.handlers:
                handler.flush()

            first_text = first_log.read_text(encoding="utf-8")
            second_text = second_log.read_text(encoding="utf-8")

            self.assertIn("first-message", first_text)
            self.assertNotIn("second-message", first_text)
            self.assertIn("second-message", second_text)


if __name__ == "__main__":
    unittest.main()
