from __future__ import annotations

import unittest

from scripts.launch_voc_attack_suite import append_attack_protocol_exports as append_suite_exports
from scripts.launch_voc_transfer_protocol import append_attack_protocol_exports as append_transfer_exports


class LaunchProtocolScriptTest(unittest.TestCase):
    def test_suite_exports_include_protocol_overrides(self) -> None:
        export_items = append_suite_exports(
            ["ALL", "MODE=attack"],
            epsilon_scale=1.5,
            epsilon_radius_255=4.0,
            attack_backward_mode="bpda_ste",
            num_restarts=5,
            eot_iters=4,
        )

        self.assertIn("EPSILON_SCALE=1.5", export_items)
        self.assertIn("EPSILON_RADIUS_255=4.0", export_items)
        self.assertIn("ATTACK_BACKWARD_MODE=bpda_ste", export_items)
        self.assertIn("NUM_RESTARTS=5", export_items)
        self.assertIn("EOT_ITERS=4", export_items)

    def test_transfer_exports_skip_absolute_budget_when_unset(self) -> None:
        export_items = append_transfer_exports(
            ["CASE_JSON=/tmp/case.json"],
            epsilon_scale=1.0,
            epsilon_radius_255=None,
            attack_backward_mode="default",
            num_restarts=1,
            eot_iters=1,
        )

        self.assertIn("EPSILON_SCALE=1.0", export_items)
        self.assertNotIn("EPSILON_RADIUS_255=", export_items)
        self.assertIn("ATTACK_BACKWARD_MODE=default", export_items)
        self.assertIn("NUM_RESTARTS=1", export_items)
        self.assertIn("EOT_ITERS=1", export_items)


if __name__ == "__main__":
    unittest.main()
