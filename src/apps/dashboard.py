from __future__ import annotations

from src.datasets.scanner import DatasetScanResult
from src.datasets.stats import DatasetStatistics


def build_overview_cards(scan_result: DatasetScanResult, dataset_stats: DatasetStatistics) -> list[tuple[str, str]]:
    return [
        ("Matched pairs", str(len(scan_result.matched_pairs))),
        ("Missing masks", str(len(scan_result.missing_masks))),
        ("Orphan masks", str(len(scan_result.orphan_masks))),
        ("Size mismatches", str(len(scan_result.mismatched_shapes))),
        ("Empty masks", str(len(scan_result.empty_masks))),
        ("Labeled pixels", str(dataset_stats.total_labeled_pixels)),
    ]
