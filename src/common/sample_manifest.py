from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_voc_sample_id(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Sample id cannot be empty.")
    return Path(text).stem


def load_voc_sample_id_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        sample_ids = [normalize_voc_sample_id(str(item)) for item in payload]
        return {
            "type": "voc_sample_id_list",
            "name": manifest_path.stem,
            "sample_ids": sample_ids,
        }
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported sample manifest payload type: {type(payload)!r}")
    raw_ids = payload.get("sample_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("Sample manifest must define a `sample_ids` list.")
    manifest = dict(payload)
    manifest["sample_ids"] = [normalize_voc_sample_id(str(item)) for item in raw_ids]
    return manifest


def filter_voc_sample_ids(sample_ids: list[str], allowed_ids: list[str]) -> list[str]:
    allowed = set(normalize_voc_sample_id(item) for item in allowed_ids)
    return [sample_id for sample_id in sample_ids if normalize_voc_sample_id(sample_id) in allowed]
