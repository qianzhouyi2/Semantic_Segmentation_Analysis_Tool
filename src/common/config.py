from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_suffixes(values: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if not values:
        return ()
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        suffix = value if value.startswith(".") else f".{value}"
        normalized.append(suffix.lower())
    return tuple(normalized)


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {yaml_path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@dataclass(slots=True)
class DatasetConfig:
    name: str
    image_dir: Path
    mask_dir: Path
    image_suffixes: tuple[str, ...]
    mask_suffixes: tuple[str, ...]


@dataclass(slots=True)
class LabelConfig:
    class_names: dict[int, str]
    palette: dict[int, tuple[int, int, int]]
    class_ids: tuple[int, ...]
    ignore_index: int | None
    background_ids: tuple[int, ...]


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = load_yaml(path)
    return DatasetConfig(
        name=str(raw.get("name", Path(path).stem)),
        image_dir=resolve_project_path(raw.get("image_dir", "datasets/images")),
        mask_dir=resolve_project_path(raw.get("mask_dir", "datasets/masks")),
        image_suffixes=_normalize_suffixes(raw.get("image_suffixes", [".jpg", ".jpeg", ".png"])),
        mask_suffixes=_normalize_suffixes(raw.get("mask_suffixes", [".png"])),
    )


def load_label_config(path: str | Path) -> LabelConfig:
    raw = load_yaml(path)
    classes = raw.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError("`classes` must be a list of mappings.")

    class_names: dict[int, str] = {}
    palette: dict[int, tuple[int, int, int]] = {}
    class_ids: list[int] = []
    for entry in classes:
        if not isinstance(entry, dict):
            raise ValueError("Each class entry must be a mapping.")
        class_id = int(entry["id"])
        class_ids.append(class_id)
        class_names[class_id] = str(entry.get("name", f"class_{class_id}"))
        color = entry.get("color", [0, 0, 0])
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError("Each class color must be a 3-element list.")
        palette[class_id] = (int(color[0]), int(color[1]), int(color[2]))

    background_ids = tuple(int(item) for item in raw.get("background_ids", [0]))
    ignore_index = raw.get("ignore_index")
    return LabelConfig(
        class_names=class_names,
        palette=palette,
        class_ids=tuple(class_ids),
        ignore_index=int(ignore_index) if ignore_index is not None else None,
        background_ids=background_ids,
    )
