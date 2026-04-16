from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from src.common.config import load_yaml
from src.models.sparse import POSTSPARSE_VARIANTS, SPARSE_DEFENSE_CHOICES, SparseDefenseConfig


DEFAULT_VOC_SPARSE_PROTOCOL_VARIANTS = SPARSE_DEFENSE_CHOICES[:2]
POSTSPARSE_CONFIG_KEYS = (
    "direction_mode",
    "lambda_mix",
    "alpha0",
    "alpha0_mode",
    "beta",
    "beta_scale",
    "tau",
)


def relativize_path(target: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(target, start=base_dir)
    except ValueError:
        return str(target)


def parse_sparse_variants(
    raw: str | Iterable[str] | None,
    *,
    default: Iterable[str] = DEFAULT_VOC_SPARSE_PROTOCOL_VARIANTS,
) -> list[str]:
    if raw is None:
        tokens = list(default)
    elif isinstance(raw, str):
        tokens = [item.strip() for chunk in raw.split(",") for item in chunk.split() if item.strip()]
    else:
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    if not tokens:
        tokens = list(default)

    variants: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        expanded = SPARSE_DEFENSE_CHOICES if token == "all" else (token,)
        for variant in expanded:
            if variant not in SPARSE_DEFENSE_CHOICES:
                raise ValueError(
                    f"Unknown sparse defense variant: {variant}. Expected one of {SPARSE_DEFENSE_CHOICES} or `all`."
                )
            if variant in seen:
                continue
            variants.append(variant)
            seen.add(variant)
    return variants


def resolve_sparse_defense_config(
    *,
    variant: str,
    family: str | None = None,
    threshold: float | None = None,
    stats_path: str | Path | None = None,
    strict_stats: bool | None = None,
    template_config_path: str | Path | None = None,
    template_payload: dict[str, Any] | None = None,
    template_base_dir: str | Path | None = None,
) -> SparseDefenseConfig:
    merged: dict[str, Any] = {}
    base_dir: Path | None = Path(template_base_dir) if template_base_dir is not None else None

    if template_config_path:
        template_path = Path(template_config_path)
        if not template_path.is_absolute() and base_dir is not None:
            template_path = base_dir / template_path
        merged.update(load_yaml(template_path))
        base_dir = template_path.parent

    if template_payload:
        merged.update(template_payload)

    template_variant = merged.get("variant", merged.get("name"))
    if template_variant and str(template_variant) != variant:
        raise ValueError(
            f"Sparse defense template variant mismatch: expected `{variant}`, found `{template_variant}`."
        )

    merged["variant"] = variant
    if family is not None:
        merged["family"] = family
    if threshold is not None:
        merged["threshold"] = threshold
    if stats_path is not None:
        merged["stats_path"] = str(Path(stats_path).resolve())
    if strict_stats is not None:
        merged["strict_stats"] = strict_stats
    return SparseDefenseConfig.from_dict(merged, base_dir=base_dir)


def serialize_sparse_defense_config(
    config: SparseDefenseConfig,
    *,
    relative_to: str | Path | None = None,
    include_variant_alias: bool = False,
    include_family: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if include_variant_alias:
        payload["variant"] = config.variant
    payload["name"] = config.name or config.variant
    if include_family and config.family is not None:
        payload["family"] = config.family
    payload["threshold"] = float(config.threshold)
    if config.stats_path is not None:
        stats_path = config.stats_path.resolve()
        if relative_to is not None:
            payload["stats_path"] = relativize_path(stats_path, Path(relative_to).resolve())
        else:
            payload["stats_path"] = str(stats_path)
    payload["strict_stats"] = bool(config.strict_stats)
    if config.variant in POSTSPARSE_VARIANTS:
        payload["direction_mode"] = config.direction_mode
        payload["lambda_mix"] = float(config.lambda_mix)
        if config.alpha0 is not None:
            payload["alpha0"] = float(config.alpha0)
        payload["alpha0_mode"] = config.alpha0_mode
        payload["beta"] = float(config.beta)
        if config.beta_scale is not None:
            payload["beta_scale"] = float(config.beta_scale)
        payload["tau"] = float(config.tau)
    return payload


def extract_variant_hyperparameters(config: SparseDefenseConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "strict_stats": bool(config.strict_stats),
    }
    if config.variant in POSTSPARSE_VARIANTS:
        payload["direction_mode"] = config.direction_mode
        payload["lambda_mix"] = float(config.lambda_mix)
        if config.alpha0 is not None:
            payload["alpha0"] = float(config.alpha0)
        payload["alpha0_mode"] = config.alpha0_mode
        payload["beta"] = float(config.beta)
        if config.beta_scale is not None:
            payload["beta_scale"] = float(config.beta_scale)
        payload["tau"] = float(config.tau)
    return payload


def resolve_sparse_config_from_search_summary(
    summary_payload: dict[str, Any],
    *,
    summary_path: str | Path | None = None,
) -> SparseDefenseConfig:
    best_threshold = summary_payload.get("best_threshold", {})
    template_path = summary_payload.get("defense_template_config")
    effective_payload = summary_payload.get("effective_defense_config")
    if effective_payload is None and isinstance(best_threshold, dict):
        effective_payload = best_threshold.get("effective_defense_config")
    if effective_payload is not None and not isinstance(effective_payload, dict):
        raise ValueError("Expected `effective_defense_config` to be a mapping.")

    threshold = best_threshold.get("threshold", summary_payload.get("threshold"))
    stats_path = None
    if effective_payload is not None:
        stats_path = effective_payload.get("stats_path")
    if stats_path is None:
        stats_path = summary_payload.get("stats_path")

    base_dir = Path(summary_path).parent if summary_path is not None else None
    return resolve_sparse_defense_config(
        variant=str(summary_payload["variant"]),
        family=None if summary_payload.get("family") is None else str(summary_payload["family"]),
        threshold=None if threshold is None else float(threshold),
        stats_path=None if stats_path is None else str(stats_path),
        template_config_path=None if template_path in {None, ""} else str(template_path),
        template_payload=effective_payload,
        template_base_dir=base_dir,
    )
