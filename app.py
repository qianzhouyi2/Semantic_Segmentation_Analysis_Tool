from __future__ import annotations

from copy import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.apps.adversarial_preview import (
    build_layer_visualization,
    build_sample_delta_visualization,
    compute_layer_heatmaps,
    discover_attack_config_options,
    discover_checkpoint_options,
    discover_defense_config_options,
    generate_feature_preview,
    heatmap_display_note,
    run_feature_preview_sweep,
    select_representative_layer_names,
)
from src.apps.dashboard import build_overview_cards
from src.attacks import AttackConfig
from src.common.config import load_dataset_config, load_label_config
from src.common.config import load_yaml
from src.common.sample_manifest import filter_voc_sample_ids, load_voc_sample_id_manifest
from src.datasets import discover_ade20k_samples, discover_cityscapes_samples, discover_pascal_voc_samples
from src.datasets.scanner import scan_dataset
from src.datasets.stats import compute_class_statistics
from src.datasets.voc import PascalVOCValidationDataset
from src.io.image_io import DEFAULT_IMAGE_SUFFIXES, DEFAULT_MASK_SUFFIXES, save_image
from src.models import MODEL_FAMILY_CHOICES, TorchSegmentationModelAdapter, build_model_from_checkpoint
from src.reporting.exporter import write_csv, write_json, write_markdown
from src.robustness.visualization import HEATMAP_SCALE_MODE_CHOICES
from src.visualization.cam import build_cam_visualization, select_representative_cam_feature_keys
from src.visualization.response_region import build_response_region_visualization
from src.visualization.triplet import discover_triplet_samples, overlay_mask, render_triplet_from_paths


ANALYSIS_RESULT_STATE_KEYS = (
    "analysis_preview_result",
    "analysis_preview_signature",
    "analysis_preview_checkpoint_info",
    "analysis_sweep_results",
    "analysis_sweep_signature",
    "analysis_sweep_rows",
    "analysis_sweep_layer_rows",
    "cam_preview_cam_results",
    "cam_preview_cam_signature",
    "response_region_result",
    "response_region_signature",
)


def _clear_cached_analysis_resources() -> None:
    _load_model_adapter.clear()
    _load_voc_validation_dataset.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_analysis_result_state() -> None:
    for key in ANALYSIS_RESULT_STATE_KEYS:
        st.session_state.pop(key, None)


def _build_prediction_overlay_panels(result, label_config_path: str) -> list[tuple[str, np.ndarray]]:
    label_config = _optional_label_config(label_config_path)
    palette = label_config.palette if label_config else None
    ignore_index = label_config.ignore_index if label_config else None
    return [
        ("Ground Truth", overlay_mask(result.clean_image, result.ground_truth, palette=palette, ignore_index=ignore_index)),
        ("Clean Prediction", overlay_mask(result.clean_image, result.clean_prediction, palette=palette, ignore_index=ignore_index)),
        (
            "Adversarial Prediction",
            overlay_mask(result.adversarial_image, result.adversarial_prediction, palette=palette, ignore_index=ignore_index),
        ),
    ]


def _format_heatmap_scale_mode(mode: str) -> str:
    labels = {
        "independent": "独立色标",
        "shared": "共享色标",
        "fixed": "固定 [0,1] 色标",
    }
    return labels.get(mode, mode)


def _export_output_dirs(export_name: str) -> tuple[Path, Path, Path]:
    figures_dir = Path("results/figures/frontend_analysis") / export_name
    tables_dir = Path("results/tables/frontend_analysis") / export_name
    reports_dir = Path("results/reports/frontend_analysis") / export_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir, reports_dir


def _save_matplotlib_figure(path: Path, figure) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _build_export_name(sample_id: str, section_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_sample_id = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in sample_id)
    return f"{safe_sample_id}_{section_name}_{timestamp}"


def _sanitize_export_fragment(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value) or "artifact"


def _discover_sample_class_ids(preview_result) -> list[int]:
    return sorted(
        {
            *np.unique(preview_result.ground_truth).tolist(),
            *np.unique(preview_result.clean_prediction).tolist(),
            *np.unique(preview_result.adversarial_prediction).tolist(),
        }
    )


def _discover_target_class_ids(
    preview_result,
    label_config_path: str,
    *,
    show_all_classes: bool,
) -> tuple[list[int], dict[int, str], tuple[int, ...], list[int]]:
    label_config = _optional_label_config(label_config_path)
    class_names = label_config.class_names if label_config else {}
    background_ids = label_config.background_ids if label_config else (0,)
    present_class_ids = _discover_sample_class_ids(preview_result)
    if show_all_classes and label_config is not None:
        class_ids = list(label_config.class_ids)
    else:
        class_ids = present_class_ids
    if not class_ids and label_config is not None:
        class_ids = list(label_config.class_ids)
    return class_ids, class_names, background_ids, present_class_ids


def _build_sweep_rows(sweep_results, layer_names: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    for result in sweep_results:
        radius_255 = int(round(result.epsilon * 255.0))
        summary_rows.append(
            {
                "radius_255": radius_255,
                "epsilon": round(result.epsilon, 6),
                "step_size": round(result.step_size, 6),
                "sample_delta_mean": round(result.sample_delta_mean, 6),
                "sample_delta_max": round(result.sample_delta_max, 6),
            }
        )
        for layer_name in layer_names:
            _, _, diff_heatmap = compute_layer_heatmaps(result, layer_name)
            layer_rows.append(
                {
                    "radius_255": radius_255,
                    "layer_name": layer_name,
                    "mean_abs_diff": float(diff_heatmap.mean()),
                    "max_abs_diff": float(diff_heatmap.max()),
                }
            )
    return summary_rows, layer_rows


def _build_sweep_summary_figure(summary_rows: list[dict[str, object]]):
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    radii = [int(row["radius_255"]) for row in summary_rows]
    mean_values = [float(row["sample_delta_mean"]) for row in summary_rows]
    max_values = [float(row["sample_delta_max"]) for row in summary_rows]
    axis.plot(radii, mean_values, marker="o", label="sample_delta_mean")
    axis.plot(radii, max_values, marker="s", label="sample_delta_max")
    axis.set_xlabel("radius_255")
    axis.set_ylabel("delta")
    axis.set_title("Perturbation Sweep: Input Delta")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def _build_sweep_layer_figure(layer_rows: list[dict[str, object]]):
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    layer_names = sorted({str(row["layer_name"]) for row in layer_rows})
    for layer_name in layer_names:
        rows = [row for row in layer_rows if str(row["layer_name"]) == layer_name]
        rows.sort(key=lambda row: int(row["radius_255"]))
        axis.plot(
            [int(row["radius_255"]) for row in rows],
            [float(row["mean_abs_diff"]) for row in rows],
            marker="o",
            label=layer_name,
        )
    axis.set_xlabel("radius_255")
    axis.set_ylabel("mean_abs_diff")
    axis.set_title("Perturbation Sweep: Layer Diff Curves")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def _parse_sweep_radii(text: str) -> list[int]:
    if not text.strip():
        raise ValueError("Sweep radii cannot be empty.")
    radii: list[int] = []
    for chunk in text.replace(";", ",").split(","):
        raw = chunk.strip()
        if not raw:
            continue
        radius = int(float(raw))
        if radius < 0 or radius > 255:
            raise ValueError(f"Invalid sweep radius `{raw}`. Expected a value within [0, 255].")
        radii.append(radius)
    if not radii:
        raise ValueError("Sweep radii cannot be empty.")
    return sorted(set(radii))


def _optional_label_config(path_text: str):
    candidate = Path(path_text)
    if path_text and candidate.exists():
        return load_label_config(candidate)
    return None


def _resolve_dataset_inputs(dataset_config_path: str, image_dir_text: str, mask_dir_text: str):
    if dataset_config_path and Path(dataset_config_path).exists():
        dataset_config = load_dataset_config(dataset_config_path)
        return (
            dataset_config.image_dir,
            dataset_config.mask_dir,
            dataset_config.image_suffixes,
            dataset_config.mask_suffixes,
        )

    return Path(image_dir_text), Path(mask_dir_text), DEFAULT_IMAGE_SUFFIXES, DEFAULT_MASK_SUFFIXES


def _default_pascal_voc_config_paths() -> tuple[Path, Path]:
    return Path("configs/datasets/pascal_voc.yaml"), Path("configs/labels/pascal_voc.yaml")


def _default_cityscapes_config_path() -> Path:
    return Path("configs/datasets/cityscapes.yaml")


def _default_ade20k_config_path() -> Path:
    return Path("configs/datasets/ade20k.yaml")


def _resolve_pascal_voc_label_config_path(default_label_config_path: str) -> str:
    _, voc_label_config_path = _default_pascal_voc_config_paths()
    if voc_label_config_path.exists():
        return str(voc_label_config_path)
    return default_label_config_path


def _load_optional_voc_sample_manifest(path_text: str):
    candidate = Path(path_text.strip())
    if not path_text.strip():
        return None
    if not candidate.exists():
        raise FileNotFoundError(f"VOC sample id JSON not found: {candidate}")
    return load_voc_sample_id_manifest(candidate)


def _discover_voc_sample_manifest_options(root: str | Path = "samples") -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = [("<none>", "")]
    sample_root = Path(root)
    if not sample_root.exists():
        return options
    for path in sorted(sample_root.glob("*.json")):
        try:
            manifest = load_voc_sample_id_manifest(path)
        except ValueError:
            continue
        sample_ids = manifest.get("sample_ids", [])
        options.append((f"{manifest.get('name', path.stem)} ({len(sample_ids)})", str(path)))
    return options


def _filter_pascal_voc_samples_by_manifest(samples, manifest):
    allowed_ids = manifest.get("sample_ids", [])
    filtered_ids = set(filter_voc_sample_ids([sample.sample_id for sample in samples], allowed_ids))
    return [sample for sample in samples if sample.sample_id in filtered_ids]


def _filter_voc_dataset_by_manifest(dataset: PascalVOCValidationDataset, manifest) -> PascalVOCValidationDataset:
    filtered_dataset = copy(dataset)
    filtered_dataset.sample_ids = filter_voc_sample_ids(list(dataset.sample_ids), manifest.get("sample_ids", []))
    return filtered_dataset


def _resolve_optional_repo_label_config_path(filename: str) -> str:
    candidate = Path("configs/labels") / filename
    if candidate.exists():
        return str(candidate)
    return ""


@st.cache_resource(show_spinner=False)
def _load_voc_validation_dataset(dataset_root: str) -> PascalVOCValidationDataset:
    return PascalVOCValidationDataset(
        dataset_root,
        split="val",
        resize_short=473,
        crop_size=473,
        remap_ignore_to_background=False,
    )


@st.cache_resource(show_spinner=False)
def _load_model_adapter(
    family: str,
    checkpoint_path: str,
    defense_config_path: str,
    num_classes: int,
    device: str,
    strict: bool,
) -> tuple[TorchSegmentationModelAdapter, tuple[str, ...], tuple[str, ...], dict[str, object] | None]:
    model, missing_keys, unexpected_keys = build_model_from_checkpoint(
        family=family,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        map_location="cpu",
        strict=strict,
        defense_config_path=defense_config_path or None,
    )
    return (
        TorchSegmentationModelAdapter(model=model, num_classes=num_classes, device=device),
        tuple(missing_keys),
        tuple(unexpected_keys),
        getattr(model, "_sparse_defense_info", None),
    )


def _run_dataset_scan(dataset_config_path: str, label_config_path: str, image_dir_text: str, mask_dir_text: str):
    image_dir, mask_dir, image_suffixes, mask_suffixes = _resolve_dataset_inputs(
        dataset_config_path, image_dir_text, mask_dir_text
    )

    if not image_dir.exists() or not mask_dir.exists():
        st.error("Image or mask directory does not exist.")
        return

    label_config = _optional_label_config(label_config_path)
    class_names = label_config.class_names if label_config else {}
    allowed_label_ids = set(label_config.class_ids) if label_config else None
    ignore_index = label_config.ignore_index if label_config else None
    background_ids = label_config.background_ids if label_config else (0,)

    scan_result = scan_dataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_suffixes=image_suffixes,
        mask_suffixes=mask_suffixes,
        allowed_label_ids=allowed_label_ids,
        ignore_index=ignore_index,
        background_ids=background_ids,
    )
    dataset_stats = compute_class_statistics(
        scan_result.matched_pairs,
        class_names=class_names,
        ignore_index=ignore_index,
        background_ids=background_ids,
    )

    columns = st.columns(3)
    for index, (title, value) in enumerate(build_overview_cards(scan_result, dataset_stats)):
        columns[index % 3].metric(title, value)

    st.subheader("Class Distribution")
    st.dataframe(pd.DataFrame([row.to_dict() for row in dataset_stats.class_rows]), use_container_width=True)

    if scan_result.mismatched_shapes:
        st.warning(f"Found {len(scan_result.mismatched_shapes)} image/mask size mismatches.")
        st.code("\n".join(scan_result.mismatched_shapes[:20]))

    if scan_result.empty_masks:
        st.info(f"Found {len(scan_result.empty_masks)} masks that contain background only.")
        st.code("\n".join(scan_result.empty_masks[:20]))

    if scan_result.invalid_label_samples:
        st.error(f"Found {len(scan_result.invalid_label_samples)} masks with invalid label IDs.")
        st.code("\n".join(scan_result.invalid_label_samples[:20]))


def _run_triplet_preview(
    label_config_path: str,
    image_path_text: str,
    gt_path_text: str,
    pred_path_text: str,
    alpha: float,
    show_legend: bool,
):
    image_path_value = image_path_text.strip()
    if not image_path_value:
        st.error("Image path is required.")
        return

    image_path = Path(image_path_value)
    if not image_path.exists():
        st.error("Image path does not exist.")
        return

    gt_path = Path(gt_path_text.strip()) if gt_path_text.strip() else None
    pred_path = Path(pred_path_text.strip()) if pred_path_text.strip() else None
    palette = None
    class_names = None
    ignore_index = None
    if label_config_path and Path(label_config_path).exists():
        label_config = load_label_config(label_config_path)
        palette = label_config.palette
        class_names = label_config.class_names
        ignore_index = label_config.ignore_index

    try:
        figure = render_triplet_from_paths(
            image_path=image_path,
            ground_truth_path=gt_path if gt_path and gt_path.exists() else None,
            prediction_path=pred_path if pred_path and pred_path.exists() else None,
            palette=palette,
            class_names=class_names,
            alpha=alpha,
            ignore_index=ignore_index,
            show_legend=show_legend,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.pyplot(figure, clear_figure=True, use_container_width=True)


def _render_dataset_triplet_preview(
    dataset_config_path: str,
    label_config_path: str,
    image_dir_text: str,
    mask_dir_text: str,
):
    image_dir, mask_dir, image_suffixes, mask_suffixes = _resolve_dataset_inputs(
        dataset_config_path, image_dir_text, mask_dir_text
    )
    prediction_dir_text = st.text_input("Prediction directory", "")
    alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    show_legend = st.checkbox("Show class legend", value=True)
    require_prediction = st.checkbox("Only show samples with prediction", value=bool(prediction_dir_text.strip()))

    if not image_dir.exists() or not mask_dir.exists():
        st.info("Set valid image and GT mask directories to browse dataset samples.")
        return

    prediction_dir = Path(prediction_dir_text.strip()) if prediction_dir_text.strip() else None
    if prediction_dir is not None and not prediction_dir.exists():
        st.warning("Prediction directory does not exist. Samples will be shown without prediction overlays.")
        prediction_dir = None
        require_prediction = False

    try:
        samples = discover_triplet_samples(
            image_dir=image_dir,
            ground_truth_dir=mask_dir,
            prediction_dir=prediction_dir,
            image_suffixes=image_suffixes,
            mask_suffixes=mask_suffixes,
            require_prediction=require_prediction,
        )
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    if not samples:
        st.warning("No matched samples found for triplet preview.")
        return

    predicted_samples = sum(sample.prediction_path is not None for sample in samples)
    overview_columns = st.columns(3)
    overview_columns[0].metric("Samples", len(samples))
    overview_columns[1].metric("With prediction", predicted_samples)
    overview_columns[2].metric("Without prediction", len(samples) - predicted_samples)

    selected_key = st.selectbox("Sample", options=[sample.key for sample in samples])
    selected_sample = next(sample for sample in samples if sample.key == selected_key)

    st.caption(
        "\n".join(
            [
                f"Image: {selected_sample.image_path}",
                f"GT: {selected_sample.ground_truth_path}" if selected_sample.ground_truth_path else "GT: <missing>",
                (
                    f"Prediction: {selected_sample.prediction_path}"
                    if selected_sample.prediction_path
                    else "Prediction: <missing>"
                ),
            ]
        )
    )

    _run_triplet_preview(
        label_config_path=label_config_path,
        image_path_text=str(selected_sample.image_path),
        gt_path_text=str(selected_sample.ground_truth_path) if selected_sample.ground_truth_path else "",
        pred_path_text=str(selected_sample.prediction_path) if selected_sample.prediction_path else "",
        alpha=alpha,
        show_legend=show_legend,
    )


def _render_pascal_voc_triplet_preview(default_label_config_path: str):
    voc_dataset_config_path, voc_label_config_path = _default_pascal_voc_config_paths()
    dataset_config_path = str(voc_dataset_config_path) if voc_dataset_config_path.exists() else ""
    label_config_path = _resolve_pascal_voc_label_config_path(default_label_config_path)

    dataset_root = st.text_input("VOC dataset root", "datasets")
    sample_manifest_options = _discover_voc_sample_manifest_options()
    sample_manifest_preset_index = st.selectbox(
        "VOC sample id JSON (optional)",
        options=range(len(sample_manifest_options)),
        format_func=lambda index: sample_manifest_options[index][0],
        help="从 samples/*.json 里选择一个现成样本列表。",
    )
    prediction_dir_text = st.text_input("VOC prediction directory", "")
    alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05, key="voc_alpha")
    show_legend = st.checkbox("Show class legend", value=True, key="voc_legend")

    try:
        samples = discover_pascal_voc_samples(dataset_root, split="val")
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        if dataset_config_path:
            st.caption(f"Suggested dataset config: {dataset_config_path}")
        if label_config_path:
            st.caption(f"Suggested label config: {label_config_path}")
        return

    resolved_sample_manifest_path = sample_manifest_options[sample_manifest_preset_index][1]
    try:
        sample_manifest = _load_optional_voc_sample_manifest(resolved_sample_manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(str(exc))
        return
    if sample_manifest is not None:
        total_samples = len(samples)
        samples = _filter_pascal_voc_samples_by_manifest(samples, sample_manifest)
        st.caption(
            f"Filtered by sample id JSON: {len(samples)} / {total_samples} samples "
            f"from {sample_manifest.get('name', Path(resolved_sample_manifest_path).stem)}"
        )
        if not samples:
            st.warning("No VOC samples matched the provided sample id JSON.")
            return

    prediction_dir = Path(prediction_dir_text.strip()) if prediction_dir_text.strip() else None
    if prediction_dir is not None and not prediction_dir.exists():
        st.warning("VOC prediction directory does not exist. Preview will use image and GT only.")
        prediction_dir = None

    selected_sample_id = st.selectbox("VOC val sample", options=[sample.sample_id for sample in samples])
    selected_sample = next(sample for sample in samples if sample.sample_id == selected_sample_id)
    prediction_path = prediction_dir / f"{selected_sample.sample_id}.png" if prediction_dir is not None else None

    st.caption(
        "\n".join(
            [
                f"Image: {selected_sample.image_path}",
                f"GT: {selected_sample.mask_path}",
                f"Prediction: {prediction_path}" if prediction_path is not None else "Prediction: <missing>",
            ]
        )
    )

    _run_triplet_preview(
        label_config_path=label_config_path,
        image_path_text=str(selected_sample.image_path),
        gt_path_text=str(selected_sample.mask_path),
        pred_path_text=str(prediction_path) if prediction_path is not None else "",
        alpha=alpha,
        show_legend=show_legend,
    )


def _render_cityscapes_triplet_preview() -> None:
    dataset_config_path = _default_cityscapes_config_path()
    label_config_path = _resolve_optional_repo_label_config_path("cityscapes.yaml")

    dataset_root = st.text_input("Cityscapes dataset root", "datasets")
    prediction_dir_text = st.text_input("Cityscapes prediction directory", "")
    alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05, key="cityscapes_alpha")
    show_legend = st.checkbox("Show class legend", value=True, key="cityscapes_legend")

    try:
        samples = discover_cityscapes_samples(dataset_root, split="val")
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        if dataset_config_path.exists():
            st.caption(f"Suggested dataset config: {dataset_config_path}")
        return

    prediction_dir = Path(prediction_dir_text.strip()) if prediction_dir_text.strip() else None
    if prediction_dir is not None and not prediction_dir.exists():
        st.warning("Cityscapes prediction directory does not exist. Preview will use image and GT only.")
        prediction_dir = None

    selected_sample_id = st.selectbox("Cityscapes val sample", options=[sample.sample_id for sample in samples])
    selected_sample = next(sample for sample in samples if sample.sample_id == selected_sample_id)
    prediction_path = prediction_dir / selected_sample.relative_mask_path if prediction_dir is not None else None

    st.caption(
        "\n".join(
            [
                f"Image: {selected_sample.image_path}",
                f"GT: {selected_sample.mask_path}",
                f"Prediction: {prediction_path}" if prediction_path is not None else "Prediction: <missing>",
            ]
        )
    )

    _run_triplet_preview(
        label_config_path=label_config_path,
        image_path_text=str(selected_sample.image_path),
        gt_path_text=str(selected_sample.mask_path),
        pred_path_text=str(prediction_path) if prediction_path is not None else "",
        alpha=alpha,
        show_legend=show_legend,
    )


def _render_ade20k_triplet_preview() -> None:
    dataset_config_path = _default_ade20k_config_path()
    label_config_path = _resolve_optional_repo_label_config_path("ade20k.yaml")

    dataset_root = st.text_input("ADE20K dataset root", "datasets")
    prediction_dir_text = st.text_input("ADE20K prediction directory", "")
    alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05, key="ade20k_alpha")
    show_legend = st.checkbox("Show class legend", value=True, key="ade20k_legend")

    try:
        samples = discover_ade20k_samples(dataset_root, split="validation")
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        if dataset_config_path.exists():
            st.caption(f"Suggested dataset config: {dataset_config_path}")
        return

    prediction_dir = Path(prediction_dir_text.strip()) if prediction_dir_text.strip() else None
    if prediction_dir is not None and not prediction_dir.exists():
        st.warning("ADE20K prediction directory does not exist. Preview will use image and GT only.")
        prediction_dir = None

    selected_sample_id = st.selectbox("ADE20K validation sample", options=[sample.sample_id for sample in samples])
    selected_sample = next(sample for sample in samples if sample.sample_id == selected_sample_id)
    prediction_path = prediction_dir / selected_sample.relative_mask_path if prediction_dir is not None else None

    st.caption(
        "\n".join(
            [
                f"Image: {selected_sample.image_path}",
                f"GT: {selected_sample.mask_path}",
                f"Prediction: {prediction_path}" if prediction_path is not None else "Prediction: <missing>",
            ]
        )
    )

    _run_triplet_preview(
        label_config_path=label_config_path,
        image_path_text=str(selected_sample.image_path),
        gt_path_text=str(selected_sample.mask_path),
        pred_path_text=str(prediction_path) if prediction_path is not None else "",
        alpha=alpha,
        show_legend=show_legend,
    )


def _render_prediction_overlays(result, label_config_path: str) -> None:
    columns = st.columns(3)
    panels = _build_prediction_overlay_panels(result, label_config_path)
    for column, (title, image) in zip(columns, panels, strict=True):
        column.image(image, caption=title, use_container_width=True)


def _default_cam_class_id(result, background_ids: tuple[int, ...] = (0,)) -> int:
    labels, counts = np.unique(result.clean_prediction, return_counts=True)
    ranked = sorted(
        ((int(label), int(count)) for label, count in zip(labels.tolist(), counts.tolist(), strict=True)),
        key=lambda item: item[1],
        reverse=True,
    )
    for label, _count in ranked:
        if label not in background_ids:
            return label
    return int(ranked[0][0]) if ranked else 0


def _prepare_shared_analysis_preview() -> dict[str, object] | None:
    with st.expander(
        "Shared Analysis Controls",
        expanded=st.session_state.get("analysis_preview_result") is None,
    ):
        st.caption("下面的模型、攻击和热图显示设置由三个分析 tab 共用，攻击只会运行一次。")

        dataset_root = st.text_input("VOC dataset root", "datasets", key="analysis_dataset_root")
        sample_manifest_options = _discover_voc_sample_manifest_options()
        sample_manifest_preset_index = st.selectbox(
            "VOC sample id JSON (optional)",
            options=range(len(sample_manifest_options)),
            format_func=lambda index: sample_manifest_options[index][0],
            key="analysis_sample_manifest_preset_index",
            help="从 samples/*.json 里选择一个现成样本列表。",
        )
        try:
            dataset = _load_voc_validation_dataset(dataset_root.strip())
        except (FileNotFoundError, ValueError) as exc:
            st.info(str(exc))
            return None
        resolved_sample_manifest_path = sample_manifest_options[sample_manifest_preset_index][1]
        try:
            sample_manifest = _load_optional_voc_sample_manifest(resolved_sample_manifest_path)
        except (FileNotFoundError, ValueError) as exc:
            st.warning(str(exc))
            return None
        if sample_manifest is not None:
            total_samples = len(dataset.sample_ids)
            dataset = _filter_voc_dataset_by_manifest(dataset, sample_manifest)
            st.caption(
                f"Filtered analysis samples: {len(dataset.sample_ids)} / {total_samples} "
                f"from {sample_manifest.get('name', Path(resolved_sample_manifest_path).stem)}"
            )
            if not dataset.sample_ids:
                st.warning("No VOC samples matched the provided sample id JSON.")
                return None

        checkpoint_options = discover_checkpoint_options()
        defense_options = discover_defense_config_options()
        attack_options = discover_attack_config_options()
        attack_names = sorted({option.attack_name for option in attack_options})
        if not attack_names:
            st.error("No attack configs with valid `name` were found under configs/attacks.")
            return None

        control_left, control_middle, control_right = st.columns(3)
        with control_left:
            sample_index = st.selectbox(
                "样本",
                options=range(len(dataset.sample_ids)),
                format_func=lambda index: f"{index:04d} | {dataset.sample_ids[index]}",
                key="analysis_sample_index",
            )
            family = st.selectbox("模型 family", MODEL_FAMILY_CHOICES, key="analysis_model_family")
            family_checkpoints = [option for option in checkpoint_options if option.family == family]
            auto_checkpoint_path = ""
            if family_checkpoints:
                checkpoint_choice = st.selectbox(
                    "自动发现 checkpoint",
                    options=range(len(family_checkpoints)),
                    format_func=lambda index: family_checkpoints[index].label,
                    key="analysis_checkpoint_choice",
                )
                auto_checkpoint_path = str(family_checkpoints[checkpoint_choice].path)
                st.caption(f"Auto checkpoint: {auto_checkpoint_path}")
            else:
                st.warning("当前 family 没有自动发现到 checkpoint。")

            manual_checkpoint_path = st.text_input(
                "Manual checkpoint path (optional)",
                "",
                key="analysis_manual_checkpoint_path",
                help="填写后会覆盖自动发现的 checkpoint 选择。",
            )
            checkpoint_path = manual_checkpoint_path.strip() or auto_checkpoint_path
            if manual_checkpoint_path.strip():
                st.caption(f"Using manual checkpoint override: {manual_checkpoint_path.strip()}")

        with control_middle:
            family_defense_options = [option for option in defense_options if option.family in {None, family}]
            defense_mode_options = ["<none>"]
            if family_defense_options:
                defense_mode_options.append("<auto>")
            defense_mode_options.append("<manual>")
            defense_mode = st.selectbox(
                "稀疏防御配置",
                options=defense_mode_options,
                format_func=lambda value: {
                    "<none>": "不使用",
                    "<auto>": "自动发现配置",
                    "<manual>": "手动输入路径",
                }[value],
                key="analysis_defense_mode",
            )
            defense_config_path = ""
            if defense_mode == "<auto>":
                defense_choice = st.selectbox(
                    "防御配置文件",
                    options=range(len(family_defense_options)),
                    format_func=lambda index: family_defense_options[index].label,
                    key="analysis_defense_choice",
                )
                defense_config_path = str(family_defense_options[defense_choice].path)
                st.caption(f"Defense config: {defense_config_path}")
            elif defense_mode == "<manual>":
                defense_config_path = st.text_input("Defense config path", "", key="analysis_defense_config_path")
                if family_defense_options:
                    st.caption(f"已自动发现 {len(family_defense_options)} 个与当前 family 兼容的防御配置。")
                else:
                    st.caption("当前未自动发现与该 family 兼容的防御配置，请手动输入。")

            attack_name = st.selectbox("攻击", attack_names, key="analysis_attack_name")
            matched_attack_configs = [option for option in attack_options if option.attack_name == attack_name]
            attack_config_index = st.selectbox(
                "攻击配置",
                options=range(len(matched_attack_configs)),
                format_func=lambda index: matched_attack_configs[index].label,
                key="analysis_attack_config_index",
            )
            attack_config_path = matched_attack_configs[attack_config_index].path
            base_attack_config = AttackConfig.from_dict(load_yaml(attack_config_path))
            default_radius_255 = int(round(base_attack_config.epsilon * 255.0))
            if st.session_state.get("analysis_radius_config_path") != str(attack_config_path):
                st.session_state["analysis_radius_config_path"] = str(attack_config_path)
                st.session_state["analysis_radius_255"] = default_radius_255
            radius_255 = st.slider(
                "扰动半径 (0-255)",
                min_value=0,
                max_value=255,
                step=1,
                key="analysis_radius_255",
            )
            device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
            device = st.selectbox("Device", device_options, key="analysis_device")
            strict = st.checkbox("Strict checkpoint load", value=True, key="analysis_strict")

        with control_right:
            heatmap_scale_mode = st.selectbox(
                "热图色标",
                options=list(HEATMAP_SCALE_MODE_CHOICES),
                format_func=_format_heatmap_scale_mode,
                key="analysis_heatmap_scale_mode",
            )
            heatmap_percentile_clip_upper = st.slider(
                "热图百分位裁剪上界",
                min_value=90,
                max_value=100,
                value=100,
                step=1,
                key="analysis_heatmap_percentile_clip_upper",
                help="用于显示层面的色标裁剪；100 表示不裁剪。",
            )
            st.caption(heatmap_display_note(heatmap_scale_mode, float(heatmap_percentile_clip_upper)))

            button_columns = st.columns(3)
            run_clicked = button_columns[0].button("运行共享预览", use_container_width=True, key="analysis_run_button")
            force_reload_clicked = button_columns[1].button(
                "强制重载并运行",
                use_container_width=True,
                key="analysis_force_reload_button",
            )
            clear_cache_clicked = button_columns[2].button(
                "清空模型缓存",
                use_container_width=True,
                key="analysis_clear_cache_button",
            )
            if clear_cache_clicked:
                _clear_cached_analysis_resources()
                _clear_analysis_result_state()
                st.success("已清空模型/数据缓存，并移除当前分析结果。")

        attack_config = base_attack_config.with_radius_255(radius_255)
        st.caption(
            f"Effective attack config: radius={radius_255}/255, "
            f"epsilon={attack_config.epsilon:.6f}, "
            f"step_size={attack_config.resolved_step_size():.6f} ({attack_config.resolved_step_size() * 255.0:.2f}/255), "
            f"steps={attack_config.steps}"
        )
        with st.expander("查看攻击配置", expanded=False):
            st.code(attack_config_path.read_text(encoding="utf-8"), language="yaml")

        current_signature = {
            "dataset_root": dataset_root.strip(),
            "sample_index": int(sample_index),
            "family": family,
            "checkpoint_path": checkpoint_path,
            "defense_config_path": defense_config_path,
            "attack_config_path": str(attack_config_path),
            "radius_255": int(radius_255),
            "device": device,
            "strict": bool(strict),
        }

        if run_clicked or force_reload_clicked:
            checkpoint_candidate = Path(checkpoint_path.strip())
            if not checkpoint_path.strip():
                st.error("Checkpoint path is required.")
                return None
            if not checkpoint_candidate.exists():
                st.error("Checkpoint path does not exist.")
                return None
            defense_config_candidate = None
            if defense_config_path.strip():
                defense_config_candidate = Path(defense_config_path.strip())
                if not defense_config_candidate.exists():
                    st.error("Defense config path does not exist.")
                    return None

            if force_reload_clicked:
                _clear_cached_analysis_resources()
                _clear_analysis_result_state()

            with st.spinner("Loading model and generating preview..."):
                try:
                    adapter, missing_keys, unexpected_keys, sparse_defense_info = _load_model_adapter(
                        family=family,
                        checkpoint_path=str(checkpoint_candidate),
                        defense_config_path=str(defense_config_candidate) if defense_config_candidate is not None else "",
                        num_classes=21,
                        device=device,
                        strict=strict,
                    )
                    image, target, filename = dataset[sample_index]
                    preview_result = generate_feature_preview(
                        model=adapter,
                        attack_config=attack_config,
                        image=image,
                        target=target,
                        sample_id=Path(filename).stem,
                    )
                except Exception as exc:  # pragma: no cover - surfaced in UI
                    st.error(str(exc))
                    st.exception(exc)
                    return None

            _clear_analysis_result_state()
            st.session_state["analysis_preview_result"] = preview_result
            st.session_state["analysis_preview_signature"] = current_signature
            st.session_state["analysis_preview_checkpoint_info"] = {
                "missing_keys": list(missing_keys),
                "unexpected_keys": list(unexpected_keys),
                "checkpoint_path": str(checkpoint_candidate.resolve()),
                "defense_config_path": (
                    str(defense_config_candidate.resolve()) if defense_config_candidate is not None else None
                ),
                "sparse_defense_info": sparse_defense_info,
                "family": family,
            }

        preview_result = st.session_state.get("analysis_preview_result")
        if preview_result is None:
            return None

        stored_signature = st.session_state.get("analysis_preview_signature", {})
        if stored_signature != current_signature:
            st.info("当前展示的是上一次运行结果。修改模型、攻击或样本后，需要重新点击运行按钮更新。")

        checkpoint_info = st.session_state.get("analysis_preview_checkpoint_info", {})
        return {
            "preview_result": preview_result,
            "checkpoint_info": checkpoint_info,
            "stored_signature": stored_signature,
            "family": family,
            "checkpoint_path": checkpoint_path,
            "heatmap_scale_mode": heatmap_scale_mode,
            "heatmap_percentile_clip_upper": float(heatmap_percentile_clip_upper),
        }


def _render_preview_summary(preview_result, checkpoint_info: dict[str, object], family: str, checkpoint_path: str) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric("Sample", preview_result.sample_id)
    metric_columns[1].metric("Layers", len(preview_result.layer_names))
    metric_columns[2].metric("Radius", f"{int(round(preview_result.epsilon * 255.0))}/255")
    metric_columns[3].metric("step_size", f"{preview_result.step_size * 255.0:.2f}/255")
    st.caption(
        "\n".join(
            [
                f"Family: {checkpoint_info.get('family', family)}",
                f"Checkpoint: {checkpoint_info.get('checkpoint_path', checkpoint_path)}",
                (
                    f"Defense config: {checkpoint_info.get('defense_config_path')}"
                    if checkpoint_info.get("defense_config_path")
                    else "Defense config: <none>"
                ),
                (
                    "Sparse defense: "
                    f"{checkpoint_info.get('sparse_defense_info', {}).get('variant')} "
                    f"(threshold={checkpoint_info.get('sparse_defense_info', {}).get('threshold', 0.0):.4f})"
                    if checkpoint_info.get("sparse_defense_info")
                    else "Sparse defense: <none>"
                ),
                f"Missing keys: {len(checkpoint_info.get('missing_keys', []))}",
                f"Unexpected keys: {len(checkpoint_info.get('unexpected_keys', []))}",
            ]
        )
    )


def _render_preview_images(
    preview_result,
    label_config_path: str,
    *,
    heatmap_scale_mode: str,
    heatmap_percentile_clip_upper: float,
) -> None:
    sample_delta_payload = build_sample_delta_visualization(
        preview_result,
        scale_mode=heatmap_scale_mode,
        percentile_clip_upper=heatmap_percentile_clip_upper,
    )
    input_columns = st.columns(4)
    input_columns[0].image(preview_result.clean_image, caption="Clean Input", use_container_width=True)
    input_columns[1].image(preview_result.adversarial_image, caption="Adversarial Input", use_container_width=True)
    input_columns[2].image(preview_result.perturbation_image, caption="Perturbation", use_container_width=True)
    input_columns[3].image(
        sample_delta_payload["delta_image"],
        caption=(
            f"Sample Delta Normalized Heatmap | mean={preview_result.sample_delta_mean:.4f} "
            f"max={preview_result.sample_delta_max:.4f} | {sample_delta_payload['display_note']}"
        ),
        use_container_width=True,
    )

    _render_prediction_overlays(preview_result, label_config_path)


def _load_context_adapter(context: dict[str, object]) -> TorchSegmentationModelAdapter:
    checkpoint_info = context["checkpoint_info"]
    stored_signature = context["stored_signature"]
    family = context["family"]
    checkpoint_path = context["checkpoint_path"]
    adapter, _, _, _ = _load_model_adapter(
        family=checkpoint_info.get("family", family),
        checkpoint_path=checkpoint_info.get("checkpoint_path", checkpoint_path),
        defense_config_path=str(checkpoint_info.get("defense_config_path") or ""),
        num_classes=21,
        device=stored_signature.get("device", "cpu"),
        strict=bool(stored_signature.get("strict", True)),
    )
    return adapter


def _render_sweep_mode(context: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    preview_result = context["preview_result"]
    stored_signature = context["stored_signature"]
    st.subheader("Perturbation Sweep")
    radii_text = st.text_input(
        "Sweep radii (0-255, comma separated)",
        "0,1,2,4,8",
        key="analysis_sweep_radii_text",
    )
    representative_layers = select_representative_layer_names(preview_result.layer_names, max_layers=3)
    if representative_layers:
        st.caption("Sweep 默认输出代表层曲线: " + " | ".join(representative_layers))

    if st.button("运行扰动 Sweep", use_container_width=True, key="analysis_sweep_run_button"):
        try:
            radii_255 = _parse_sweep_radii(radii_text)
        except ValueError as exc:
            st.error(str(exc))
            return [], []

        sweep_signature = {
            **stored_signature,
            "sweep_radii_255": tuple(radii_255),
        }
        if st.session_state.get("analysis_sweep_signature") != sweep_signature:
            adapter = _load_context_adapter(context)
            base_attack_config = AttackConfig.from_dict(load_yaml(Path(stored_signature["attack_config_path"])))
            with st.spinner("Running perturbation sweep..."):
                sweep_results = run_feature_preview_sweep(
                    model=adapter,
                    attack_config=base_attack_config,
                    image=preview_result.clean_tensor,
                    target=torch.from_numpy(preview_result.ground_truth.copy()).long(),
                    sample_id=preview_result.sample_id,
                    radii_255=radii_255,
                )
            summary_rows, layer_rows = _build_sweep_rows(sweep_results, representative_layers)
            st.session_state["analysis_sweep_results"] = sweep_results
            st.session_state["analysis_sweep_rows"] = summary_rows
            st.session_state["analysis_sweep_layer_rows"] = layer_rows
            st.session_state["analysis_sweep_signature"] = sweep_signature

    summary_rows = st.session_state.get("analysis_sweep_rows", [])
    layer_rows = st.session_state.get("analysis_sweep_layer_rows", [])
    sweep_results = st.session_state.get("analysis_sweep_results", [])
    if not summary_rows or not layer_rows or not sweep_results:
        return [], []

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    summary_figure = _build_sweep_summary_figure(summary_rows)
    st.pyplot(summary_figure, clear_figure=True, use_container_width=True)
    layer_figure = _build_sweep_layer_figure(layer_rows)
    st.pyplot(layer_figure, clear_figure=True, use_container_width=True)

    key_indices = sorted({0, len(sweep_results) // 2, len(sweep_results) - 1})
    key_results = [sweep_results[index] for index in key_indices]
    if representative_layers:
        st.caption(f"关键图默认展示代表层中的最深层: {representative_layers[-1]}")
    key_columns = st.columns(len(key_results))
    for column, sweep_result in zip(key_columns, key_results, strict=True):
        radius_255 = int(round(sweep_result.epsilon * 255.0))
        sample_delta_payload = build_sample_delta_visualization(
            sweep_result,
            scale_mode=context["heatmap_scale_mode"],
            percentile_clip_upper=context["heatmap_percentile_clip_upper"],
        )
        column.image(
            sample_delta_payload["delta_image"],
            caption=f"radius={radius_255}/255 | input normalized heatmap",
            use_container_width=True,
        )
        if representative_layers:
            layer_visualization = build_layer_visualization(
                sweep_result,
                representative_layers[-1],
                scale_mode=context["heatmap_scale_mode"],
                percentile_clip_upper=context["heatmap_percentile_clip_upper"],
            )
            column.image(
                layer_visualization["diff_image"],
                caption=f"{representative_layers[-1]} | diff normalized heatmap",
                use_container_width=True,
            )

    return summary_rows, layer_rows


def _export_adversarial_feature_preview(
    context: dict[str, object],
    label_config_path: str,
    layer_name: str,
    layer_visualization: dict[str, object],
    sweep_rows: list[dict[str, object]],
    sweep_layer_rows: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    preview_result = context["preview_result"]
    checkpoint_info = context["checkpoint_info"]
    export_name = _build_export_name(preview_result.sample_id, "adversarial_feature")
    figures_dir, tables_dir, reports_dir = _export_output_dirs(export_name)
    sample_delta_payload = build_sample_delta_visualization(
        preview_result,
        scale_mode=context["heatmap_scale_mode"],
        percentile_clip_upper=context["heatmap_percentile_clip_upper"],
    )
    save_image(figures_dir / "clean_input.png", preview_result.clean_image)
    save_image(figures_dir / "adversarial_input.png", preview_result.adversarial_image)
    save_image(figures_dir / "perturbation.png", preview_result.perturbation_image)
    save_image(figures_dir / "sample_delta_normalized_heatmap.png", sample_delta_payload["delta_image"])
    for title, image in _build_prediction_overlay_panels(preview_result, label_config_path):
        save_image(figures_dir / f"{_sanitize_export_fragment(title.lower())}.png", image)
    save_image(figures_dir / f"{_sanitize_export_fragment(layer_name)}_clean.png", layer_visualization["clean_image"])
    save_image(figures_dir / f"{_sanitize_export_fragment(layer_name)}_adv.png", layer_visualization["adversarial_image"])
    save_image(figures_dir / f"{_sanitize_export_fragment(layer_name)}_diff.png", layer_visualization["diff_image"])
    if sweep_rows and sweep_layer_rows:
        write_csv(tables_dir / "perturbation_sweep_summary.csv", sweep_rows)
        write_csv(tables_dir / "perturbation_sweep_layer_curves.csv", sweep_layer_rows)
        _save_matplotlib_figure(figures_dir / "perturbation_sweep_summary.png", _build_sweep_summary_figure(sweep_rows))
        _save_matplotlib_figure(figures_dir / "perturbation_sweep_layer_curves.png", _build_sweep_layer_figure(sweep_layer_rows))

    write_json(
        reports_dir / "summary.json",
        {
            "sample_id": preview_result.sample_id,
            "layer_name": layer_name,
            "heatmap_display": heatmap_display_note(
                context["heatmap_scale_mode"],
                context["heatmap_percentile_clip_upper"],
            ),
            "checkpoint": checkpoint_info.get("checkpoint_path"),
            "attack_name": preview_result.attack_name,
            "epsilon": preview_result.epsilon,
            "step_size": preview_result.step_size,
            "sample_delta_mean": preview_result.sample_delta_mean,
            "sample_delta_max": preview_result.sample_delta_max,
            "layer_mean_abs_diff": layer_visualization["mean_abs_diff"],
            "layer_max_abs_diff": layer_visualization["max_abs_diff"],
        },
    )
    write_markdown(
        reports_dir / "summary.md",
        "Adversarial Feature Preview",
        [
            f"- sample_id: {preview_result.sample_id}",
            f"- checkpoint: {checkpoint_info.get('checkpoint_path')}",
            f"- attack: {preview_result.attack_name}",
            f"- epsilon: {preview_result.epsilon:.6f}",
            f"- step_size: {preview_result.step_size:.6f}",
            f"- selected_layer: {layer_name}",
            f"- layer_mean_abs_diff: {layer_visualization['mean_abs_diff']:.6f}",
            f"- layer_max_abs_diff: {layer_visualization['max_abs_diff']:.6f}",
            f"- heatmap_display: {heatmap_display_note(context['heatmap_scale_mode'], context['heatmap_percentile_clip_upper'])}",
        ],
    )
    return figures_dir, tables_dir, reports_dir


def _export_cam_preview(
    context: dict[str, object],
    cam_results,
    summary_rows: list[dict[str, object]],
    selected_class_name: str,
) -> tuple[Path, Path, Path]:
    preview_result = context["preview_result"]
    checkpoint_info = context["checkpoint_info"]
    export_name = _build_export_name(preview_result.sample_id, "cam_preview")
    figures_dir, tables_dir, reports_dir = _export_output_dirs(export_name)
    for index, cam_result in enumerate(cam_results):
        prefix = f"{index:02d}_{_sanitize_export_fragment(cam_result.feature_key)}"
        save_image(figures_dir / f"{prefix}_clean.png", cam_result.clean_overlay)
        save_image(figures_dir / f"{prefix}_adv.png", cam_result.adversarial_overlay)
        save_image(figures_dir / f"{prefix}_diff.png", cam_result.diff_image)
    write_csv(tables_dir / "cam_summary.csv", summary_rows)
    write_json(
        reports_dir / "summary.json",
        {
            "sample_id": preview_result.sample_id,
            "target_class": selected_class_name,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
            "heatmap_display": heatmap_display_note(
                context["heatmap_scale_mode"],
                context["heatmap_percentile_clip_upper"],
            ),
            "rows": summary_rows,
        },
    )
    write_markdown(
        reports_dir / "summary.md",
        "CAM Preview",
        [
            f"- sample_id: {preview_result.sample_id}",
            f"- target_class: {selected_class_name}",
            f"- checkpoint: {checkpoint_info.get('checkpoint_path')}",
            f"- heatmap_display: {heatmap_display_note(context['heatmap_scale_mode'], context['heatmap_percentile_clip_upper'])}",
            "",
            "## Layers",
            *[
                (
                    f"- {row['layer']}: feature_key={row['feature_key']}, "
                    f"cam_mean(clean)={row['cam_mean(clean)']}, cam_mean(adv)={row['cam_mean(adv)']}, "
                    f"cam_diff_mean={row['cam_diff_mean']}"
                )
                for row in summary_rows
            ],
        ],
    )
    return figures_dir, tables_dir, reports_dir


def _export_response_region_preview(
    context: dict[str, object],
    response_result,
    selected_class_name: str,
) -> tuple[Path, Path, Path]:
    preview_result = context["preview_result"]
    checkpoint_info = context["checkpoint_info"]
    export_name = _build_export_name(preview_result.sample_id, "response_region")
    figures_dir, tables_dir, reports_dir = _export_output_dirs(export_name)
    save_image(figures_dir / "clean_response_overlay.png", response_result.clean_overlay)
    save_image(figures_dir / "adv_response_overlay.png", response_result.adversarial_overlay)
    save_image(figures_dir / "response_diff_overlay.png", response_result.diff_overlay)
    save_image(figures_dir / "clean_region_overlay.png", response_result.clean_region_overlay)
    save_image(figures_dir / "adv_region_overlay.png", response_result.adversarial_region_overlay)
    save_image(figures_dir / "overlap_region_overlay.png", response_result.overlap_region_overlay)
    write_csv(
        tables_dir / "response_region_summary.csv",
        [
            {
                "class_id": response_result.class_id,
                "threshold_percentile": response_result.threshold_percentile,
                "clean_mean": response_result.clean_mean,
                "adv_mean": response_result.adversarial_mean,
                "diff_mean": response_result.diff_mean,
                "clean_area_ratio": response_result.clean_active_ratio,
                "adv_area_ratio": response_result.adversarial_active_ratio,
                "overlap_iou": response_result.overlap_iou,
                "clean_target_pixels": response_result.clean_target_pixels,
                "adv_target_pixels": response_result.adversarial_target_pixels,
            }
        ],
    )
    write_json(
        reports_dir / "summary.json",
        {
            "sample_id": preview_result.sample_id,
            "target_class": selected_class_name,
            "checkpoint": checkpoint_info.get("checkpoint_path"),
            "heatmap_display": heatmap_display_note(
                context["heatmap_scale_mode"],
                context["heatmap_percentile_clip_upper"],
            ),
            "response_metrics": {
                "clean_mean": response_result.clean_mean,
                "adv_mean": response_result.adversarial_mean,
                "diff_mean": response_result.diff_mean,
                "clean_area_ratio": response_result.clean_active_ratio,
                "adv_area_ratio": response_result.adversarial_active_ratio,
                "overlap_iou": response_result.overlap_iou,
            },
        },
    )
    write_markdown(
        reports_dir / "summary.md",
        "Response Region Preview",
        [
            f"- sample_id: {preview_result.sample_id}",
            f"- target_class: {selected_class_name}",
            f"- checkpoint: {checkpoint_info.get('checkpoint_path')}",
            f"- threshold_percentile: {response_result.threshold_percentile}",
            f"- clean_mean: {response_result.clean_mean:.6f}",
            f"- adv_mean: {response_result.adversarial_mean:.6f}",
            f"- diff_mean: {response_result.diff_mean:.6f}",
            f"- overlap_iou: {response_result.overlap_iou:.6f}",
            f"- heatmap_display: {heatmap_display_note(context['heatmap_scale_mode'], context['heatmap_percentile_clip_upper'])}",
        ],
    )
    return figures_dir, tables_dir, reports_dir


def _render_adversarial_feature_preview(label_config_path: str, context: dict[str, object] | None) -> None:
    if context is None:
        st.info("请先在 Shared Analysis Controls 中运行共享预览。")
        return

    resolved_label_config_path = _resolve_pascal_voc_label_config_path(label_config_path)
    preview_result = context["preview_result"]
    checkpoint_info = context["checkpoint_info"]
    family = context["family"]
    checkpoint_path = context["checkpoint_path"]

    _render_preview_summary(preview_result, checkpoint_info, family, checkpoint_path)
    _render_preview_images(
        preview_result,
        resolved_label_config_path,
        heatmap_scale_mode=context["heatmap_scale_mode"],
        heatmap_percentile_clip_upper=context["heatmap_percentile_clip_upper"],
    )

    if not preview_result.layer_names:
        st.warning("The selected model did not return any feature layers for visualization.")
        return

    selected_layer_index = st.slider(
        "层编号",
        min_value=0,
        max_value=len(preview_result.layer_names) - 1,
        value=min(st.session_state.get("analysis_layer_index", 0), len(preview_result.layer_names) - 1),
        key="analysis_layer_index",
    )
    layer_name = preview_result.layer_names[selected_layer_index]
    layer_visualization = build_layer_visualization(
        preview_result,
        layer_name,
        scale_mode=context["heatmap_scale_mode"],
        percentile_clip_upper=context["heatmap_percentile_clip_upper"],
    )
    st.caption(
        f"当前层 {selected_layer_index + 1}/{len(preview_result.layer_names)}: {layer_name} | "
        f"{layer_visualization['display_note']}"
    )

    feature_columns = st.columns(3)
    feature_columns[0].image(
        layer_visualization["clean_image"],
        caption=f"Clean Feature Normalized Heatmap | shape={layer_visualization['clean_shape']}",
        use_container_width=True,
    )
    feature_columns[1].image(
        layer_visualization["adversarial_image"],
        caption=f"Adversarial Feature Normalized Heatmap | shape={layer_visualization['adversarial_shape']}",
        use_container_width=True,
    )
    feature_columns[2].image(
        layer_visualization["diff_image"],
        caption=(
            f"Abs Diff Normalized Heatmap | mean={layer_visualization['mean_abs_diff']:.4f} "
            f"max={layer_visualization['max_abs_diff']:.4f}"
        ),
        use_container_width=True,
    )

    sweep_rows, sweep_layer_rows = _render_sweep_mode(context)
    if st.button("导出当前对抗特征分析", use_container_width=True, key="analysis_export_feature_button"):
        figures_dir, tables_dir, reports_dir = _export_adversarial_feature_preview(
            context,
            resolved_label_config_path,
            layer_name,
            layer_visualization,
            sweep_rows,
            sweep_layer_rows,
        )
        st.success(
            "导出完成："
            f"\nfigures={figures_dir.resolve()}"
            f"\ntables={tables_dir.resolve()}"
            f"\nreports={reports_dir.resolve()}"
        )


def _render_cam_preview(label_config_path: str, context: dict[str, object] | None) -> None:
    if context is None:
        st.info("请先在 Shared Analysis Controls 中运行共享预览。")
        return

    resolved_label_config_path = _resolve_pascal_voc_label_config_path(label_config_path)
    preview_result = context["preview_result"]
    checkpoint_info = context["checkpoint_info"]
    stored_signature = context["stored_signature"]
    family = context["family"]
    checkpoint_path = context["checkpoint_path"]

    _render_preview_summary(preview_result, checkpoint_info, family, checkpoint_path)
    _render_preview_images(
        preview_result,
        resolved_label_config_path,
        heatmap_scale_mode=context["heatmap_scale_mode"],
        heatmap_percentile_clip_upper=context["heatmap_percentile_clip_upper"],
    )

    adapter = _load_context_adapter(context)
    cam_feature_keys = select_representative_cam_feature_keys(adapter, preview_result.layer_names, max_keys=3)
    if not cam_feature_keys:
        st.info("CAM 当前只支持返回 4D 特征图的层。当前模型没有可用的 CAM 层。")
        return

    show_all_classes = st.checkbox("Show all classes", value=False, key="cam_preview_show_all_classes")
    cam_class_ids, class_names, background_ids, present_class_ids = _discover_target_class_ids(
        preview_result,
        resolved_label_config_path,
        show_all_classes=show_all_classes,
    )

    default_cam_class = _default_cam_class_id(preview_result, background_ids=background_ids)
    if st.session_state.get("cam_preview_default_class_initialized") != preview_result.sample_id:
        st.session_state["cam_preview_default_class_initialized"] = preview_result.sample_id
        st.session_state["cam_preview_class_id"] = default_cam_class

    cam_class_id = st.selectbox(
        "CAM target class",
        options=cam_class_ids,
        format_func=lambda class_id: class_names.get(int(class_id), f"class_{class_id}"),
        key="cam_preview_class_id",
    )
    selected_class_name = class_names.get(int(cam_class_id), f"class_{cam_class_id}")
    if int(cam_class_id) not in set(present_class_ids):
        st.warning("当前所选类别不在该样本的 GT / clean prediction / adv prediction 中，热图将使用 fallback 目标分数。")

    if len(cam_feature_keys) == 1:
        st.caption(f"CAM 自动输出当前唯一可用层: {cam_feature_keys[0]}")
    elif len(cam_feature_keys) == 2:
        st.caption(f"CAM 自动输出两层代表层: 浅层={cam_feature_keys[0]} | 深层={cam_feature_keys[1]}")
    else:
        st.caption(
            "CAM 自动输出三层代表层: "
            f"浅层={cam_feature_keys[0]} | 中层={cam_feature_keys[1]} | 深层={cam_feature_keys[2]}"
        )

    cam_signature = {
        **stored_signature,
        "cam_feature_keys": tuple(cam_feature_keys),
        "cam_class_id": int(cam_class_id),
        "heatmap_scale_mode": context["heatmap_scale_mode"],
        "heatmap_percentile_clip_upper": context["heatmap_percentile_clip_upper"],
    }
    if st.session_state.get("cam_preview_cam_signature") != cam_signature:
        with st.spinner("Computing CAM..."):
            try:
                cam_results = [
                    build_cam_visualization(
                        model=adapter,
                        clean_tensor=preview_result.clean_tensor,
                        adversarial_tensor=preview_result.adversarial_tensor,
                        clean_image=preview_result.clean_image,
                        adversarial_image=preview_result.adversarial_image,
                        feature_key=feature_key,
                        class_id=int(cam_class_id),
                        ground_truth=preview_result.ground_truth,
                        clean_prediction=preview_result.clean_prediction,
                        heatmap_scale_mode=context["heatmap_scale_mode"],
                        heatmap_percentile_clip_upper=context["heatmap_percentile_clip_upper"],
                    )
                    for feature_key in cam_feature_keys
                ]
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(str(exc))
                st.exception(exc)
                return
        st.session_state["cam_preview_cam_results"] = cam_results
        st.session_state["cam_preview_cam_signature"] = cam_signature

    cam_results = st.session_state.get("cam_preview_cam_results")
    if not cam_results:
        return

    if any(cam_result.clean_used_fallback or cam_result.adversarial_used_fallback for cam_result in cam_results):
        st.warning("当前 CAM 至少有一侧使用了 fallback 目标分数：当目标类别在当前预测中不存在时，会退化为整图平均 logit。")

    clean_pred_pixels = int((preview_result.clean_prediction == int(cam_class_id)).sum())
    adversarial_pred_pixels = int((preview_result.adversarial_prediction == int(cam_class_id)).sum())
    st.subheader("Global CAM Summary")
    st.caption(
        f"Target class: {selected_class_name} ({int(cam_class_id)}) | "
        f"{heatmap_display_note(context['heatmap_scale_mode'], context['heatmap_percentile_clip_upper'])}"
    )
    global_columns = st.columns(2)
    global_columns[0].metric("pred_pixels(clean)", clean_pred_pixels)
    global_columns[1].metric("pred_pixels(adv)", adversarial_pred_pixels)

    layer_labels = ["浅层", "中层", "深层"]
    if len(cam_results) == 1:
        layer_labels = ["可用层"]
    elif len(cam_results) == 2:
        layer_labels = ["浅层", "深层"]

    summary_rows: list[dict[str, object]] = []
    for layer_label, cam_result in zip(layer_labels, cam_results, strict=True):
        summary_rows.append(
            {
                "layer": layer_label,
                "feature_key": cam_result.feature_key,
                "cam_mean(clean)": round(cam_result.clean_mean, 4),
                "cam_mean(adv)": round(cam_result.adversarial_mean, 4),
                "cam_diff_mean": round(cam_result.diff_mean, 4),
                "cam_area_top20%(clean)": round(cam_result.clean_top20_area_ratio * 100.0, 2),
                "cam_area_top20%(adv)": round(cam_result.adversarial_top20_area_ratio * 100.0, 2),
                "cam_inside_gt_ratio(clean)": round(cam_result.clean_inside_gt_ratio, 4),
                "cam_inside_gt_ratio(adv)": round(cam_result.adversarial_inside_gt_ratio, 4),
                "cam_inside_clean_pred_ratio(clean)": round(cam_result.clean_inside_clean_prediction_ratio, 4),
                "cam_inside_clean_pred_ratio(adv)": round(cam_result.adversarial_inside_clean_prediction_ratio, 4),
                "cam_centroid_shift_px": (
                    round(cam_result.centroid_shift, 2) if cam_result.centroid_shift is not None else None
                ),
            }
        )

    st.subheader("Per-Layer CAM Metrics")
    st.caption(
        "cam_area_top20% 使用 CAM 热图前 20% 高响应区域；inside_*_ratio 表示该区域落在 GT 或 clean prediction 目标区域内的比例。"
    )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    for layer_label, cam_result in zip(layer_labels, cam_results, strict=True):
        st.caption(f"{layer_label} CAM 层: {cam_result.feature_key}")
        cam_columns = st.columns(3)
        cam_columns[0].image(
            cam_result.clean_overlay,
            caption=f"Clean CAM Normalized Heatmap | mean={cam_result.clean_mean:.4f}",
            use_container_width=True,
        )
        cam_columns[1].image(
            cam_result.adversarial_overlay,
            caption=f"Adv CAM Normalized Heatmap | mean={cam_result.adversarial_mean:.4f}",
            use_container_width=True,
        )
        cam_columns[2].image(
            cam_result.diff_image,
            caption=f"CAM Diff Normalized Heatmap | mean={cam_result.diff_mean:.4f}",
            use_container_width=True,
        )

    if st.button("导出当前 CAM 分析", use_container_width=True, key="cam_preview_export_button"):
        figures_dir, tables_dir, reports_dir = _export_cam_preview(
            context,
            cam_results,
            summary_rows,
            selected_class_name,
        )
        st.success(
            "导出完成："
            f"\nfigures={figures_dir.resolve()}"
            f"\ntables={tables_dir.resolve()}"
            f"\nreports={reports_dir.resolve()}"
        )


def _render_response_region_preview(label_config_path: str, context: dict[str, object] | None) -> None:
    if context is None:
        st.info("请先在 Shared Analysis Controls 中运行共享预览。")
        return

    resolved_label_config_path = _resolve_pascal_voc_label_config_path(label_config_path)
    preview_result = context["preview_result"]
    checkpoint_info = context["checkpoint_info"]
    stored_signature = context["stored_signature"]
    family = context["family"]
    checkpoint_path = context["checkpoint_path"]

    _render_preview_summary(preview_result, checkpoint_info, family, checkpoint_path)
    _render_preview_images(
        preview_result,
        resolved_label_config_path,
        heatmap_scale_mode=context["heatmap_scale_mode"],
        heatmap_percentile_clip_upper=context["heatmap_percentile_clip_upper"],
    )

    show_all_classes = st.checkbox("Show all classes", value=False, key="response_region_show_all_classes")
    class_ids, class_names, background_ids, present_class_ids = _discover_target_class_ids(
        preview_result,
        resolved_label_config_path,
        show_all_classes=show_all_classes,
    )
    default_class_id = _default_cam_class_id(preview_result, background_ids=background_ids)
    if st.session_state.get("response_region_default_class_initialized") != preview_result.sample_id:
        st.session_state["response_region_default_class_initialized"] = preview_result.sample_id
        st.session_state["response_region_class_id"] = default_class_id

    control_left, control_right = st.columns(2)
    with control_left:
        class_id = st.selectbox(
            "响应目标类别",
            options=class_ids,
            format_func=lambda value: class_names.get(int(value), f"class_{value}"),
            key="response_region_class_id",
        )
    with control_right:
        threshold_percentile = st.slider(
            "响应区域分位阈值",
            min_value=50,
            max_value=99,
            value=85,
            step=1,
            key="response_region_threshold_percentile",
            help="仅保留响应热图中高于该分位阈值的像素作为响应区域。",
        )

    if int(class_id) not in set(present_class_ids):
        st.warning("当前所选类别不在该样本的 GT / clean prediction / adv prediction 中，响应热图将使用 fallback 目标分数。")

    st.caption(
        "响应得分定义为目标类别在当前预测区域上的平均 logit；如果该类别当前没有预测像素，则回退为整张图上的该类平均 logit。"
    )

    response_signature = {
        **stored_signature,
        "class_id": int(class_id),
        "threshold_percentile": int(threshold_percentile),
        "heatmap_scale_mode": context["heatmap_scale_mode"],
        "heatmap_percentile_clip_upper": context["heatmap_percentile_clip_upper"],
    }
    if st.session_state.get("response_region_signature") != response_signature:
        with st.spinner("Computing response regions..."):
            adapter = _load_context_adapter(context)
            response_result = build_response_region_visualization(
                model=adapter,
                clean_tensor=preview_result.clean_tensor,
                adversarial_tensor=preview_result.adversarial_tensor,
                clean_image=preview_result.clean_image,
                adversarial_image=preview_result.adversarial_image,
                class_id=int(class_id),
                threshold_percentile=int(threshold_percentile),
                heatmap_scale_mode=context["heatmap_scale_mode"],
                heatmap_percentile_clip_upper=context["heatmap_percentile_clip_upper"],
            )

        st.session_state["response_region_result"] = response_result
        st.session_state["response_region_signature"] = response_signature

    response_result = st.session_state.get("response_region_result")
    if response_result is None:
        return

    if response_result.clean_used_fallback or response_result.adversarial_used_fallback:
        st.warning("当前响应区域分析至少有一侧使用了 fallback 目标分数：当目标类别在当前预测中不存在时，会退化为整图平均 logit。")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Clean Mean", f"{response_result.clean_mean:.4f}")
    metric_columns[1].metric("Adv Mean", f"{response_result.adversarial_mean:.4f}")
    metric_columns[2].metric("Diff Mean", f"{response_result.diff_mean:.4f}")
    metric_columns[3].metric("Clean Area", f"{response_result.clean_active_ratio * 100.0:.1f}%")
    metric_columns[4].metric("Adv Area", f"{response_result.adversarial_active_ratio * 100.0:.1f}%")
    metric_columns[5].metric("Overlap IoU", f"{response_result.overlap_iou:.3f}")

    st.caption(
        "\n".join(
            [
                f"Heatmap display: {heatmap_display_note(context['heatmap_scale_mode'], context['heatmap_percentile_clip_upper'])}",
                f"Clean target pixels: {response_result.clean_target_pixels} | score={response_result.clean_score:.4f}",
                (
                    f"Adv target pixels: {response_result.adversarial_target_pixels} "
                    f"| score={response_result.adversarial_score:.4f}"
                ),
                (
                    f"Threshold percentile: {response_result.threshold_percentile} "
                    f"| clean_peak={response_result.clean_peak:.4f} "
                    f"| adv_peak={response_result.adversarial_peak:.4f}"
                ),
            ]
        )
    )

    heatmap_columns = st.columns(3)
    heatmap_columns[0].image(
        response_result.clean_overlay,
        caption="Clean Response Normalized Heatmap",
        use_container_width=True,
    )
    heatmap_columns[1].image(
        response_result.adversarial_overlay,
        caption="Adversarial Response Normalized Heatmap",
        use_container_width=True,
    )
    heatmap_columns[2].image(
        response_result.diff_overlay,
        caption="Response Diff Normalized Heatmap",
        use_container_width=True,
    )

    region_columns = st.columns(3)
    region_columns[0].image(
        response_result.clean_region_overlay,
        caption="Clean High-Response Region",
        use_container_width=True,
    )
    region_columns[1].image(
        response_result.adversarial_region_overlay,
        caption="Adversarial High-Response Region",
        use_container_width=True,
    )
    region_columns[2].image(
        response_result.overlap_region_overlay,
        caption="Stable Overlap Region",
        use_container_width=True,
    )

    if st.button("导出当前响应区域分析", use_container_width=True, key="response_region_export_button"):
        figures_dir, tables_dir, reports_dir = _export_response_region_preview(
            context,
            response_result,
            class_names.get(int(class_id), f"class_{class_id}"),
        )
        st.success(
            "导出完成："
            f"\nfigures={figures_dir.resolve()}"
            f"\ntables={tables_dir.resolve()}"
            f"\nreports={reports_dir.resolve()}"
        )


def main() -> None:
    st.set_page_config(page_title="语义分割分析工具", layout="wide")
    st.title("语义分割分析工具")
    st.caption("数据集扫描、类别统计、语义分割预览、对抗特征可视化、CAM 预览与响应区域分析。")

    dataset_config_path = st.sidebar.text_input(
        "Dataset config",
        "configs/datasets/example.yaml",
        key="global_dataset_config_path",
    )
    label_config_path = st.sidebar.text_input(
        "Label config",
        "configs/labels/example.yaml",
        key="global_label_config_path",
    )
    default_image_dir, default_mask_dir, _, _ = _resolve_dataset_inputs(dataset_config_path, "datasets/images", "datasets/masks")
    analysis_context = _prepare_shared_analysis_preview()

    scan_tab, preview_tab, adversarial_tab, cam_tab, response_tab = st.tabs(
        ["Dataset Scan", "Triplet Preview", "Adversarial Feature Preview", "CAM Preview", "Response Region Analysis"]
    )

    with scan_tab:
        image_dir = st.text_input("Image directory", str(default_image_dir))
        mask_dir = st.text_input("Mask directory", str(default_mask_dir))
        if st.button("Run Dataset Scan", use_container_width=True):
            _run_dataset_scan(dataset_config_path, label_config_path, image_dir, mask_dir)

    with preview_tab:
        preview_mode = st.radio(
            "Preview mode",
            options=["Dataset browser", "Pascal VOC demo", "Cityscapes demo", "ADE20K demo", "Manual paths"],
            horizontal=True,
        )
        if preview_mode == "Dataset browser":
            _render_dataset_triplet_preview(
                dataset_config_path=dataset_config_path,
                label_config_path=label_config_path,
                image_dir_text=str(default_image_dir),
                mask_dir_text=str(default_mask_dir),
            )
        elif preview_mode == "Pascal VOC demo":
            _render_pascal_voc_triplet_preview(label_config_path)
        elif preview_mode == "Cityscapes demo":
            _render_cityscapes_triplet_preview()
        elif preview_mode == "ADE20K demo":
            _render_ade20k_triplet_preview()
        else:
            image_path = st.text_input("Image path", "")
            gt_path = st.text_input("GT mask path", "")
            pred_path = st.text_input("Prediction mask path", "")
            alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
            show_legend = st.checkbox("Show class legend", value=True)
            if st.button("Render Triplet", use_container_width=True):
                _run_triplet_preview(label_config_path, image_path, gt_path, pred_path, alpha, show_legend)

    with adversarial_tab:
        _render_adversarial_feature_preview(label_config_path, analysis_context)

    with cam_tab:
        _render_cam_preview(label_config_path, analysis_context)

    with response_tab:
        _render_response_region_preview(label_config_path, analysis_context)


if __name__ == "__main__":
    main()
