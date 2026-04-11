from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.apps.adversarial_preview import (
    build_layer_visualization,
    discover_attack_config_options,
    discover_checkpoint_options,
    generate_feature_preview,
)
from src.apps.dashboard import build_overview_cards
from src.attacks import AttackConfig
from src.common.config import load_dataset_config, load_label_config
from src.common.config import load_yaml
from src.datasets import discover_pascal_voc_samples
from src.datasets.scanner import scan_dataset
from src.datasets.stats import compute_class_statistics
from src.datasets.voc import PascalVOCValidationDataset
from src.io.image_io import DEFAULT_IMAGE_SUFFIXES, DEFAULT_MASK_SUFFIXES
from src.models import MODEL_FAMILY_CHOICES, TorchSegmentationModelAdapter, build_model_from_checkpoint
from src.visualization.triplet import discover_triplet_samples, overlay_mask, render_triplet_from_paths


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


@st.cache_resource(show_spinner=False)
def _load_voc_validation_dataset(dataset_root: str) -> PascalVOCValidationDataset:
    return PascalVOCValidationDataset(dataset_root, split="val", resize_short=473, crop_size=473)


@st.cache_resource(show_spinner=False)
def _load_model_adapter(
    family: str,
    checkpoint_path: str,
    num_classes: int,
    device: str,
    strict: bool,
) -> tuple[TorchSegmentationModelAdapter, tuple[str, ...], tuple[str, ...]]:
    model, missing_keys, unexpected_keys = build_model_from_checkpoint(
        family=family,
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        map_location="cpu",
        strict=strict,
    )
    return (
        TorchSegmentationModelAdapter(model=model, num_classes=num_classes, device=device),
        tuple(missing_keys),
        tuple(unexpected_keys),
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
    label_config_path = str(voc_label_config_path) if voc_label_config_path.exists() else default_label_config_path

    dataset_root = st.text_input("VOC dataset root", "datasets")
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


def _render_prediction_overlays(result, label_config_path: str) -> None:
    label_config = _optional_label_config(label_config_path)
    palette = label_config.palette if label_config else None
    ignore_index = label_config.ignore_index if label_config else None

    columns = st.columns(3)
    panels = (
        ("Ground Truth", overlay_mask(result.clean_image, result.ground_truth, palette=palette, ignore_index=ignore_index)),
        (
            "Clean Prediction",
            overlay_mask(result.clean_image, result.clean_prediction, palette=palette, ignore_index=ignore_index),
        ),
        (
            "Adversarial Prediction",
            overlay_mask(result.adversarial_image, result.adversarial_prediction, palette=palette, ignore_index=ignore_index),
        ),
    )
    for column, (title, image) in zip(columns, panels, strict=True):
        column.image(image, caption=title, use_container_width=True)


def _render_adversarial_feature_preview(label_config_path: str) -> None:
    st.caption("基于 Pascal VOC val 单样本执行攻击，并在前端查看逐层特征变化。")

    dataset_root = st.text_input("VOC dataset root", "datasets", key="adv_dataset_root")
    try:
        dataset = _load_voc_validation_dataset(dataset_root.strip())
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        return

    checkpoint_options = discover_checkpoint_options()
    attack_options = discover_attack_config_options()
    attack_names = sorted({option.attack_name for option in attack_options})
    if not attack_names:
        st.error("No attack configs with valid `name` were found under configs/attacks.")
        return

    control_left, control_right = st.columns(2)
    with control_left:
        sample_index = st.selectbox(
            "样本",
            options=range(len(dataset.sample_ids)),
            format_func=lambda index: f"{index:04d} | {dataset.sample_ids[index]}",
            key="adv_sample_index",
        )
        family = st.selectbox("模型 family", MODEL_FAMILY_CHOICES, key="adv_model_family")
        family_checkpoints = [option for option in checkpoint_options if option.family == family]
        if family_checkpoints:
            checkpoint_choice = st.selectbox(
                "模型 checkpoint",
                options=range(len(family_checkpoints)),
                format_func=lambda index: family_checkpoints[index].label,
                key="adv_checkpoint_choice",
            )
            checkpoint_path = str(family_checkpoints[checkpoint_choice].path)
            st.caption(f"Checkpoint path: {checkpoint_path}")
        else:
            st.warning("No checkpoint was auto-discovered for this family. Enter a checkpoint path manually.")
            checkpoint_path = st.text_input("Checkpoint path", "", key="adv_checkpoint_path")

    with control_right:
        attack_name = st.selectbox("攻击", attack_names, key="adv_attack_name")
        matched_attack_configs = [option for option in attack_options if option.attack_name == attack_name]
        attack_config_index = st.selectbox(
            "攻击配置",
            options=range(len(matched_attack_configs)),
            format_func=lambda index: matched_attack_configs[index].label,
            key="adv_attack_config_index",
        )
        attack_config_path = matched_attack_configs[attack_config_index].path
        base_attack_config = AttackConfig.from_dict(load_yaml(attack_config_path))
        default_radius_255 = int(round(base_attack_config.epsilon * 255.0))
        if st.session_state.get("adv_radius_config_path") != str(attack_config_path):
            st.session_state["adv_radius_config_path"] = str(attack_config_path)
            st.session_state["adv_radius_255"] = default_radius_255
        radius_255 = st.slider("扰动半径 (0-255)", min_value=0, max_value=255, step=1, key="adv_radius_255")
        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Device", device_options, key="adv_device")
        strict = st.checkbox("Strict checkpoint load", value=True, key="adv_strict")

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
        "attack_config_path": str(attack_config_path),
        "radius_255": int(radius_255),
        "device": device,
        "strict": bool(strict),
    }

    if st.button("运行对抗特征可视化", use_container_width=True, key="adv_run_button"):
        checkpoint_candidate = Path(checkpoint_path.strip())
        if not checkpoint_path.strip():
            st.error("Checkpoint path is required.")
            return
        if not checkpoint_candidate.exists():
            st.error("Checkpoint path does not exist.")
            return

        with st.spinner("Loading model and generating feature preview..."):
            try:
                adapter, missing_keys, unexpected_keys = _load_model_adapter(
                    family=family,
                    checkpoint_path=str(checkpoint_candidate),
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
                return

        st.session_state["adv_preview_result"] = preview_result
        st.session_state["adv_preview_signature"] = current_signature
        st.session_state["adv_preview_checkpoint_info"] = {
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "checkpoint_path": str(checkpoint_candidate.resolve()),
            "family": family,
        }

    preview_result = st.session_state.get("adv_preview_result")
    if preview_result is None:
        return

    stored_signature = st.session_state.get("adv_preview_signature", {})
    if stored_signature != current_signature:
        st.info("当前展示的是上一次运行结果。修改模型、攻击或样本后，需要重新点击按钮更新。")

    checkpoint_info = st.session_state.get("adv_preview_checkpoint_info", {})
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
                f"Missing keys: {len(checkpoint_info.get('missing_keys', []))}",
                f"Unexpected keys: {len(checkpoint_info.get('unexpected_keys', []))}",
            ]
        )
    )

    input_columns = st.columns(4)
    input_columns[0].image(preview_result.clean_image, caption="Clean Input", use_container_width=True)
    input_columns[1].image(preview_result.adversarial_image, caption="Adversarial Input", use_container_width=True)
    input_columns[2].image(preview_result.perturbation_image, caption="Perturbation", use_container_width=True)
    input_columns[3].image(
        preview_result.sample_delta_heatmap,
        caption=(
            f"Sample Delta Heatmap | mean={preview_result.sample_delta_mean:.4f} "
            f"max={preview_result.sample_delta_max:.4f}"
        ),
        use_container_width=True,
    )

    _render_prediction_overlays(preview_result, label_config_path)

    if not preview_result.layer_names:
        st.warning("The selected model did not return any feature layers for visualization.")
        return

    selected_layer_index = st.slider(
        "层编号",
        min_value=0,
        max_value=len(preview_result.layer_names) - 1,
        value=min(st.session_state.get("adv_layer_index", 0), len(preview_result.layer_names) - 1),
        key="adv_layer_index",
    )
    layer_name = preview_result.layer_names[selected_layer_index]
    layer_visualization = build_layer_visualization(preview_result, layer_name)
    st.caption(f"当前层 {selected_layer_index + 1}/{len(preview_result.layer_names)}: {layer_name}")

    feature_columns = st.columns(3)
    feature_columns[0].image(
        layer_visualization["clean_image"],
        caption=f"Clean Feature | shape={layer_visualization['clean_shape']}",
        use_container_width=True,
    )
    feature_columns[1].image(
        layer_visualization["adversarial_image"],
        caption=f"Adversarial Feature | shape={layer_visualization['adversarial_shape']}",
        use_container_width=True,
    )
    feature_columns[2].image(
        layer_visualization["diff_image"],
        caption=(
            f"Abs Diff | mean={layer_visualization['mean_abs_diff']:.4f} "
            f"max={layer_visualization['max_abs_diff']:.4f}"
        ),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="语义分割分析工具", layout="wide")
    st.title("语义分割分析工具")
    st.caption("数据集扫描、类别统计、语义分割预览，以及对抗攻击下的逐层特征可视化。")

    dataset_config_path = st.sidebar.text_input("Dataset config", "configs/datasets/example.yaml")
    label_config_path = st.sidebar.text_input("Label config", "configs/labels/example.yaml")
    default_image_dir, default_mask_dir, _, _ = _resolve_dataset_inputs(dataset_config_path, "datasets/images", "datasets/masks")

    scan_tab, preview_tab, adversarial_tab = st.tabs(["Dataset Scan", "Triplet Preview", "Adversarial Feature Preview"])

    with scan_tab:
        image_dir = st.text_input("Image directory", str(default_image_dir))
        mask_dir = st.text_input("Mask directory", str(default_mask_dir))
        if st.button("Run Dataset Scan", use_container_width=True):
            _run_dataset_scan(dataset_config_path, label_config_path, image_dir, mask_dir)

    with preview_tab:
        preview_mode = st.radio(
            "Preview mode",
            options=["Dataset browser", "Pascal VOC demo", "Manual paths"],
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
        else:
            image_path = st.text_input("Image path", "")
            gt_path = st.text_input("GT mask path", "")
            pred_path = st.text_input("Prediction mask path", "")
            alpha = st.slider("Overlay alpha", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
            show_legend = st.checkbox("Show class legend", value=True)
            if st.button("Render Triplet", use_container_width=True):
                _run_triplet_preview(label_config_path, image_path, gt_path, pred_path, alpha, show_legend)

    with adversarial_tab:
        _render_adversarial_feature_preview(label_config_path)


if __name__ == "__main__":
    main()
