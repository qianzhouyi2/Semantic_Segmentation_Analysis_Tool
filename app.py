from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.apps.adversarial_preview import (
    build_layer_visualization,
    discover_attack_config_options,
    discover_checkpoint_options,
    discover_defense_config_options,
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
from src.visualization.cam import build_cam_visualization, select_representative_cam_feature_keys
from src.visualization.response_region import build_response_region_visualization
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


def _resolve_pascal_voc_label_config_path(default_label_config_path: str) -> str:
    _, voc_label_config_path = _default_pascal_voc_config_paths()
    if voc_label_config_path.exists():
        return str(voc_label_config_path)
    return default_label_config_path


@st.cache_resource(show_spinner=False)
def _load_voc_validation_dataset(dataset_root: str) -> PascalVOCValidationDataset:
    return PascalVOCValidationDataset(dataset_root, split="val", resize_short=473, crop_size=473)


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


def _prepare_adversarial_preview(
    state_prefix: str,
    section_caption: str,
    run_button_label: str,
):
    st.caption(section_caption)

    dataset_root = st.text_input("VOC dataset root", "datasets", key=f"{state_prefix}_dataset_root")
    try:
        dataset = _load_voc_validation_dataset(dataset_root.strip())
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        return None, None, None, None, None

    checkpoint_options = discover_checkpoint_options()
    defense_options = discover_defense_config_options()
    attack_options = discover_attack_config_options()
    attack_names = sorted({option.attack_name for option in attack_options})
    if not attack_names:
        st.error("No attack configs with valid `name` were found under configs/attacks.")
        return None, None, None, None, None

    control_left, control_right = st.columns(2)
    with control_left:
        sample_index = st.selectbox(
            "样本",
            options=range(len(dataset.sample_ids)),
            format_func=lambda index: f"{index:04d} | {dataset.sample_ids[index]}",
            key=f"{state_prefix}_sample_index",
        )
        family = st.selectbox("模型 family", MODEL_FAMILY_CHOICES, key=f"{state_prefix}_model_family")
        family_checkpoints = [option for option in checkpoint_options if option.family == family]
        if family_checkpoints:
            checkpoint_choice = st.selectbox(
                "模型 checkpoint",
                options=range(len(family_checkpoints)),
                format_func=lambda index: family_checkpoints[index].label,
                key=f"{state_prefix}_checkpoint_choice",
            )
            checkpoint_path = str(family_checkpoints[checkpoint_choice].path)
            st.caption(f"Checkpoint path: {checkpoint_path}")
        else:
            st.warning("No checkpoint was auto-discovered for this family. Enter a checkpoint path manually.")
            checkpoint_path = st.text_input("Checkpoint path", "", key=f"{state_prefix}_checkpoint_path")

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
            key=f"{state_prefix}_defense_mode",
        )
        defense_config_path = ""
        if defense_mode == "<auto>":
            defense_choice = st.selectbox(
                "防御配置文件",
                options=range(len(family_defense_options)),
                format_func=lambda index: family_defense_options[index].label,
                key=f"{state_prefix}_defense_choice",
            )
            defense_config_path = str(family_defense_options[defense_choice].path)
            st.caption(f"Defense config: {defense_config_path}")
        elif defense_mode == "<manual>":
            defense_config_path = st.text_input("Defense config path", "", key=f"{state_prefix}_defense_config_path")
            if family_defense_options:
                st.caption(f"已自动发现 {len(family_defense_options)} 个与当前 family 兼容的防御配置。")
            else:
                st.caption("当前未自动发现与该 family 兼容的防御配置，请手动输入。")

    with control_right:
        attack_name = st.selectbox("攻击", attack_names, key=f"{state_prefix}_attack_name")
        matched_attack_configs = [option for option in attack_options if option.attack_name == attack_name]
        attack_config_index = st.selectbox(
            "攻击配置",
            options=range(len(matched_attack_configs)),
            format_func=lambda index: matched_attack_configs[index].label,
            key=f"{state_prefix}_attack_config_index",
        )
        attack_config_path = matched_attack_configs[attack_config_index].path
        base_attack_config = AttackConfig.from_dict(load_yaml(attack_config_path))
        default_radius_255 = int(round(base_attack_config.epsilon * 255.0))
        if st.session_state.get(f"{state_prefix}_radius_config_path") != str(attack_config_path):
            st.session_state[f"{state_prefix}_radius_config_path"] = str(attack_config_path)
            st.session_state[f"{state_prefix}_radius_255"] = default_radius_255
        radius_255 = st.slider(
            "扰动半径 (0-255)",
            min_value=0,
            max_value=255,
            step=1,
            key=f"{state_prefix}_radius_255",
        )
        device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        device = st.selectbox("Device", device_options, key=f"{state_prefix}_device")
        strict = st.checkbox("Strict checkpoint load", value=True, key=f"{state_prefix}_strict")

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

    if st.button(run_button_label, use_container_width=True, key=f"{state_prefix}_run_button"):
        checkpoint_candidate = Path(checkpoint_path.strip())
        if not checkpoint_path.strip():
            st.error("Checkpoint path is required.")
            return None, None, None, None, None
        if not checkpoint_candidate.exists():
            st.error("Checkpoint path does not exist.")
            return None, None, None, None, None
        defense_config_candidate = None
        if defense_config_path.strip():
            defense_config_candidate = Path(defense_config_path.strip())
            if not defense_config_candidate.exists():
                st.error("Defense config path does not exist.")
                return None, None, None, None, None

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
                return None, None, None, None, None

        st.session_state[f"{state_prefix}_preview_result"] = preview_result
        st.session_state[f"{state_prefix}_preview_signature"] = current_signature
        st.session_state[f"{state_prefix}_preview_checkpoint_info"] = {
            "missing_keys": list(missing_keys),
            "unexpected_keys": list(unexpected_keys),
            "checkpoint_path": str(checkpoint_candidate.resolve()),
            "defense_config_path": (
                str(defense_config_candidate.resolve()) if defense_config_candidate is not None else None
            ),
            "sparse_defense_info": sparse_defense_info,
            "family": family,
        }

    preview_result = st.session_state.get(f"{state_prefix}_preview_result")
    if preview_result is None:
        return None, None, None, None, None

    stored_signature = st.session_state.get(f"{state_prefix}_preview_signature", {})
    if stored_signature != current_signature:
        st.info("当前展示的是上一次运行结果。修改模型、攻击或样本后，需要重新点击按钮更新。")

    checkpoint_info = st.session_state.get(f"{state_prefix}_preview_checkpoint_info", {})
    return preview_result, checkpoint_info, stored_signature, family, checkpoint_path


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


def _render_preview_images(preview_result, label_config_path: str) -> None:
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


def _render_adversarial_feature_preview(label_config_path: str) -> None:
    resolved_label_config_path = _resolve_pascal_voc_label_config_path(label_config_path)
    preview_result, checkpoint_info, _stored_signature, family, checkpoint_path = _prepare_adversarial_preview(
        state_prefix="adv",
        section_caption="基于 Pascal VOC val 单样本执行攻击，并在前端查看逐层特征变化。",
        run_button_label="运行对抗特征可视化",
    )
    if preview_result is None:
        return

    _render_preview_summary(preview_result, checkpoint_info, family, checkpoint_path)
    _render_preview_images(preview_result, resolved_label_config_path)

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


def _render_cam_preview(label_config_path: str) -> None:
    resolved_label_config_path = _resolve_pascal_voc_label_config_path(label_config_path)
    preview_result, checkpoint_info, stored_signature, family, checkpoint_path = _prepare_adversarial_preview(
        state_prefix="cam_preview",
        section_caption="基于 Pascal VOC val 单样本执行攻击，并单独查看类激活图（CAM）。",
        run_button_label="运行 CAM 预览",
    )
    if preview_result is None:
        return

    _render_preview_summary(preview_result, checkpoint_info, family, checkpoint_path)
    _render_preview_images(preview_result, resolved_label_config_path)

    adapter, _, _, _ = _load_model_adapter(
        family=checkpoint_info.get("family", family),
        checkpoint_path=checkpoint_info.get("checkpoint_path", checkpoint_path),
        defense_config_path=str(checkpoint_info.get("defense_config_path") or ""),
        num_classes=21,
        device=stored_signature.get("device", "cpu"),
        strict=bool(stored_signature.get("strict", True)),
    )
    cam_feature_keys = select_representative_cam_feature_keys(adapter, preview_result.layer_names, max_keys=3)
    if not cam_feature_keys:
        st.info("CAM 当前只支持返回 4D 特征图的层。当前模型没有可用的 CAM 层。")
        return

    label_config = _optional_label_config(resolved_label_config_path)
    class_names = label_config.class_names if label_config else {}
    background_ids = label_config.background_ids if label_config else (0,)
    if class_names:
        cam_class_ids = list(label_config.class_ids)
    else:
        cam_class_ids = sorted(
            {
                *np.unique(preview_result.ground_truth).tolist(),
                *np.unique(preview_result.clean_prediction).tolist(),
                *np.unique(preview_result.adversarial_prediction).tolist(),
            }
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

    layer_labels = ["浅层", "中层", "深层"]
    if len(cam_results) == 1:
        layer_labels = ["可用层"]
    elif len(cam_results) == 2:
        layer_labels = ["浅层", "深层"]

    for layer_label, cam_result in zip(layer_labels, cam_results, strict=True):
        st.caption(f"{layer_label} CAM 层: {cam_result.feature_key}")
        st.caption("pred_pixels 表示当前预测中属于 CAM 目标类别的像素数，不是 CAM 热图的高响应像素数。")
        cam_columns = st.columns(3)
        cam_columns[0].image(
            cam_result.clean_overlay,
            caption=f"Clean CAM | mean={cam_result.clean_mean:.4f} pred_pixels={cam_result.clean_target_pixels}",
            use_container_width=True,
        )
        cam_columns[1].image(
            cam_result.adversarial_overlay,
            caption=f"Adv CAM | mean={cam_result.adversarial_mean:.4f} pred_pixels={cam_result.adversarial_target_pixels}",
            use_container_width=True,
        )
        cam_columns[2].image(
            cam_result.diff_image,
            caption=f"CAM Diff | mean={cam_result.diff_mean:.4f}",
            use_container_width=True,
        )


def _discover_response_class_ids(preview_result, label_config_path: str) -> tuple[list[int], dict[int, str], tuple[int, ...]]:
    label_config = _optional_label_config(label_config_path)
    class_names = label_config.class_names if label_config else {}
    background_ids = label_config.background_ids if label_config else (0,)
    if label_config is not None:
        class_ids = list(label_config.class_ids)
    else:
        class_ids = sorted(
            {
                *np.unique(preview_result.ground_truth).tolist(),
                *np.unique(preview_result.clean_prediction).tolist(),
                *np.unique(preview_result.adversarial_prediction).tolist(),
            }
        )
    return class_ids, class_names, background_ids


def _render_response_region_preview(label_config_path: str) -> None:
    resolved_label_config_path = _resolve_pascal_voc_label_config_path(label_config_path)
    preview_result, checkpoint_info, stored_signature, family, checkpoint_path = _prepare_adversarial_preview(
        state_prefix="response_region",
        section_caption="基于 Pascal VOC val 单样本执行攻击，并分析目标类别在输入空间的响应区域。",
        run_button_label="运行响应区域分析",
    )
    if preview_result is None:
        return

    _render_preview_summary(preview_result, checkpoint_info, family, checkpoint_path)
    _render_preview_images(preview_result, resolved_label_config_path)

    class_ids, class_names, background_ids = _discover_response_class_ids(preview_result, resolved_label_config_path)
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
            help="仅保留响应热图中高于该分位阈值的像素作为响应区域。",
        )

    st.caption(
        "响应得分定义为目标类别在当前预测区域上的平均 logit；如果该类别当前没有预测像素，则回退为整张图上的该类平均 logit。"
    )

    response_signature = {
        **stored_signature,
        "class_id": int(class_id),
        "threshold_percentile": int(threshold_percentile),
    }
    if st.session_state.get("response_region_signature") != response_signature:
        with st.spinner("Computing response regions..."):
            adapter, _, _, _ = _load_model_adapter(
                family=checkpoint_info.get("family", family),
                checkpoint_path=checkpoint_info.get("checkpoint_path", checkpoint_path),
                defense_config_path=str(checkpoint_info.get("defense_config_path") or ""),
                num_classes=21,
                device=stored_signature.get("device", "cpu"),
                strict=bool(stored_signature.get("strict", True)),
            )
            response_result = build_response_region_visualization(
                model=adapter,
                clean_tensor=preview_result.clean_tensor,
                adversarial_tensor=preview_result.adversarial_tensor,
                clean_image=preview_result.clean_image,
                adversarial_image=preview_result.adversarial_image,
                class_id=int(class_id),
                threshold_percentile=int(threshold_percentile),
            )

        st.session_state["response_region_result"] = response_result
        st.session_state["response_region_signature"] = response_signature

    response_result = st.session_state.get("response_region_result")
    if response_result is None:
        return

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
        caption="Clean Response Heatmap",
        use_container_width=True,
    )
    heatmap_columns[1].image(
        response_result.adversarial_overlay,
        caption="Adversarial Response Heatmap",
        use_container_width=True,
    )
    heatmap_columns[2].image(
        response_result.diff_overlay,
        caption="Response Diff Heatmap",
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

    with cam_tab:
        _render_cam_preview(label_config_path)

    with response_tab:
        _render_response_region_preview(label_config_path)


if __name__ == "__main__":
    main()
