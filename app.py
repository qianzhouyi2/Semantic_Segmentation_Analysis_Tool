from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.apps.dashboard import build_overview_cards
from src.common.config import load_dataset_config, load_label_config
from src.datasets.scanner import scan_dataset
from src.datasets.stats import compute_class_statistics
from src.visualization.triplet import render_triplet_from_paths


def _optional_label_config(path_text: str):
    candidate = Path(path_text)
    if path_text and candidate.exists():
        return load_label_config(candidate)
    return None


def _run_dataset_scan(dataset_config_path: str, label_config_path: str, image_dir_text: str, mask_dir_text: str):
    image_dir = Path(image_dir_text)
    mask_dir = Path(mask_dir_text)

    if dataset_config_path and Path(dataset_config_path).exists():
        dataset_config = load_dataset_config(dataset_config_path)
        image_dir = dataset_config.image_dir
        mask_dir = dataset_config.mask_dir
        image_suffixes = dataset_config.image_suffixes
        mask_suffixes = dataset_config.mask_suffixes
    else:
        image_suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        mask_suffixes = (".png", ".bmp", ".tif", ".tiff")

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


def _run_triplet_preview(label_config_path: str, image_path_text: str, gt_path_text: str, pred_path_text: str):
    image_path = Path(image_path_text)
    if not image_path.exists():
        st.error("Image path does not exist.")
        return

    gt_path = Path(gt_path_text) if gt_path_text else None
    pred_path = Path(pred_path_text) if pred_path_text else None
    palette = None
    ignore_index = None
    if label_config_path and Path(label_config_path).exists():
        label_config = load_label_config(label_config_path)
        palette = label_config.palette
        ignore_index = label_config.ignore_index

    figure = render_triplet_from_paths(
        image_path=image_path,
        ground_truth_path=gt_path if gt_path and gt_path.exists() else None,
        prediction_path=pred_path if pred_path and pred_path.exists() else None,
        palette=palette,
        ignore_index=ignore_index,
    )
    st.pyplot(figure, clear_figure=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="语义分割分析工具", layout="wide")
    st.title("语义分割分析工具")
    st.caption("MVP skeleton: dataset scan, class statistics, and triplet visualization.")

    dataset_config_path = st.sidebar.text_input("Dataset config", "configs/datasets/example.yaml")
    label_config_path = st.sidebar.text_input("Label config", "configs/labels/example.yaml")

    scan_tab, preview_tab = st.tabs(["Dataset Scan", "Triplet Preview"])

    with scan_tab:
        image_dir = st.text_input("Image directory", "datasets/images")
        mask_dir = st.text_input("Mask directory", "datasets/masks")
        if st.button("Run Dataset Scan", use_container_width=True):
            _run_dataset_scan(dataset_config_path, label_config_path, image_dir, mask_dir)

    with preview_tab:
        image_path = st.text_input("Image path", "")
        gt_path = st.text_input("GT mask path", "")
        pred_path = st.text_input("Prediction mask path", "")
        if st.button("Render Triplet", use_container_width=True):
            _run_triplet_preview(label_config_path, image_path, gt_path, pred_path)


if __name__ == "__main__":
    main()
