from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.common.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an adversarial attack against a segmentation model.")
    parser.add_argument("--attack-config", required=True, help="Path to the attack YAML config.")
    parser.add_argument("--model-adapter", default="", help="Registered model adapter name.")
    parser.add_argument("--input-dir", default="", help="Input image directory.")
    parser.add_argument("--mask-dir", default="", help="Ground-truth mask directory.")
    parser.add_argument("--output-dir", default="results/reports/attacks", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.attack_config)
    print("Attack scaffold ready.")
    print(f"attack_config: {args.attack_config}")
    print(f"attack_name: {config.get('name', '<missing>')}")
    print(f"model_adapter: {args.model_adapter or '<not provided>'}")
    print("Next step: bind a concrete model adapter and batch loader.")


if __name__ == "__main__":
    main()
