"""Load config.yaml once so every module reads the same settings."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the YAML config and check the benchmark weights sum to 1."""
    with open(path) as f:
        config = yaml.safe_load(f)

    weights = [c["weight"] for c in config["benchmark"]["components"]]
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError(f"Benchmark weights must sum to 1.0, got {sum(weights)}")

    return config
