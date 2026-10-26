"""
Configuration loader with YAML inheritance.

Task-specific configs inherit from base.yaml. Any key present in the task
config overrides the base value; all other base keys are preserved.

Usage:
    from src.utils.config import load_config
    cfg = load_config("sentiment")  # Loads base.yaml + sentiment.yaml
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Module-level default: config directory relative to project root
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (non-destructive).

    - Dict values are merged recursively.
    - All other types in *override* replace those in *base*.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public API for deep-merging two config dicts."""
    return _deep_merge(base, override)


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Load a single YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if data is not None else {}


def load_config(
    task_name: Optional[str] = None,
    config_dir: Optional[Path | str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and merge configuration for a given task.

    Parameters
    ----------
    task_name : str, optional
        Name of the task (e.g. "sentiment", "ner"). If ``None``, only
        ``base.yaml`` is loaded.
    config_dir : Path or str, optional
        Directory containing the YAML config files. Defaults to
        ``<project_root>/config/``.
    overrides : dict, optional
        Programmatic overrides applied on top of the merged config.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    config_dir = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    # 1. Load base config
    base_cfg = load_yaml(config_dir / "base.yaml")

    # 2. Optionally layer task-specific config
    if task_name:
        task_path = config_dir / f"{task_name}.yaml"
        if task_path.exists():
            task_cfg = load_yaml(task_path)
            base_cfg = _deep_merge(base_cfg, task_cfg)
        else:
            raise FileNotFoundError(
                f"Task config '{task_name}.yaml' not found in {config_dir}"
            )

    # 3. Apply programmatic overrides
    if overrides:
        base_cfg = _deep_merge(base_cfg, overrides)

    return base_cfg


def get_project_root() -> Path:
    """Return the project root directory (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve a path from config relative to the project root.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (must contain ``project`` section).
    key : str
        Key inside ``cfg["project"]`` to resolve.

    Returns
    -------
    Path
        Absolute path.
    """
    root = get_project_root()
    rel = cfg.get("project", {}).get(key, "")
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    return path
