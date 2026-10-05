"""Configuration loading helpers (YAML -> dict) with project-relative path resolution."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into a deep copy of ``base``.

    Args:
        base: Original dictionary.
        overrides: Values that take precedence.

    Returns:
        A new merged dictionary.
    """
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load ``config.yaml``.

    Args:
        path: Optional path to a YAML file (defaults to the project's config.yaml).
        overrides: Optional nested dict merged on top of the file contents.

    Returns:
        The configuration dictionary.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def resolve_path(p: str) -> Path:
    """Resolve a (possibly relative) config path against the project root.

    Args:
        p: Path string from the config.

    Returns:
        Absolute :class:`pathlib.Path`.
    """
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def smoke_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a tiny CPU-friendly config for smoke tests (synthetic data, no downloads).

    Args:
        overrides: Extra nested overrides applied last.

    Returns:
        Configuration dictionary with sizes shrunk for fast tests.
    """
    cfg = load_config()
    s = cfg["smoke"]
    small = {
        "data": {"image_size": s["image_size"], "num_workers": 0},
        "model": {"pretrained": s["pretrained"]},
        "student": {"pretrained": s["pretrained"]},
        "train": {"epochs": s["epochs"], "batch_size": s["batch_size"], "amp": False, "log_every": 1},
        "distill": {"epochs": 1},
        "loss": {"evidential_kl_anneal_epochs": 2},
        "paths": {"output_dir": "./outputs/smoke", "checkpoint_dir": "./outputs/smoke/checkpoints",
                  "report_dir": "./outputs/smoke/reports"},
    }
    cfg = deep_update(cfg, small)
    return deep_update(cfg, overrides) if overrides else cfg
