from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(env_var: str, default_relative: str) -> Path:
    value = os.getenv(env_var)
    if value:
        return Path(value).expanduser().resolve()

    primary = (PROJECT_ROOT / default_relative).resolve()
    if primary.exists():
        return primary

    alternative = (PROJECT_ROOT / "GroundingDINO" / default_relative).resolve()
    if alternative.exists():
        return alternative

    return primary


def _resolve_float(env_var: str, default: float) -> float:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve_bool(env_var: str, default: bool) -> bool:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "t", "yes", "y"}


def _resolve_device(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class RuntimeSettings:
    model_config_path: Path
    weights_path: Path
    device: str
    box_threshold: float
    text_threshold: float
    images_dir: Path
    results_dir: Path
    annotate_results: bool
    search_dir: Path


def get_settings() -> RuntimeSettings:
    device_env = os.getenv("GDINO_DEVICE")
    return RuntimeSettings(
        model_config_path=_resolve_path(
            "GDINO_MODEL_CONFIG",
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        weights_path=_resolve_path(
            "GDINO_WEIGHTS_PATH",
            "weights/groundingdino_swint_ogc.pth",
        ),
        device=_resolve_device(device_env),
        box_threshold=_resolve_float("GDINO_BOX_THRESHOLD", 0.25),
        text_threshold=_resolve_float("GDINO_TEXT_THRESHOLD", 0.25),
        images_dir=_resolve_path("GDINO_IMAGES_DIR", "data/images"),
        results_dir=_resolve_path("GDINO_RESULTS_DIR", "data/results"),
        annotate_results=_resolve_bool("GDINO_ANNOTATE_RESULTS", True),
        search_dir=_resolve_path("GDINO_SEARCH_DIR", "data/gallery"),
    )
