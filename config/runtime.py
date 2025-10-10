from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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


def _resolve_optional_path(env_var: str, default_relative: str) -> Optional[Path]:
    value = os.getenv(env_var)
    candidates = []
    if value:
        candidates.append(Path(value).expanduser().resolve())
    if default_relative:
        candidates.append((PROJECT_ROOT / default_relative).resolve())
        candidates.append((PROJECT_ROOT / "GroundingDINO" / default_relative).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def _resolve_optional_str(env_var: str) -> Optional[str]:
    raw = os.getenv(env_var)
    if raw is None:
        return None
    value = raw.strip()
    return value or None

def _resolve_class_names(env_var: str) -> Optional[List[str]]:
    raw = os.getenv(env_var)
    if not raw:
        return None
    candidate_path = Path(raw).expanduser()
    if candidate_path.exists():
        try:
            return [
                line.strip()
                for line in candidate_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            return None
    return [chunk.strip() for chunk in raw.split(",") if chunk.strip()]


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
    default_detection_model: str
    omdet_model_id: Optional[str]
    omdet_weights_path: Optional[Path]
    omdet_device: str
    omdet_confidence_threshold: float
    omdet_class_names: Optional[List[str]]


def get_settings() -> RuntimeSettings:
    device_env = os.getenv("GDINO_DEVICE")
    omdet_device_env = os.getenv("OMDET_DEVICE")
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
        default_detection_model=os.getenv("DETECTION_DEFAULT_MODEL", "grounding_dino"),
        omdet_model_id=_resolve_optional_str("OMDET_MODEL_ID") or "omlab/omdet-turbo-swin-tiny-hf",
        omdet_weights_path=_resolve_optional_path(
            "OMDET_WEIGHTS_PATH",
            "weights/omdet_turbo.pth",
        ),
        omdet_device=_resolve_device(omdet_device_env),
        omdet_confidence_threshold=_resolve_float("OMDET_CONFIDENCE_THRESHOLD", 0.3),
        omdet_class_names=_resolve_class_names("OMDET_CLASS_NAMES"),
    )
