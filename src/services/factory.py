from __future__ import annotations

from functools import lru_cache
from typing import Dict

from config.runtime import RuntimeSettings, get_settings
from src.adapters.grounding_dino import GroundingDinoModelAdapter
from src.adapters.omdet_turbo import OmDetTurboModelAdapter
from src.services.detection_service import DetectionService
from src.services.manager import DetectionServiceManager


def _build_grounding_dino_service(settings: RuntimeSettings) -> DetectionService:
    adapter = GroundingDinoModelAdapter(
        config_path=settings.model_config_path,
        weights_path=settings.weights_path,
        device=settings.device,
    )
    return DetectionService(
        model_adapter=adapter,
        model_name="grounding_dino",
        images_dir=settings.images_dir,
        results_dir=settings.results_dir,
        search_dir=settings.search_dir,
        default_box_threshold=settings.box_threshold,
        default_text_threshold=settings.text_threshold,
        annotate_results=settings.annotate_results,
    )


def _maybe_build_omdet_turbo_service(
    settings: RuntimeSettings,
) -> DetectionService | None:
    if settings.omdet_model_id is None and settings.omdet_weights_path is None:
        return None
    adapter = OmDetTurboModelAdapter(
        model_id=settings.omdet_model_id,
        weights_path=settings.omdet_weights_path,
        device=settings.omdet_device,
        confidence_threshold=settings.omdet_confidence_threshold,
        class_names=settings.omdet_class_names,
    )
    return DetectionService(
        model_adapter=adapter,
        model_name="omdet_turbo",
        images_dir=settings.images_dir,
        results_dir=settings.results_dir,
        search_dir=settings.search_dir,
        default_box_threshold=settings.box_threshold,
        default_text_threshold=settings.text_threshold,
        annotate_results=settings.annotate_results,
    )


def create_detection_manager() -> DetectionServiceManager:
    settings = get_settings()
    services: Dict[str, DetectionService] = {
        "grounding_dino": _build_grounding_dino_service(settings),
    }

    omdet_service = _maybe_build_omdet_turbo_service(settings)
    if omdet_service is not None:
        services["omdet_turbo"] = omdet_service

    aliases = {
        "grounding_dino": ("groundingdino", "gdino"),
        "omdet_turbo": ("omdet", "omdetturbo", "turbo"),
    }

    default_model = settings.default_detection_model
    if default_model not in services:
        default_model = "grounding_dino"

    return DetectionServiceManager(
        services=services,
        default_model=default_model,
        aliases=aliases,
    )


@lru_cache(maxsize=1)
def get_detection_manager() -> DetectionServiceManager:
    return create_detection_manager()


def create_detection_service() -> DetectionService:
    manager = create_detection_manager()
    return manager.resolve(manager.default_model)


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    return create_detection_service()
