from __future__ import annotations

from functools import lru_cache

from config.runtime import get_settings
from src.adapters.grounding_dino import GroundingDinoModelAdapter
from src.services.detection_service import DetectionService


def create_detection_service() -> DetectionService:
    settings = get_settings()
    adapter = GroundingDinoModelAdapter(
        config_path=settings.model_config_path,
        weights_path=settings.weights_path,
        device=settings.device,
    )
    return DetectionService(
        model_adapter=adapter,
        images_dir=settings.images_dir,
        results_dir=settings.results_dir,
        default_box_threshold=settings.box_threshold,
        default_text_threshold=settings.text_threshold,
        annotate_results=settings.annotate_results,
    )


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    return create_detection_service()

