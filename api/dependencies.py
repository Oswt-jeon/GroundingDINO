from __future__ import annotations

from fastapi import FastAPI, Request

from src.services.factory import get_detection_manager
from src.services.manager import DetectionServiceManager


def register_dependencies(app: FastAPI) -> None:
    app.state.detection_manager = get_detection_manager()


def get_detection_manager_dependency(request: Request) -> DetectionServiceManager:
    manager: DetectionServiceManager = getattr(
        request.app.state,
        "detection_manager",
        None,
    )
    if manager is None:
        manager = get_detection_manager()
        request.app.state.detection_manager = manager
    return manager
