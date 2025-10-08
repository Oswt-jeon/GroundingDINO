from __future__ import annotations

from fastapi import FastAPI, Request

from src.services.detection_service import DetectionService
from src.services.factory import get_detection_service


def register_dependencies(app: FastAPI) -> None:
    app.state.detection_service = get_detection_service()


def get_detection_service_dependency(request: Request) -> DetectionService:
    service = getattr(request.app.state, "detection_service", None)
    if service is None:
        service = get_detection_service()
        request.app.state.detection_service = service
    return service

