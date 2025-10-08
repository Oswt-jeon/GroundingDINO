from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependencies import get_detection_service_dependency
from api.schemas.detections import DetectItem
from src.services.detection_service import DetectionService


router = APIRouter()


@router.get("/healthz")
def health_check():
    return {"ok": True}


@router.post("/detect", response_model=List[DetectItem])
async def detect(
    file: UploadFile = File(...),
    text: str = Form(...),
    box_threshold: Optional[float] = Form(None),
    text_threshold: Optional[float] = Form(None),
    detection_service: DetectionService = Depends(get_detection_service_dependency),
) -> List[DetectItem]:
    payload = await file.read()
    result = detection_service.detect_from_bytes(
        data=payload,
        filename=file.filename,
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )
    return [DetectItem.from_domain(item) for item in result.items]

