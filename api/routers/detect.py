from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependencies import get_detection_service_dependency
from api.schemas.detections import (
    AnnotatedImageResponse,
    DetectItem,
    ImageSearchResult,
    SearchRequest,
    SearchResponse,
)
from src.services.detection_service import DetectionService
from src.utils.file_io import encode_file_to_base64


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


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    detection_service: DetectionService = Depends(get_detection_service_dependency),
) -> SearchResponse:
    results = detection_service.detect_in_directory(
        caption=request.text,
        directory=None,
        patterns=request.patterns,
        box_threshold=request.box_threshold,
        text_threshold=request.text_threshold,
        limit=request.limit,
        only_with_detections=True,
    )

    response_items = []
    for payload in results:
        annotated = None
        if payload.annotated_path and payload.annotated_path.exists():
            try:
                data, mime = encode_file_to_base64(payload.annotated_path)
                annotated = AnnotatedImageResponse(data=data, mime_type=mime)
            except FileNotFoundError:
                annotated = None
        response_items.append(
            ImageSearchResult(
                image=str(payload.source_path),
                detections=[DetectItem.from_domain(item) for item in payload.items],
                annotated_image=annotated,
            )
        )

    return SearchResponse(results=response_items)
