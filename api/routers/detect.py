from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import get_detection_manager_dependency, get_db_session_dependency
from api.schemas.detections import (
    AnnotatedImageResponse,
    DetectItem,
    ImageSearchResult,
    SearchRequest,
    SearchResponse,
)
from src.db.database import Database
from src.repositories.owlv2_examples import OwlV2ExampleRepository
from src.services.manager import DetectionServiceManager
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
    model: Optional[str] = Form(None),
    detection_manager: DetectionServiceManager = Depends(get_detection_manager_dependency),
) -> List[DetectItem]:
    detection_service = detection_manager.resolve(model)
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
    detection_manager: DetectionServiceManager = Depends(get_detection_manager_dependency),
    session: Session = Depends(get_db_session_dependency),
) -> SearchResponse:
    detection_service = detection_manager.resolve(request.model)

    results_payload = None
    if getattr(detection_service, "model_name", None) == "owlv2":
        owns_session = False
        override_session = None
        database: Optional[Database] = None
        if request.database_url:
            database = Database.create(request.database_url)
            override_session = database.session()
            repository = OwlV2ExampleRepository(session=override_session)
            owns_session = True
        else:
            repository = OwlV2ExampleRepository(session=session)

        examples = repository.list_examples(query_text=request.text)

        if not examples:
            if owns_session and override_session is not None:
                override_session.close()
                if database is not None:
                    database.engine.dispose()
            return SearchResponse(results=[])

        query_images = [example.image_data for example in examples]
        query_labels = [
            example.filename or f"example_{example.id}" for example in examples
        ]
        try:
            results_payload = detection_service.detect_in_directory(
                caption=request.text,
                directory=None,
                patterns=request.patterns,
                box_threshold=request.box_threshold,
                text_threshold=request.text_threshold,
                limit=request.limit,
                only_with_detections=True,
                query_images=query_images,
                query_labels=query_labels,
            )
        finally:
            if owns_session and override_session is not None:
                override_session.close()
                if database is not None:
                    database.engine.dispose()
    else:
        results_payload = detection_service.detect_in_directory(
            caption=request.text,
            directory=None,
            patterns=request.patterns,
            box_threshold=request.box_threshold,
            text_threshold=request.text_threshold,
            limit=request.limit,
            only_with_detections=True,
        )

    response_items = []
    for payload in results_payload:
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
