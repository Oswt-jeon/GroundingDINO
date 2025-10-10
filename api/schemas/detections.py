from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from src.services.detection_service import Detection


class DetectItem(BaseModel):
    box: List[float]
    label: str
    score: float

    @classmethod
    def from_domain(cls, detection: Detection) -> "DetectItem":
        return cls(
            box=list(detection.box),
            label=detection.label,
            score=detection.score,
        )


class AnnotatedImageResponse(BaseModel):
    data: str
    mime_type: str


class ImageSearchResult(BaseModel):
    image: str
    detections: List[DetectItem]
    annotated_image: Optional[AnnotatedImageResponse] = None


class SearchResponse(BaseModel):
    results: List[ImageSearchResult]


class SearchRequest(BaseModel):
    text: str
    box_threshold: Optional[float] = None
    text_threshold: Optional[float] = None
    limit: Optional[int] = None
    patterns: Optional[List[str]] = None
    model: Optional[str] = None
