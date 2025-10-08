from __future__ import annotations

from typing import List

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

