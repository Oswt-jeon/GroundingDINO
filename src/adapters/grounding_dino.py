from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from groundingdino.util.inference import (
    annotate,
    load_image,
    load_model,
    predict,
)


@dataclass
class PredictionResult:
    boxes: torch.Tensor
    logits: torch.Tensor
    phrases: List[str]


class GroundingDinoModelAdapter:
    """Thin wrapper around the GroundingDINO inference helpers."""

    def __init__(
        self,
        *,
        config_path: Path,
        weights_path: Path,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self._model = load_model(
            model_config_path=str(config_path),
            model_checkpoint_path=str(weights_path),
            device=device,
        )

    @property
    def model(self):
        return self._model

    def resolve_device(self) -> str:
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        return load_image(str(image_path))

    def predict(
        self,
        *,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> PredictionResult:
        device = self.resolve_device()
        boxes, logits, phrases = predict(
            model=self._model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
        return PredictionResult(
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )

    def annotate(
        self,
        *,
        image_source: np.ndarray,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        phrases: List[str],
    ) -> np.ndarray:
        return annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases,
        )

