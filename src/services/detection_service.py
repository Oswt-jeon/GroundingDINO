from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.adapters.grounding_dino import GroundingDinoModelAdapter, PredictionResult
from src.utils.file_io import ensure_directory, write_bytes_to_temp, write_image


@dataclass
class Detection:
    box: List[float]
    label: str
    score: float


@dataclass
class DetectionResultPayload:
    items: List[Detection]
    source_path: Path
    annotated_path: Optional[Path]


class DetectionService:
    def __init__(
        self,
        *,
        model_adapter: GroundingDinoModelAdapter,
        images_dir: Path,
        results_dir: Path,
        search_dir: Path,
        default_box_threshold: float,
        default_text_threshold: float,
        annotate_results: bool = True,
    ) -> None:
        self._adapter = model_adapter
        self._images_dir = ensure_directory(images_dir)
        self._results_dir = ensure_directory(results_dir)
        self._search_dir = ensure_directory(search_dir)
        self._default_box_threshold = default_box_threshold
        self._default_text_threshold = default_text_threshold
        self._annotate_results = annotate_results

    def detect_from_bytes(
        self,
        *,
        data: bytes,
        filename: Optional[str],
        caption: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        persist_input: bool = False,
    ) -> DetectionResultPayload:
        image_path = write_bytes_to_temp(
            data,
            filename=filename,
            directory=self._images_dir,
        )
        try:
            return self.detect_from_path(
                image_path=image_path,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        finally:
            if not persist_input and image_path.exists():
                try:
                    image_path.unlink()
                except OSError:
                    pass

    def detect_from_path(
        self,
        *,
        image_path: Path,
        caption: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> DetectionResultPayload:
        image_source, image_tensor = self._adapter.load_image(image_path)
        prediction = self._adapter.predict(
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold or self._default_box_threshold,
            text_threshold=text_threshold or self._default_text_threshold,
        )
        detections = self._build_detections(prediction)
        annotated_path = self._maybe_annotate(
            image_source=image_source,
            prediction=prediction,
            original_path=image_path,
        )
        return DetectionResultPayload(
            items=detections,
            source_path=image_path,
            annotated_path=annotated_path,
        )

    def detect_in_directory(
        self,
        *,
        caption: str,
        directory: Optional[Path] = None,
        patterns: Optional[List[str]] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        limit: Optional[int] = None,
        only_with_detections: bool = True,
    ) -> List[DetectionResultPayload]:
        target_dir = directory or self._search_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"Search directory not found: {target_dir}")

        glob_patterns = patterns or ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        collected: List[DetectionResultPayload] = []

        candidates = set()
        for pattern in glob_patterns:
            candidates.update(target_dir.glob(pattern))

        for image_path in sorted(candidates):
            if not image_path.is_file():
                continue
            result = self.detect_from_path(
                image_path=image_path,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if only_with_detections and not result.items:
                continue
            collected.append(result)
            if limit is not None and len(collected) >= limit:
                return collected

        return collected

    def _build_detections(
        self,
        prediction: PredictionResult,
    ) -> List[Detection]:
        boxes_cpu = prediction.boxes.cpu()
        logits_cpu = prediction.logits.cpu()
        detections: List[Detection] = []
        for box, score, phrase in zip(
            boxes_cpu,
            logits_cpu,
            prediction.phrases,
        ):
            detections.append(
                Detection(
                    box=[float(v) for v in box.tolist()],
                    label=str(phrase),
                    score=float(score.item() if hasattr(score, "item") else score),
                )
            )
        return detections

    def _maybe_annotate(
        self,
        *,
        image_source,
        prediction: PredictionResult,
        original_path: Path,
    ) -> Optional[Path]:
        if not self._annotate_results or len(prediction.phrases) == 0:
            return None
        annotated = self._adapter.annotate(
            image_source=image_source,
            boxes=prediction.boxes,
            logits=prediction.logits,
            phrases=prediction.phrases,
        )
        target_name = f"{original_path.stem}_annotated.jpg"
        target_path = self._results_dir / target_name
        write_image(annotated[:, :, ::-1], target_path=target_path)  # convert BGR->RGB
        return target_path
