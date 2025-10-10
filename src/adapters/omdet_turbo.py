from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import inspect
from transformers import AutoProcessor, OmDetTurboForObjectDetection


@dataclass
class OmDetTurboPredictionResult:
    boxes: torch.Tensor
    logits: torch.Tensor
    phrases: List[str]


class OmDetTurboModelAdapter:
    """
    Hugging Face Transformers-based adapter for OmDet Turbo.

    Uses AutoProcessor + OmDetTurboForObjectDetection to support
    text-guided detection similar to the official examples.
    """

    def __init__(
        self,
        *,
        model_id: Optional[str],
        weights_path: Optional[Path],
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        class_names: Optional[List[str]] = None,
    ) -> None:
        if model_id is None and weights_path is None:
            raise ValueError("Either model_id or weights_path must be provided for OmDet Turbo.")

        pretrained_source = str(weights_path) if weights_path else model_id
        if pretrained_source is None:
            raise ValueError("Unable to resolve OmDet Turbo model source.")

        resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)
        self.confidence_threshold = confidence_threshold

        self.processor = AutoProcessor.from_pretrained(pretrained_source)
        self.model = OmDetTurboForObjectDetection.from_pretrained(pretrained_source)
        self.model.to(self.device)
        self.model.eval()

        # Optional override of class names (not generally used for OmDet Turbo).
        self._class_names = class_names

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """Load an image as RGB numpy array and channel-first torch tensor."""
        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB")
        image_np = np.array(image_rgb)
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_np, tensor

    def _prepare_text_labels(self, caption: str) -> List[str]:
        if not caption:
            return ["object"]
        raw_parts = caption.replace(";", ".").replace(",", ".").split(".")
        labels = [part.strip() for part in raw_parts if part.strip()]
        if not labels:
            labels = ["object"]
        return labels

    def _resolve_labels(self, caption: str) -> List[str]:
        if self._class_names:
            return list(self._class_names)
        return self._prepare_text_labels(caption)

    def predict(
        self,
        *,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> OmDetTurboPredictionResult:
        """
        Run OmDet Turbo inference with text-guided prompts.
        """
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        labels = self._resolve_labels(caption)
        task_prompt = f"Detect {', '.join(labels)}."

        inputs = self.processor(
            images=[image_np],
            text=[labels],
            task=[task_prompt],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [(image_np.shape[0], image_np.shape[1])]
        score_threshold = box_threshold if box_threshold is not None else self.confidence_threshold
        post_process_fn = getattr(self.processor, "post_process_grounded_object_detection")
        sig = inspect.signature(post_process_fn)
        kwargs = dict(
            outputs=outputs,
            target_sizes=target_sizes,
        )
        if "classes" in sig.parameters:
            kwargs["classes"] = [labels]
        if "threshold" in sig.parameters:
            kwargs["threshold"] = score_threshold
        elif "score_threshold" in sig.parameters:
            kwargs["score_threshold"] = score_threshold
        if "nms_threshold" in sig.parameters:
            kwargs["nms_threshold"] = 0.3
        if "text" in sig.parameters:
            kwargs["text"] = [labels]
        if "text_labels" in sig.parameters:
            kwargs["text_labels"] = [labels]
        processed_list = post_process_fn(**kwargs)
        if not processed_list:
            empty = torch.empty((0, 4), dtype=torch.float32)
            return OmDetTurboPredictionResult(
                boxes=empty,
                logits=torch.empty((0,), dtype=torch.float32),
                phrases=[],
            )

        processed = processed_list[0]

        boxes_tensor = processed.get("boxes")
        scores_tensor = processed.get("scores")
        if boxes_tensor is None or scores_tensor is None:
            empty = torch.empty((0, 4), dtype=torch.float32)
            return OmDetTurboPredictionResult(
                boxes=empty,
                logits=torch.empty((0,), dtype=torch.float32),
                phrases=[],
            )

        boxes = boxes_tensor.detach().cpu()
        scores = scores_tensor.detach().cpu()
        phrase_candidates = (
            processed.get("classes")
            or processed.get("text_labels")
            or processed.get("labels")
            or labels
        )
        if isinstance(phrase_candidates, torch.Tensor):
            phrases = [str(item) for item in phrase_candidates.tolist()]
        elif isinstance(phrase_candidates, str):
            phrases = [phrase_candidates]
        else:
            phrases = [str(item) for item in list(phrase_candidates)]

        return OmDetTurboPredictionResult(
            boxes=boxes,
            logits=scores,
            phrases=list(phrases),
        )

    def annotate(
        self,
        *,
        image_source: np.ndarray,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        phrases: List[str],
    ) -> np.ndarray:
        """Draw prediction boxes on the image (returns BGR array for OpenCV compatibility)."""
        if image_source.ndim != 3:
            raise ValueError("Expected image_source to be an RGB array.")

        image_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        for box_tensor, score_tensor, label in zip(boxes, logits, phrases):
            x0, y0, x1, y1 = [int(round(v)) for v in box_tensor.tolist()]
            score = float(score_tensor.item() if hasattr(score_tensor, "item") else score_tensor)

            cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (0, 200, 255), 2)
            caption = f"{label}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
            )
            top_left = (x0, max(0, y0 - text_height - baseline))
            bottom_right = (x0 + text_width, y0)
            cv2.rectangle(image_bgr, top_left, bottom_right, (0, 200, 255), thickness=cv2.FILLED)
            cv2.putText(
                image_bgr,
                caption,
                (x0, y0 - 2),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return image_bgr
