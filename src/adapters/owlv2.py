from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


@dataclass
class Owlv2PredictionResult:
    boxes: torch.Tensor
    logits: torch.Tensor
    phrases: List[str]


class Owlv2ModelAdapter:
    """
    OWLv2 zero-shot(one-shot) 탐지를 위한 Hugging Face 어댑터.

    공식 문서([Transformers OWLv2 가이드](https://huggingface.co/docs/transformers/main/en/model_doc/owlv2))
    에 맞춰 `Owlv2Processor.post_process_grounded_object_detection`을 사용한다.
    모델은 항상 1장 이상의 이미지를 입력으로 요구하므로 DetectionService에서 전달한
    단일 이미지를 리스트로 감싸 처리한다.
    """

    def __init__(
        self,
        *,
        model_id: Optional[str],
        weights_path: Optional[Path],
        device: str = "cuda",
        confidence_threshold: float = 0.2,
    ) -> None:
        if model_id is None and weights_path is None:
            raise ValueError("OWLv2를 초기화하려면 model_id 또는 weights_path 중 하나가 필요합니다.")

        pretrained_source = str(weights_path) if weights_path else model_id
        if not pretrained_source:
            raise ValueError("OWLv2 모델 소스를 확인할 수 없습니다.")

        resolved_device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved_device)
        self.confidence_threshold = confidence_threshold

        self.processor = Owlv2Processor.from_pretrained(pretrained_source)
        self.model = Owlv2ForObjectDetection.from_pretrained(pretrained_source)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _parse_caption(caption: str) -> List[str]:
        if not caption:
            return []
        normalized = caption.replace(";", ".").replace(",", ".")
        chunks = [piece.strip() for piece in normalized.split(".")]
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _build_text_labels(labels: List[str]) -> List[str]:
        # 문서: text_labels는 "a photo of ..." 형식 권장
        if not labels:
            return ["a photo of an object"]
        prompts: List[str] = []
        for label in labels:
            text = label if label else "object"
            prompts.append(f"a photo of {text}")
        return prompts

    def _resolve_labels(self, caption: str) -> List[str]:
        labels = self._parse_caption(caption)
        return labels or ["object"]

    def load_image(self, image_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """RGB numpy 배열과 channel-first 텐서를 반환."""
        with Image.open(image_path) as img:
            image_rgb = img.convert("RGB")
        image_np = np.array(image_rgb)
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_np, tensor

    def predict(
        self,
        *,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Owlv2PredictionResult:
        """
        단일 이미지를 zero-shot 텍스트 프롬프트로 탐지한다.
        - 입력 이미지는 (C, H, W) 텐서.
        - OWLv2는 리스트 입력을 요구하므로 `[image]` 형태로 전달한다.
        """
        labels = self._resolve_labels(caption)
        text_labels = self._build_text_labels(labels)

        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        inputs = self.processor(
            text=[text_labels],
            images=[image_np],
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        height, width = image_np.shape[0], image_np.shape[1]
        target_sizes = torch.tensor([(height, width)], dtype=torch.float32)
        score_threshold = box_threshold if box_threshold is not None else self.confidence_threshold

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
            text_labels=[text_labels],
        )

        if not results:
            empty = torch.empty((0, 4), dtype=torch.float32)
            return Owlv2PredictionResult(
                boxes=empty,
                logits=torch.empty((0,), dtype=torch.float32),
                phrases=[],
            )

        result = results[0]
        boxes_tensor = result.get("boxes")
        scores_tensor = result.get("scores")
        phrases_raw = result.get("text_labels")

        if boxes_tensor is None or scores_tensor is None:
            empty = torch.empty((0, 4), dtype=torch.float32)
            return Owlv2PredictionResult(
                boxes=empty,
                logits=torch.empty((0,), dtype=torch.float32),
                phrases=[],
            )

        boxes = boxes_tensor.detach().cpu()
        scores = scores_tensor.detach().cpu()
        if phrases_raw is None:
            phrases = [label for label in labels[: len(boxes)]]
            if len(phrases) < len(boxes):
                phrases.extend(["object"] * (len(boxes) - len(phrases)))
        else:
            if isinstance(phrases_raw, torch.Tensor):
                phrases = [str(item) for item in phrases_raw]
            else:
                phrases = [str(item) for item in phrases_raw]

        return Owlv2PredictionResult(
            boxes=boxes,
            logits=scores,
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
        """검출 결과를 BGR 이미지로 시각화."""
        if image_source.ndim != 3:
            raise ValueError("image_source는 3차원 RGB 배열이어야 합니다.")

        image_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
        for box_tensor, score_tensor, label in zip(boxes, logits, phrases):
            x0, y0, x1, y1 = [int(round(v)) for v in box_tensor.tolist()]
            score = float(score_tensor.item() if hasattr(score_tensor, "item") else score_tensor)

            cv2.rectangle(image_bgr, (x0, y0), (x1, y1), (255, 140, 0), 2)
            caption = f"{label}: {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
            )
            top_left = (x0, max(0, y0 - text_height - baseline))
            bottom_right = (x0 + text_width, y0)
            cv2.rectangle(image_bgr, top_left, bottom_right, (255, 140, 0), thickness=cv2.FILLED)
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
