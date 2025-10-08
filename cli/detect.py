from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from src.services.factory import create_detection_service


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO detections on one or more images.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help='Caption to search for (e.g. "chips . snack . bottle").',
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to image files.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
        help="Override detection box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=None,
        help="Override detection text threshold.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    service = create_detection_service()
    for image in args.images:
        image_path = Path(image).expanduser().resolve()
        if not image_path.exists():
            print(f"[warn] Skipping missing image: {image_path}")
            continue
        result = service.detect_from_path(
            image_path=image_path,
            caption=args.text,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        payload = {
            "image": str(image_path),
            "detections": [
                {
                    "box": det.box,
                    "label": det.label,
                    "score": det.score,
                }
                for det in result.items
            ],
            "annotated": str(result.annotated_path) if result.annotated_path else None,
        }
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()

