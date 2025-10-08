from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from src.services.factory import create_detection_service


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process a directory of images with GroundingDINO.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing images to scan.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help='Caption to search for (e.g. "chips . snack . bottle").',
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.jpg", "*.png", "*.jpeg", "*.bmp", "*.webp"],
        help="Glob patterns (relative to input directory) to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL file for results.",
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


def iter_images(directory: Path, patterns: Sequence[str]) -> Iterator[Path]:
    for pattern in patterns:
        yield from directory.glob(pattern)


def run_batch(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    directory = args.input_dir.expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    service = create_detection_service()
    results: List[dict] = []

    for image_path in sorted(set(iter_images(directory, args.patterns))):
        if not image_path.is_file():
            continue
        detection = service.detect_from_path(
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
                for det in detection.items
            ],
            "annotated": str(detection.annotated_path)
            if detection.annotated_path
            else None,
        }
        results.append(payload)
        print(json.dumps(payload, ensure_ascii=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            for record in results:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_batch()

