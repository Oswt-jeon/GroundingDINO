import argparse
import sys
import time

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a single frame from a webcam.")
    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="Video capture device (default: /dev/video0)",
    )
    parser.add_argument(
        "--output",
        default="webcam_capture.jpg",
        help="Path where the captured image will be saved.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Optional width to request from the camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Optional height to request from the camera.",
    )
    parser.add_argument(
        "--fourcc",
        default="YUYV",
        help="Optional FOURCC format to request (e.g., YUYV, MJPG).",
    )
    parser.add_argument(
        "--fallback-device",
        default=None,
        help="Optional fallback device path if the primary device fails.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=800,
        help="Delay in milliseconds to let the camera warm up before capture.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    def open_device(path: str) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        return cap

    cap = open_device(args.device)
    if cap is None and args.fallback_device:
        print(f"[warn] Unable to open {args.device}. Trying fallback {args.fallback_device}...")
        cap = open_device(args.fallback_device)

    if cap is None:
        print(f"[error] Unable to open camera device: {args.device}", file=sys.stderr)
        if args.fallback_device:
            print(f"[error] Fallback device also failed: {args.fallback_device}", file=sys.stderr)
        return 1

    if args.fourcc:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Give the camera a brief moment to adjust.
    time.sleep(args.warmup_ms / 1000)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("[error] Failed to capture frame from camera.", file=sys.stderr)
        return 1

    if cv2.imwrite(args.output, frame):
        print(f"[info] Captured image saved to {args.output}")
        return 0

    print("[error] Failed to write captured frame to disk.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
