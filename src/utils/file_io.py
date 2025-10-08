from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_suffix(filename: Optional[str], default: str = ".jpg") -> str:
    if not filename:
        return default
    suffix = Path(filename).suffix
    return suffix or default


def write_bytes_to_temp(
    data: bytes,
    *,
    filename: Optional[str] = None,
    directory: Path,
) -> Path:
    ensure_directory(directory)
    suffix = get_suffix(filename)
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        dir=directory,
        delete=False,
    ) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def write_image(
    image,
    *,
    target_path: Path,
) -> Path:
    ensure_directory(target_path.parent)
    from PIL import Image

    if isinstance(image, Image.Image):
        image.save(target_path)
    else:
        # Assume numpy array in BGR or RGB; let Pillow infer mode.
        Image.fromarray(image).save(target_path)
    return target_path

