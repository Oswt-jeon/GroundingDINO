from __future__ import annotations

import argparse
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List

from sqlalchemy import select

from src.db.database import Database
from src.db.models import OwlV2Example
from src.repositories.owlv2_examples import OwlV2ExampleRepository


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_database_url(cli_database_url: str | None) -> str:
    if cli_database_url:
        return cli_database_url

    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url

    project_root = Path(__file__).resolve().parents[1]
    default_path = (project_root / "data/sqlite/owlv2_examples.db").resolve()
    default_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{default_path}"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="지정된 폴더의 이미지를 OWLv2 예시 데이터베이스에 등록합니다.",
    )
    parser.add_argument(
        "directory",
        help="예시 이미지 폴더",
    )
    parser.add_argument(
        "--database-url",
        help="사용할 SQLite 데이터베이스 URL (미지정 시 기본 설정 사용)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="하위 디렉터리를 재귀적으로 스캔합니다.",
    )
    return parser.parse_args(argv)


def derive_query_name(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"-\d+$", "", stem)
    stem = stem.replace("_", " ")
    return stem.strip()


def is_image_file(path: Path) -> bool:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return True
    mime, _ = mimetypes.guess_type(str(path))
    return bool(mime and mime.startswith("image/"))


def collect_images(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        candidates = sorted(root.rglob("*"))
    else:
        candidates = sorted(root.iterdir())
    return [path for path in candidates if path.is_file() and is_image_file(path)]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[error] 디렉터리를 찾을 수 없습니다: {root}", file=sys.stderr)
        return 1

    db_url = _resolve_database_url(args.database_url)
    database = Database.create(db_url)
    session = database.session()
    repository = OwlV2ExampleRepository(session=session)

    images = collect_images(root, args.recursive)
    if not images:
        print("[info] 가져올 이미지가 없습니다.")
        session.close()
        return 0

    inserted = 0
    skipped = 0

    try:
        for image_path in images:
            query_text = derive_query_name(image_path)
            if not query_text:
                print(f"[warn] 파일명에서 쿼리 텍스트를 추출할 수 없어 건너뜁니다: {image_path}")
                skipped += 1
                continue

            try:
                image_bytes = image_path.read_bytes()
            except OSError as exc:
                print(f"[warn] 파일을 읽을 수 없어 건너뜁니다 ({exc}): {image_path}")
                skipped += 1
                continue

            normalized_query = repository.normalize_query(query_text)
            data_hash = repository._hash_bytes(image_bytes)  # pylint: disable=protected-access

            existing = session.execute(
                select(OwlV2Example).where(
                    OwlV2Example.query_text == normalized_query,
                    OwlV2Example.data_hash == data_hash,
                ).limit(1)
            ).scalars().first()

            if existing:
                skipped += 1
                continue

            mime_type, _ = mimetypes.guess_type(str(image_path))
            repository.add_example(
                query_text=query_text,
                image_bytes=image_bytes,
                filename=image_path.name,
                mime_type=mime_type,
            )
            inserted += 1

        print(f"[done] 처리 완료. 새로 저장: {inserted}, 건너뜀: {skipped}, 총 파일: {len(images)}")
    finally:
        session.close()
        database.engine.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
