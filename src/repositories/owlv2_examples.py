from __future__ import annotations

import hashlib
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.db.models import OwlV2Example


class OwlV2ExampleRepository:
    """데이터베이스에 저장된 OWLv2 예시 이미지를 관리하는 저장소."""

    def __init__(self, *, session: Session) -> None:
        self._session = session

    @staticmethod
    def normalize_query(query_text: str) -> str:
        normalized = " ".join(query_text.strip().lower().split())
        return normalized

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def add_example(
        self,
        *,
        query_text: str,
        image_bytes: bytes,
        filename: Optional[str],
        mime_type: Optional[str],
    ) -> OwlV2Example:
        normalized_query = self.normalize_query(query_text)
        data_hash = self._hash_bytes(image_bytes)

        existing_stmt = (
            select(OwlV2Example)
            .where(
                OwlV2Example.query_text == normalized_query,
                OwlV2Example.data_hash == data_hash,
            )
            .limit(1)
        )
        existing = self._session.execute(existing_stmt).scalars().first()
        if existing:
            return existing

        example = OwlV2Example(
            query_text=normalized_query,
            filename=filename,
            mime_type=mime_type,
            data_hash=data_hash,
            image_data=image_bytes,
        )
        self._session.add(example)
        try:
            self._session.commit()
        except IntegrityError:
            self._session.rollback()
            # 경쟁 상태에서 동일한 레코드가 저장된 경우 기존 값으로 대체
            existing = self._session.execute(existing_stmt).scalars().first()
            if existing:
                return existing
            raise
        self._session.refresh(example)
        return example

    def list_examples(self, *, query_text: str) -> List[OwlV2Example]:
        normalized_query = self.normalize_query(query_text)
        stmt = (
            select(OwlV2Example)
            .where(OwlV2Example.query_text == normalized_query)
            .order_by(OwlV2Example.created_at.asc())
        )
        return list(self._session.execute(stmt).scalars().all())

    def delete_example(self, example_id: int) -> bool:
        instance = self._session.get(OwlV2Example, example_id)
        if instance is None:
            return False
        self._session.delete(instance)
        self._session.commit()
        return True
