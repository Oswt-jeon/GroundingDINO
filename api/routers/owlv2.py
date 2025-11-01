from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from api.dependencies import get_db_session_dependency
from api.schemas.owlv2_examples import OwlV2ExampleListResponse, OwlV2ExampleResponse
from src.db.database import Database
from src.db.models import OwlV2Example
from src.repositories.owlv2_examples import OwlV2ExampleRepository


router = APIRouter(prefix="/owlv2", tags=["owlv2"])


def _serialize_example(example: OwlV2Example) -> OwlV2ExampleResponse:
    image_base64 = None
    if getattr(example, "image_data", None):
        image_base64 = base64.b64encode(example.image_data).decode("utf-8")
    return OwlV2ExampleResponse(
        id=example.id,
        query_text=example.query_text,
        filename=example.filename,
        mime_type=example.mime_type,
        created_at=example.created_at,
        image_base64=image_base64,
    )


@router.post(
    "/examples",
    response_model=OwlV2ExampleResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_owlv2_example(
    query: str = Form(..., description="검색어 (예: 'red backpack')"),
    image: UploadFile = File(..., description="OWLv2 학습용 예시 이미지"),
    database_url: Optional[str] = Form(
        None,
        description="기본값을 덮어쓸 SQLite 연결 문자열",
    ),
    session: Session = Depends(get_db_session_dependency),
) -> OwlV2ExampleResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미지 파일이 비어있습니다.")

    repository_session = session
    temp_database: Optional[Database] = None
    if database_url:
        temp_database = Database.create(database_url)
        repository_session = temp_database.session()

    repository = OwlV2ExampleRepository(session=repository_session)
    try:
        example = repository.add_example(
            query_text=query,
            image_bytes=image_bytes,
            filename=image.filename,
            mime_type=image.content_type,
        )
    finally:
        if temp_database is not None:
            repository_session.close()
            temp_database.engine.dispose()

    return _serialize_example(example)


@router.get(
    "/examples",
    response_model=OwlV2ExampleListResponse,
)
def list_owlv2_examples(
    query: str = Query(..., description="검색어 (예: 'red backpack')"),
    database_url: Optional[str] = Query(None, description="기본값을 덮어쓸 SQLite 연결 문자열"),
    session: Session = Depends(get_db_session_dependency),
) -> OwlV2ExampleListResponse:
    repository_session = session
    temp_database: Optional[Database] = None
    if database_url:
        temp_database = Database.create(database_url)
        repository_session = temp_database.session()

    repository = OwlV2ExampleRepository(session=repository_session)
    try:
        examples = repository.list_examples(query_text=query)
    finally:
        if temp_database is not None:
            repository_session.close()
            temp_database.engine.dispose()

    return OwlV2ExampleListResponse(
        examples=[_serialize_example(item) for item in examples],
    )


@router.delete(
    "/examples/{example_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_owlv2_example(
    example_id: int,
    database_url: Optional[str] = Query(None, description="기본값을 덮어쓸 SQLite 연결 문자열"),
    session: Session = Depends(get_db_session_dependency),
) -> None:
    repository_session = session
    temp_database: Optional[Database] = None
    if database_url:
        temp_database = Database.create(database_url)
        repository_session = temp_database.session()

    repository = OwlV2ExampleRepository(session=repository_session)
    try:
        removed = repository.delete_example(example_id)
    finally:
        if temp_database is not None:
            repository_session.close()
            temp_database.engine.dispose()

    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="예시 이미지를 찾을 수 없습니다.")
