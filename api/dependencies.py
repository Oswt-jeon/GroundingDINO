from __future__ import annotations

from typing import Generator

from fastapi import FastAPI, Request
from sqlalchemy.orm import Session

from config.runtime import get_settings
from src.db.database import Database
from src.services.factory import get_detection_manager
from src.services.manager import DetectionServiceManager


def register_dependencies(app: FastAPI) -> None:
    settings = get_settings()
    app.state.database = Database.create(settings.database_url)
    app.state.detection_manager = get_detection_manager()


def get_detection_manager_dependency(request: Request) -> DetectionServiceManager:
    manager: DetectionServiceManager = getattr(
        request.app.state,
        "detection_manager",
        None,
    )
    if manager is None:
        manager = get_detection_manager()
        request.app.state.detection_manager = manager
    return manager


def get_db_session_dependency(request: Request) -> Generator[Session, None, None]:
    database: Database = getattr(request.app.state, "database", None)
    if database is None:
        settings = get_settings()
        database = Database.create(settings.database_url)
        request.app.state.database = database
    session = database.session()
    try:
        yield session
    finally:
        session.close()
