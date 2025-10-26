from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.base import Base
from src.db import models  # noqa: F401  # Ensure models are registered


def _sqlite_connect_args(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


@dataclass
class Database:
    url: str
    engine: Engine
    session_factory: sessionmaker

    @classmethod
    def create(cls, database_url: str) -> "Database":
        engine = create_engine(
            database_url,
            connect_args=_sqlite_connect_args(database_url),
            future=True,
        )
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
        return cls(
            url=database_url,
            engine=engine,
            session_factory=factory,
        )

    def session(self) -> Session:
        return self.session_factory()


def get_session(database: Database) -> Generator[Session, None, None]:
    session = database.session()
    try:
        yield session
    finally:
        session.close()
