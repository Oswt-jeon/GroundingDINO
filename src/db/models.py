from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, LargeBinary, String, UniqueConstraint, func

from src.db.base import Base


class OwlV2Example(Base):
    __tablename__ = "owlv2_examples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(String(255), nullable=False, index=True)
    filename = Column(String(255), nullable=True)
    mime_type = Column(String(100), nullable=True)
    data_hash = Column(String(128), nullable=False)
    image_data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("query_text", "data_hash", name="uq_owlv2_query_hash"),
    )
