from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class OwlV2ExampleResponse(BaseModel):
    id: int
    query_text: str
    filename: Optional[str]
    mime_type: Optional[str]
    created_at: datetime

    class Config:
        orm_mode = True


class OwlV2ExampleListResponse(BaseModel):
    examples: List[OwlV2ExampleResponse]
