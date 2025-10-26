from __future__ import annotations

from fastapi import FastAPI

from api.dependencies import register_dependencies
from api.routers import detect
from api.routers import owlv2


app = FastAPI(title="GroundingDINO Server")

register_dependencies(app)
app.include_router(detect.router)
app.include_router(owlv2.router)
