from __future__ import annotations

from fastapi import FastAPI

from app.api.v1 import api_router
from app.core.db_init import init_db
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(title="Palmmind RAG API", version="1.0.0")
register_exception_handlers(app)
app.include_router(api_router, prefix="/api/v1")


@app.on_event("startup")
async def startup() -> None:
    await init_db()


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
