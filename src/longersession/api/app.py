"""FastAPI 应用。"""
from __future__ import annotations

from fastapi import FastAPI

from longersession.api.schemas import (
    ProcessRequest,
    ProcessResponse,
    QueryRequest,
    QueryResponse,
)
from longersession.api.service import Service
from longersession.config import Config, get_config


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config if config is not None else get_config()
    service = Service(cfg)
    app = FastAPI(title="LongerSession", version="0.1.0")
    app.state.service = service

    @app.post("/process", response_model=ProcessResponse)
    def process(req: ProcessRequest) -> ProcessResponse:
        return service.process(req)

    @app.get("/query", response_model=QueryResponse)
    def query(
        session_id: str, label: str | None = None, kind: str | None = None
    ) -> QueryResponse:
        return service.query(
            QueryRequest(session_id=session_id, label=label, kind=kind)
        )

    return app


app = create_app()
