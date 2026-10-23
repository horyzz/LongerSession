"""LongEval 真实基准服务：FastAPI，把 DeepSeek × 四策略评估做成可调 API。"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from eval_longeval.run import run_eval


class EvalRequest(BaseModel):
    num_topics: list[int] = Field(default=[5, 10])
    num_samples: int = Field(default=2, ge=1, le=10)
    budget_tokens: int = Field(default=1500, ge=128, le=100_000)
    seed: int = Field(default=0)
    strategies: list[str] | None = Field(
        default=None,
        description="可选：full/truncation/compression/index-conversation/index-segmentation",
    )


def create_app() -> FastAPI:
    app = FastAPI(title="LongerSession LongEval Benchmark", version="0.1.0")
    app.state.last_report: dict | None = None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "longeval-benchmark"}

    @app.post("/longeval/run")
    def longeval_run(req: EvalRequest) -> dict:
        report = run_eval(
            num_topics_list=req.num_topics,
            num_samples=req.num_samples,
            budget_tokens=req.budget_tokens,
            seed=req.seed,
            strategies=req.strategies,
        )
        app.state.last_report = report
        return report

    @app.get("/longeval/report")
    def longeval_report() -> dict:
        if app.state.last_report is None:
            return longeval_run(EvalRequest())
        return app.state.last_report

    return app


app = create_app()
