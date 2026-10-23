"""基准服务：FastAPI，把超长 session 有效控制验证做成可调用的 API。"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from benchmarks.run import run_single, run_sweep


class RunRequest(BaseModel):
    n_turns: int = Field(default=20, ge=3, le=500)
    needle_depth: float = Field(default=0.2, ge=0.0, le=1.0)
    budget_tokens: int = Field(default=512, ge=32, le=1_000_000)
    seed: int | None = Field(default=42)
    encoder_provider: str = Field(default="char_ngram")


class SweepRequest(BaseModel):
    n_turns: int = Field(default=20, ge=3, le=500)
    depths: list[float] = Field(default=[0.05, 0.2, 0.5, 0.8, 0.95])
    budgets: list[int] = Field(default=[256, 512, 1024, 2048])
    seed: int | None = Field(default=42)
    encoder_provider: str = Field(default="char_ngram")


def create_app() -> FastAPI:
    app = FastAPI(title="LongerSession Benchmark", version="0.1.0")
    app.state.last_report: dict | None = None

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "longersession-benchmark"}

    @app.post("/benchmark/run")
    def benchmark_run(req: RunRequest) -> dict:
        report = run_single(
            n_turns=req.n_turns,
            needle_depth=req.needle_depth,
            budget_tokens=req.budget_tokens,
            seed=req.seed,
            encoder_provider=req.encoder_provider,
        )
        app.state.last_report = report
        return report

    @app.post("/benchmark/sweep")
    def benchmark_sweep(req: SweepRequest) -> dict:
        report = run_sweep(
            n_turns=req.n_turns,
            depths=req.depths,
            budgets=req.budgets,
            seed=req.seed,
            encoder_provider=req.encoder_provider,
        )
        app.state.last_report = report
        return report

    @app.get("/benchmark/report")
    def benchmark_report() -> dict:
        if app.state.last_report is None:
            return benchmark_run(RunRequest())
        return app.state.last_report

    return app


app = create_app()
