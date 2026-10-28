"""MCP server 封装测试。"""
from __future__ import annotations

import pytest

fastmcp = pytest.importorskip("fastmcp")


def test_mcp_server_exposes_tools(monkeypatch):
    import longersession.mcp.server as server

    class FakeResp:
        def model_dump(self):
            return {"session_id": "s1", "topics": ["部署"]}

    class FakeService:
        def process(self, req):
            return FakeResp()

        def query(self, req):
            return FakeResp()

    monkeypatch.setattr(server, "_service", FakeService())

    assert server.mcp.name == "longersession"

    out = server.process(
        session_id="s1", messages=[{"role": "user", "content": "hi"}]
    )
    assert out["session_id"] == "s1"
    assert "部署" in out["topics"]

    out2 = server.query(session_id="s1")
    assert out2["session_id"] == "s1"
