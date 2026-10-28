"""LongerSession MCP server：把 process / query 暴露为 agent 可调用的工具。

接入方式（pi）：
    pi mcp add longer -- uv run fastmcp run longersession.mcp.server:mcp
或 streamable HTTP：
    uv run fastmcp run longersession.mcp.server:mcp --transport http --port 8002
    pi mcp add longer --url http://localhost:8002/mcp

依赖：uv sync --extra mcp
"""
from __future__ import annotations

from fastmcp import FastMCP

from longersession.api.schemas import ProcessRequest, QueryRequest
from longersession.api.service import Service
from longersession.config import Config, get_config
from longersession.models.message import Message

mcp = FastMCP("longersession")

_service: Service | None = None


def get_service(config: Config | None = None) -> Service:
    """惰性构建 Service（默认配置复用 get_config()，测试可注入 config）。"""
    global _service
    if _service is None:
        _service = Service(config if config is not None else get_config())
    return _service


@mcp.tool()
def process(
    session_id: str, messages: list[dict], previous: str | None = None
) -> dict:
    """把一段 session 做语义切割并建立 topic 索引，挂到 previous 链上。

    Args:
        session_id: 本次 session 的唯一 ID。
        messages: 对话消息列表，元素形如 {"role": "user", "content": "..."}。
        previous: 父 session ID（可选），用于串联 session 链表。
    """
    msgs = [Message.model_validate(m) for m in messages]
    resp = get_service().process(
        ProcessRequest(session_id=session_id, previous=previous, messages=msgs)
    )
    return resp.model_dump()


@mcp.tool()
def query(
    session_id: str, label: str | None = None, kind: str | None = None
) -> dict:
    """沿 previous 链定位历史内容：返回 topic/tag 列表，或指定标签下的 chunk 引用。

    Args:
        session_id: 要查询的 session ID。
        label: 可选，指定标签则返回该标签下的 chunk 引用。
        kind: 可选，"topic" 或 "tag"；省略时自动判断。
    """
    resp = get_service().query(
        QueryRequest(session_id=session_id, label=label, kind=kind)
    )
    return resp.model_dump()


if __name__ == "__main__":
    mcp.run()
