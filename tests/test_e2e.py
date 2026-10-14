"""端到端测试：mock encoder + mock 标签器跑通完整链路。"""
import pytest
from fastapi.testclient import TestClient

from longersession.api.app import create_app
from longersession.config import Config, StorageConfig


@pytest.fixture
def client(tmp_path):
    cfg = Config(storage=StorageConfig(db_path=tmp_path / "e2e.db"))
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def test_full_process_and_query(client):
    messages = [
        {"role": "user", "content": "我想部署一个服务，Docker 方案怎么样"},
        {"role": "assistant", "content": "Docker 很适合小项目"},
        {"role": "user", "content": "Docker 镜像版本怎么管理"},
        {"role": "assistant", "content": "用 tag 管理"},
        {"role": "user", "content": "数据备份怎么做"},
        {"role": "assistant", "content": "定期快照"},
        {"role": "user", "content": "备份频率怎么定"},
        {"role": "assistant", "content": "每天增量"},
    ]
    r = client.post("/process", json={"session_id": "s1", "messages": messages})
    assert r.status_code == 200
    body = r.json()
    assert body["num_turns"] == 4
    assert body["num_chunks"] >= 1

    # 无 label：返回 topic 列表
    r2 = client.get("/query", params={"session_id": "s1"})
    assert r2.status_code == 200
    assert "部署" in r2.json()["topics"]

    # 指定 label：返回 chunk 引用
    r3 = client.get("/query", params={"session_id": "s1", "label": "部署"})
    assert r3.status_code == 200
    assert r3.json()["chunk_refs"]


def test_previous_chain(client):
    # 父 session：部署话题
    client.post(
        "/process",
        json={
            "session_id": "parent",
            "messages": [
                {"role": "user", "content": "Docker 部署方案"},
                {"role": "assistant", "content": "用 Docker Compose"},
                {"role": "user", "content": "镜像仓库怎么选"},
                {"role": "assistant", "content": "Harbor 或 Docker Hub"},
            ],
        },
    )
    # 子 session：数据备份话题，previous 指向 parent
    client.post(
        "/process",
        json={
            "session_id": "child",
            "previous": "parent",
            "messages": [
                {"role": "user", "content": "数据备份怎么做"},
                {"role": "assistant", "content": "定期快照"},
                {"role": "user", "content": "备份保留多久"},
                {"role": "assistant", "content": "30 天"},
            ],
        },
    )
    # 无 label：从 child 沿 previous 链聚合出父+子的 topic
    r = client.get("/query", params={"session_id": "child"})
    assert r.status_code == 200
    topics = r.json()["topics"]
    assert "部署" in topics  # 父 session 的 topic 沿链可见

    # 按 label 取：命中父 session 的 chunk
    r2 = client.get("/query", params={"session_id": "child", "label": "部署"})
    refs = r2.json()["chunk_refs"]
    assert any(sid == "parent" for sid, _ in refs)


def test_persistence_across_restart(tmp_path):
    cfg = Config(storage=StorageConfig(db_path=tmp_path / "persist.db"))
    app1 = create_app(cfg)
    with TestClient(app1) as c1:
        r = c1.post(
            "/process",
            json={
                "session_id": "s1",
                "messages": [
                    {"role": "user", "content": "Docker 部署方案"},
                    {"role": "assistant", "content": "用 Compose"},
                ],
            },
        )
        assert r.status_code == 200

    # 重启：新建 app + service，重新读取同一 db
    app2 = create_app(cfg)
    with TestClient(app2) as c2:
        r2 = c2.get("/query", params={"session_id": "s1"})
        assert r2.status_code == 200
        assert "部署" in r2.json()["topics"]
