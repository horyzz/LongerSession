"""Store 存储层测试。"""
import pytest

from longersession.index.store import Store
from longersession.models.chunk import Chunk
from longersession.models.message import Message, Role
from longersession.models.session import LabelRef, SessionIndex
from longersession.models.turn import Turn


def _make_session(session_id: str, previous: str | None = None):
    si = SessionIndex(session_id=session_id, previous=previous)
    si.topic_refs.append(LabelRef(label="部署", chunk_ids=["chunk_0"]))
    si.chunks.append(
        Chunk(
            chunk_id="chunk_0",
            session_id=session_id,
            turn_ids=["turn_1"],
            topic="部署",
            used_tools=["read_file"],
        )
    )
    turns = [Turn(turn_id="turn_1", user=Message(role=Role.USER, content="hi"))]
    return si, turns


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def test_session_roundtrip(store):
    si, turns = _make_session("s1")
    store.save_session(si, turns)
    loaded, lturns = store.load_session("s1")
    assert loaded.session_id == "s1"
    assert loaded.previous is None
    assert loaded.chunks[0].chunk_id == "chunk_0"
    assert loaded.chunks[0].topic == "部署"
    assert loaded.chunks[0].used_tools == ["read_file"]
    assert loaded.topic_refs[0].label == "部署"
    assert lturns[0].user.content == "hi"


def test_previous_pointer(store):
    si1, t1 = _make_session("s1")
    si2, t2 = _make_session("s2", previous="s1")
    store.save_session(si1, t1)
    store.save_session(si2, t2)
    assert store.get_previous("s2") == "s1"
    assert store.get_previous("s1") is None


def test_persistence_across_restart(tmp_path):
    db = tmp_path / "persist.db"
    s1 = Store(db)
    si, turns = _make_session("s1")
    s1.save_session(si, turns)
    s1.close()
    # 重新打开，验证持久化
    s2 = Store(db)
    loaded, lturns = s2.load_session("s1")
    assert loaded.session_id == "s1"
    assert loaded.chunks[0].topic == "部署"
    assert lturns[0].user.content == "hi"
    s2.close()


def test_load_missing_session(store):
    assert store.load_session("nope") is None
