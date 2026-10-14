"""SQLite 存储层。"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from longersession.models.session import SessionIndex
from longersession.models.turn import Turn


class Store:
    """基于 SQLite 的持久化存储（sessions / turns 两张表）。

    topic/tag 是 chunk 上的字符串属性，随 session 索引一起存，
    不需要单独的 topic 表。
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                previous TEXT,
                index_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                turn_json TEXT NOT NULL,
                PRIMARY KEY (session_id, turn_id)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- session / turn ----
    def save_session(self, session: SessionIndex, turns: list[Turn]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, previous, index_json) VALUES (?, ?, ?)",
            (session.session_id, session.previous, session.model_dump_json()),
        )
        for t in turns:
            self._conn.execute(
                "INSERT OR REPLACE INTO turns (session_id, turn_id, turn_json) VALUES (?, ?, ?)",
                (session.session_id, t.turn_id, t.model_dump_json()),
            )
        self._conn.commit()

    def load_session(self, session_id: str) -> tuple[SessionIndex, list[Turn]] | None:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        session = SessionIndex.model_validate_json(row["index_json"])
        turn_rows = self._conn.execute(
            "SELECT turn_json FROM turns WHERE session_id = ? ORDER BY rowid",
            (session_id,),
        ).fetchall()
        turns = [Turn.model_validate_json(r["turn_json"]) for r in turn_rows]
        return session, turns

    def get_previous(self, session_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT previous FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row["previous"] if row else None

    def session_exists(self, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None
