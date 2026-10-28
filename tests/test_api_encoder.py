"""ApiEncoder 单元测试。"""
from __future__ import annotations

import json

import pytest

from longersession.encoder.api import ApiEncoder, ApiEncoderError, _l2_normalize


def test_l2_normalize():
    assert _l2_normalize([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    assert _l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_encode_batches_and_parses(monkeypatch):
    enc = ApiEncoder(api_key="k", batch_size=2, retries=1)
    calls: list[list[str]] = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body["input"])
        return _FakeResp(
            {
                "data": [
                    {
                        "index": i,
                        "embedding": [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0],
                    }
                    for i in range(len(body["input"]))
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    out = enc.encode(["a", "b", "c"])

    assert calls == [["a", "b"], ["c"]]
    assert len(out) == 3
    assert out[0] == [1.0, 0.0]
    assert out[1] == [0.0, 1.0]
    assert out[2] == [1.0, 0.0]


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("openai_api_key", raising=False)
    enc = ApiEncoder(api_key="", api_key_env="openai_api_key")
    with pytest.raises(ApiEncoderError):
        enc._embed_batch(["x"])


def test_length_mismatch_raises(monkeypatch):
    enc = ApiEncoder(api_key="k", batch_size=2, retries=1)

    def fake_urlopen(req, timeout=None):
        return _FakeResp({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ApiEncoderError):
        enc.encode(["a", "b"])
