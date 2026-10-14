"""Encoder 工厂。"""
from __future__ import annotations

from longersession.config import EncoderConfig
from longersession.encoder.base import Encoder
from longersession.encoder.bge_m3 import BGEM3Encoder
from longersession.encoder.char_ngram import CharNGramEncoder
from longersession.encoder.mock import MockEncoder
from longersession.encoder.word_bag import WordBagEncoder


def create_encoder(config: EncoderConfig) -> Encoder:
    """根据 config.provider 创建对应的 encoder 实例。"""
    if config.provider == "mock":
        return MockEncoder()
    if config.provider == "word_bag":
        return WordBagEncoder()
    if config.provider == "char_ngram":
        return CharNGramEncoder()
    if config.provider == "bge_m3":
        return BGEM3Encoder(
            model_name=config.model_name,
            device=config.device,
            max_seq_length=config.max_seq_length,
        )
    raise ValueError(f"unknown encoder provider: {config.provider}")
