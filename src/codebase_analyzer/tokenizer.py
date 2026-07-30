from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import tiktoken


class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class TiktokenTokenizer:
    """Exact for OpenAI models; used as a documented proxy for others."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


@lru_cache(maxsize=8)
def get_tokenizer(model: str) -> Tokenizer:
    """Return a tokenizer for `model`. Falls back to cl100k_base, which is a
    reasonable proxy for Claude and modern local models (all are BPE-based with
    comparable token/char ratios for code)."""
    m = (model or "").lower()
    try:
        if m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
            return TiktokenTokenizer(tiktoken.encoding_for_model(m).name)
    except (KeyError, ValueError):
        pass
    return TiktokenTokenizer("cl100k_base")
