from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from .parsers import LANGCHAIN_LANGUAGE, ParsedFile
from .tokenizer import Tokenizer


@dataclass
class Chunk:
    rel_path: str
    language: str
    text: str
    token_count: int
    part: int          # 0-based index of this chunk within its file
    n_parts: int       # total chunks for this file
    parsed: ParsedFile  # structural context for this file (skeleton lives here)

    @property
    def chunk_id(self) -> str:
        return f"{self.rel_path}#{self.part}" if self.n_parts > 1 else self.rel_path


def _langchain_language(name: str) -> Language | None:
    enum_name = LANGCHAIN_LANGUAGE.get(name)
    if enum_name and hasattr(Language, enum_name):
        return getattr(Language, enum_name)
    return None


def _hard_slice(text: str, tok: Tokenizer, budget: int) -> list[str]:
    """Last-resort split by token index. Guarantees each piece <= budget."""
    ids = tok.encode(text)
    return [tok.decode(ids[i : i + budget]) for i in range(0, len(ids), budget)]


class Chunker:
    def __init__(self, tokenizer: Tokenizer, code_budget: int):
        self.tok = tokenizer
        # leave 10% headroom inside the code budget for splitter imprecision
        self.budget = max(256, int(code_budget * 0.90))

    def chunk_file(self, parsed: ParsedFile, text: str) -> list[Chunk]:
        n = self.tok.count(text)
        if n <= self.budget:
            return [self._mk(parsed, text, 0, 1)]

        # oversized: syntactic split
        pieces = self._syntactic_split(parsed.language, text)

        # any piece still too big -> hard slice
        final: list[str] = []
        for p in pieces:
            if self.tok.count(p) <= self.budget:
                final.append(p)
            else:
                final.extend(_hard_slice(p, self.tok, self.budget))

        final = [p for p in final if p.strip()]
        return [self._mk(parsed, p, i, len(final)) for i, p in enumerate(final)]

    def _syntactic_split(self, language: str, text: str) -> list[str]:
        lang = _langchain_language(language)
        if lang is not None:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=self.budget,
                chunk_overlap=0,
                length_function=self.tok.count,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.budget,
                chunk_overlap=0,
                length_function=self.tok.count,
            )
        return splitter.split_text(text)

    def _mk(self, parsed: ParsedFile, text: str, part: int, n_parts: int) -> Chunk:
        return Chunk(
            rel_path=parsed.rel_path, language=parsed.language, text=text,
            token_count=self.tok.count(text), part=part, n_parts=n_parts,
            parsed=parsed,
        )
