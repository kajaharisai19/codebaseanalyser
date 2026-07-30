"""Shared parser data structures and the parser Protocol.

Parsers extract structural ground truth *before* the LLM call. This serves two
purposes: (1) a compact skeleton primes the extraction prompt, improving accuracy,
and (2) counts (methods, types) are true regardless of whether the LLM succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class MethodSig:
    name: str
    signature: str            # normalized, single-line
    annotations: list[str] = field(default_factory=list)  # or decorators
    start_line: int = 0
    end_line: int = 0


@dataclass
class TypeDecl:
    name: str
    kind: str                 # class | interface | enum | record | struct | function
    annotations: list[str] = field(default_factory=list)
    methods: list[MethodSig] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0


@dataclass
class ParsedFile:
    rel_path: str
    language: str
    module: str               # package / module path
    imports: list[str] = field(default_factory=list)
    types: list[TypeDecl] = field(default_factory=list)
    parser: str = "generic"   # which strategy produced this (for accuracy notes)

    @property
    def method_count(self) -> int:
        return sum(len(t.methods) for t in self.types)

    def skeleton(self, max_imports: int = 25) -> str:
        """Compact textual skeleton used to prime the extraction prompt."""
        lines = [f"module: {self.module or '(none)'}  [lang={self.language}]"]
        if self.imports:
            shown = self.imports[:max_imports]
            lines.append("imports: " + ", ".join(shown)
                         + (" …" if len(self.imports) > max_imports else ""))
        for t in self.types:
            ann = " ".join(t.annotations)
            lines.append(f"\n{ann + ' ' if ann else ''}{t.kind} {t.name}:")
            for m in t.methods:
                mann = " ".join(m.annotations)
                lines.append(f"  {(mann + ' ') if mann else ''}{m.signature}")
        return "\n".join(lines)


@runtime_checkable
class Parser(Protocol):
    language: str
    def parse(self, rel_path: str, text: str) -> ParsedFile: ...
