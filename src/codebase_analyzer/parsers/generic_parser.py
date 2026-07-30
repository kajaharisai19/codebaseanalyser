"""Language-agnostic fallback parser.

Used for any language without a dedicated strategy. It recovers a best-effort list
of function-like declarations via broad regexes across the common families (C/JS/
Go/Ruby/etc). It never claims more than it can see — a file that yields nothing
still produces a valid ParsedFile, and the LLM extraction pass carries the rest.
"""
from __future__ import annotations

import re

from .base import MethodSig, ParsedFile, TypeDecl

# Broad, deliberately conservative patterns across common languages.
_PATTERNS = [
    re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\(", re.M),   # go
    re.compile(r"^\s*function\s+([A-Za-z_]\w*)\s*\(", re.M),                # js/php
    re.compile(r"^\s*def\s+([A-Za-z_]\w*)", re.M),                          # ruby/py-like
    re.compile(r"^\s*fn\s+([A-Za-z_]\w*)\s*\(", re.M),                      # rust
    re.compile(r"^\s*(?:public|private|protected|internal)?\s*"
               r"(?:static\s+)?[\w<>\[\],.]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*\{", re.M),  # c-family
]
_IMPORT = re.compile(r"^\s*(?:import|#include|require|use)\s+([^\n;]+)", re.M)


class GenericParser:
    language = "generic"

    def __init__(self, language: str = "generic"):
        self.language = language

    def parse(self, rel_path: str, text: str) -> ParsedFile:
        seen: set[str] = set()
        methods: list[MethodSig] = []
        for pat in _PATTERNS:
            for m in pat.finditer(text):
                name = m.group(1)
                if name in seen:
                    continue
                seen.add(name)
                methods.append(
                    MethodSig(name=name, signature=name + "(…)",
                              start_line=text.count("\n", 0, m.start()) + 1)
                )
        imports = [i.strip() for i in _IMPORT.findall(text)][:40]
        types = [TypeDecl(name="(file)", kind="file", methods=methods)] if methods else []
        return ParsedFile(
            rel_path=rel_path, language=self.language, module="",
            imports=imports, types=types, parser="generic-regex",
        )
