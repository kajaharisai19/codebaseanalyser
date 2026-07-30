"""Parser registry + language detection.

Primary language is auto-detected from extension frequency across the repo (no CLI
flag required). Each file is parsed with its own language's strategy, falling back
to the generic parser for anything without a dedicated one.
"""
from __future__ import annotations

from collections import Counter

from .base import ParsedFile, MethodSig, TypeDecl  # re-export
from .generic_parser import GenericParser
from .java_parser import JavaParser
from .python_parser import PythonParser

EXT_LANGUAGE = {
    ".java": "java", ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".rb": "ruby", ".cpp": "cpp", ".cc": "cpp", ".c": "c",
    ".cs": "csharp", ".kt": "kotlin", ".rs": "rust", ".php": "php",
    ".scala": "scala", ".swift": "swift",
}

# LangChain Language enum names for syntactic chunking (see chunker.py).
LANGCHAIN_LANGUAGE = {
    "java": "JAVA", "python": "PYTHON", "javascript": "JS", "typescript": "TS",
    "go": "GO", "ruby": "RUBY", "cpp": "CPP", "c": "C", "csharp": "CSHARP",
    "kotlin": "KOTLIN", "rust": "RUST", "php": "PHP", "scala": "SCALA",
    "swift": "SWIFT",
}

_DEDICATED = {"java": JavaParser(), "python": PythonParser()}


def language_for_ext(ext: str) -> str:
    return EXT_LANGUAGE.get(ext.lower(), "generic")


def detect_primary_language(extensions: list[str]) -> str:
    counts = Counter(language_for_ext(e) for e in extensions)
    counts.pop("generic", None)
    return counts.most_common(1)[0][0] if counts else "generic"


def get_parser(language: str):
    return _DEDICATED.get(language, GenericParser(language))


def parse_file(rel_path: str, ext: str, text: str) -> ParsedFile:
    lang = language_for_ext(ext)
    return get_parser(lang).parse(rel_path, text)


__all__ = [
    "ParsedFile", "MethodSig", "TypeDecl",
    "language_for_ext", "detect_primary_language", "get_parser", "parse_file",
    "LANGCHAIN_LANGUAGE",
]
