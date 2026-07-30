"""Regex-based structural parser for Java.

Java has no batteries-included AST in the Python stdlib, so this is a pragmatic
regex/brace-tracking parser rather than a full grammar. It reliably recovers
package, imports, top-level type declarations with their annotations, and method
signatures (including annotation-only interface method declarations). Its known
limits are documented in the README: it does not resolve generics fully, may miss
methods with unusual formatting, and does not build a real type graph.
"""
from __future__ import annotations

import re

from .base import MethodSig, ParsedFile, TypeDecl

_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_IMPORT = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;", re.M)
_ANNOTATION = re.compile(r"@[\w.]+(?:\s*\([^)]*\))?")
_TYPE_DECL = re.compile(
    r"\b(?:public|private|protected|abstract|final|sealed|static|\s)*"
    r"\b(class|interface|enum|record)\s+([A-Za-z_]\w*)"
)
# method: optional modifiers, return type, name, (params) then '{' or ';' (iface)
_METHOD = re.compile(
    r"(?:(?:public|private|protected|static|final|abstract|default|synchronized|native)\s+)*"
    r"(?:<[^>]+>\s*)?"                       # generic type params
    r"([\w.$\[\]<>,?\s]+?)\s+"               # return type
    r"([A-Za-z_]\w*)\s*"                     # method name
    r"\(([^;{]*?)\)"                         # params (no ; or { inside)
    r"(?:\s*throws\s+[\w.,\s]+)?\s*[{;]"     # body-open or iface-decl end
)
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "new", "synchronized"}


def _collect_annotations(text: str, start: int) -> list[str]:
    """Grab annotations on the (possibly multiple) lines directly above `start`."""
    prefix = text[:start].rstrip()
    anns: list[str] = []
    # walk backwards over lines that are annotations or blank
    for line in reversed(prefix.splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith("@"):
            anns.insert(0, _ANNOTATION.search(s).group(0) if _ANNOTATION.search(s) else s)
        else:
            break
    return anns


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


class JavaParser:
    language = "java"

    def parse(self, rel_path: str, text: str) -> ParsedFile:
        pkg_m = _PACKAGE.search(text)
        module = pkg_m.group(1) if pkg_m else ""
        imports = _IMPORT.findall(text)

        types: list[TypeDecl] = []
        for tm in _TYPE_DECL.finditer(text):
            kind, name = tm.group(1), tm.group(2)
            decl = TypeDecl(
                name=name,
                kind=kind,
                annotations=_collect_annotations(text, tm.start()),
                start_line=_line_of(text, tm.start()),
            )
            types.append(decl)

        # assign methods to the nearest preceding type declaration
        type_starts = [(t.start_line, t) for t in types]
        for mm in _METHOD.finditer(text):
            ret, name, params = mm.group(1).strip(), mm.group(2), mm.group(3).strip()
            if name in _KEYWORDS or ret in _KEYWORDS or ret.endswith("="):
                continue
            if ret in ("", "return", "new"):
                continue
            line = _line_of(text, mm.start())
            owner = None
            for tstart, t in type_starts:
                if tstart <= line:
                    owner = t
                else:
                    break
            if owner is None:
                continue
            params_norm = re.sub(r"\s+", " ", params)
            sig = f"{ret} {name}({params_norm})".strip()
            owner.methods.append(
                MethodSig(
                    name=name,
                    signature=sig,
                    annotations=_collect_annotations(text, mm.start()),
                    start_line=line,
                )
            )

        return ParsedFile(
            rel_path=rel_path, language="java", module=module,
            imports=imports, types=types, parser="java-regex",
        )
