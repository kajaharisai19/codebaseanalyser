"""AST-based structural parser for Python (stdlib `ast` — exact, no heuristics)."""
from __future__ import annotations

import ast

from .base import MethodSig, ParsedFile, TypeDecl


def _sig(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = []
    a = fn.args
    posonly = getattr(a, "posonlyargs", [])
    for arg in posonly + a.args:
        args.append(arg.arg + (f": {ast.unparse(arg.annotation)}" if arg.annotation else ""))
    if a.vararg:
        args.append("*" + a.vararg.arg)
    for arg in a.kwonlyargs:
        args.append(arg.arg + (f": {ast.unparse(arg.annotation)}" if arg.annotation else ""))
    if a.kwarg:
        args.append("**" + a.kwarg.arg)
    ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    prefix = "async def " if isinstance(fn, ast.AsyncFunctionDef) else "def "
    return f"{prefix}{fn.name}({', '.join(args)}){ret}"


def _decorators(node) -> list[str]:
    out = []
    for d in getattr(node, "decorator_list", []):
        try:
            out.append("@" + ast.unparse(d))
        except Exception:
            pass
    return out


class PythonParser:
    language = "python"

    def parse(self, rel_path: str, text: str) -> ParsedFile:
        module = rel_path.replace("/", ".").rsplit(".py", 1)[0]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # unparseable file: return an empty-but-valid shell
            return ParsedFile(rel_path, "python", module, parser="python-ast(failed)")

        imports: list[str] = []
        types: list[TypeDecl] = []
        module_fns: list[MethodSig] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                imports += [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.ClassDef):
                methods = [
                    MethodSig(
                        name=b.name, signature=_sig(b),
                        annotations=_decorators(b), start_line=b.lineno,
                    )
                    for b in node.body
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                types.append(
                    TypeDecl(
                        name=node.name, kind="class",
                        annotations=_decorators(node), methods=methods,
                        start_line=node.lineno,
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_fns.append(
                    MethodSig(name=node.name, signature=_sig(node),
                              annotations=_decorators(node), start_line=node.lineno)
                )

        if module_fns:
            types.insert(0, TypeDecl(name="(module)", kind="module", methods=module_fns))

        return ParsedFile(
            rel_path=rel_path, language="python", module=module,
            imports=imports, types=types, parser="python-ast",
        )
