"""
Run:  PYTHONPATH=src python -m pytest tests/ -q
"""
from __future__ import annotations

import pytest

from codebase_analyzer.chunker import Chunker
from codebase_analyzer.config import build_budget
from codebase_analyzer.llm_client import BudgetExceededError, make_client
from codebase_analyzer.parsers import parse_file, detect_primary_language
from codebase_analyzer.tokenizer import get_tokenizer

JAVA = """package com.example.demo;
import java.util.List;

@RestController
public class DemoController {
    @GetMapping("/x")
    public List<String> getX(int a) { return null; }
    public void setY(String b) {}
}
"""


def test_java_parser_signatures():
    pf = parse_file("Demo.java", ".java", JAVA)
    assert pf.module == "com.example.demo"
    assert pf.method_count == 2
    names = {m.name for t in pf.types for m in t.methods}
    assert names == {"getX", "setY"}
    assert "@RestController" in pf.types[0].annotations


def test_python_parser_is_exact():
    src = "import os\n\nclass A:\n    def f(self, x: int) -> str:\n        return ''\n"
    pf = parse_file("a.py", ".py", src)
    assert pf.method_count == 1
    assert pf.types[0].methods[0].signature == "def f(self, x: int) -> str"


def test_language_detection():
    assert detect_primary_language([".java", ".java", ".py"]) == "java"


def test_budget_gate_blocks_oversized_prompt():
    budget = build_budget("mock")
    budget.context_window = 1000  # force a tiny budget
    client = make_client("mock", "mock", budget)
    huge = "word " * 5000
    with pytest.raises(BudgetExceededError):
        client.complete("sys", huge)


def test_chunker_never_exceeds_budget():
    tok = get_tokenizer("mock")
    chunker = Chunker(tok, code_budget=200)
    pf = parse_file("Big.java", ".java", JAVA)
    big_text = JAVA * 200  # force splitting
    chunks = chunker.chunk_file(pf, big_text)
    assert len(chunks) > 1
    assert all(c.token_count <= chunker.budget for c in chunks)


def test_mock_client_returns_valid_schema():
    from codebase_analyzer.extractor import _extract_json
    from codebase_analyzer.schema import FileExtraction

    client = make_client("mock", "mock", build_budget("mock"))
    resp = client.complete("sys", "FILE: x.java\ncode")
    FileExtraction.model_validate(_extract_json(resp.text))  # must not raise
