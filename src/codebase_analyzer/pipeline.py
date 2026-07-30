from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from .aggregator import Aggregator
from .chunker import Chunker
from .config import AnalyzerConfig
from .extractor import Extractor
from .llm_client import make_client
from .loader import load_repo
from .parsers import detect_primary_language, parse_file
from .schema import FILE_SCHEMA_HINT
from .tokenizer import get_tokenizer


def _log(msg: str) -> None:
    print(f"[cba] {msg}", flush=True)


def run(config: AnalyzerConfig) -> dict:
    t0 = time.time()
    tok = get_tokenizer(config.model)
    client = make_client(config.backend, config.model, config.budget)

    # measure the true fixed prompt overhead (system + schema) once, and fold it
    # into the budget so the code_budget reflects reality.
    fixed = tok.count(FILE_SCHEMA_HINT) + 400  # schema hint + template scaffolding
    config.budget.fixed_prompt_overhead = max(config.budget.fixed_prompt_overhead, fixed)

    # 1) LOAD
    repo = load_repo(config.repo_path, config.source_extensions, config.max_files)
    _log(f"loaded {len(repo.files)} source files from {repo.root}")
    if not repo.files:
        raise RuntimeError("no source files found (check extensions / exclusions)")

    primary = detect_primary_language([f.ext for f in repo.files])
    _log(f"primary language: {primary} | model={config.model} backend={config.backend} "
         f"input_budget={config.budget.input_budget} tok")

    # 2) PARSE
    parsed_files = [parse_file(f.rel_path, f.ext, f.text) for f in repo.files]

    # 3) CHUNK — reserve room for the skeleton that rides in each prompt
    chunker = Chunker(tok, config.budget.input_budget)
    chunks = []
    for f, pf in zip(repo.files, parsed_files):
        skel_tokens = tok.count(pf.skeleton())
        chunker.budget = max(256, int((config.budget.code_budget(skel_tokens)) * 0.90))
        chunks.extend(chunker.chunk_file(pf, f.text))
    split_files = sum(1 for c in chunks if c.n_parts > 1)
    _log(f"parsed {len(parsed_files)} files -> {len(chunks)} chunks "
         f"({split_files} chunks from oversized files)")

    # 4) EXTRACT — embarrassingly parallel; run concurrently across a thread pool.
    # Each chunk's LLM call is independent, so we fan out and keep input order.
    extractor = Extractor(client)
    workers = max(1, config.workers)
    done = [0]

    def _extract(ch):
        res = extractor.extract(ch)
        done[0] += 1  # progress only (GIL makes the increment safe enough here)
        i = done[0]
        if i % 10 == 0 or not res.ok:
            status = "ok" if res.ok else f"FAIL({(res.error or '')[:60]})"
            _log(f"extract {i}/{len(chunks)} {res.chunk_id} [{status}]")
        return res

    _log(f"extracting {len(chunks)} chunks with {workers} worker(s)...")
    if workers == 1:
        results = [_extract(ch) for ch in chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_extract, chunks))
    n_ok = sum(1 for r in results if r.ok)
    _log(f"extracted {n_ok}/{len(results)} chunks ok")

    # 5) AGGREGATE
    agg = Aggregator(client, tok, config.budget)
    files_section = agg.merge(results)
    _log("merged per-file results; synthesizing project overview...")
    overview = agg.project_summary(results)
    stats = agg.parser_stats(parsed_files)

    elapsed = round(time.time() - t0, 1)
    return {
        "schema_version": "1.0",
        "generated_by": "codebase-analyzer",
        "repo": {
            "root": repo.root,
            "primary_language": primary,
            "has_readme": repo.readme is not None,
        },
        "run": {
            "backend": config.backend,
            "model": config.model,
            "input_token_budget": config.budget.input_budget,
            "elapsed_seconds": elapsed,
            "chunks": len(chunks),
            "chunks_ok": n_ok,
            "chunks_failed": len(results) - n_ok,
        },
        "parser_stats": stats,
        "project_overview": overview.model_dump(),
        "files": list(files_section.values()),
        "errors": [
            {"chunk": r.chunk_id, "error": r.error} for r in results if not r.ok
        ],
    }
