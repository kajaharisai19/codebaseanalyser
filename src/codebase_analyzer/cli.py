from __future__ import annotations

import argparse
import json
import os
import sys

from .config import AnalyzerConfig, build_budget
from .pipeline import run


def build_config(args) -> AnalyzerConfig:
    cfg = AnalyzerConfig(
        repo_path=args.repo,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
        max_files=args.max_files,
        workers=args.workers,
    )
    cfg.budget = build_budget(args.model, max_output_tokens=args.max_output_tokens)
    return cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="codebase-analyzer",
        description="Analyze a codebase with an LLM and emit machine-readable JSON.",
    )
    p.add_argument("repo", help="local path or git URL of the repo to analyze")
    p.add_argument("-o", "--output", default="output/codebase_knowledge.json")
    p.add_argument("-b", "--backend", default="mock",
                   choices=["mock", "ollama", "anthropic", "openai"])
    p.add_argument("-m", "--model", default="mock",
                   help="model id (e.g. qwen2.5, claude-opus-4-8, gpt-4o)")
    p.add_argument("--max-files", type=int, default=0, help="limit files (0 = all)")
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent extract-stage LLM calls (default 4)")
    p.add_argument("--max-output-tokens", type=int, default=2048)
    args = p.parse_args(argv)

    cfg = build_config(args)
    try:
        result = run(cfg)
    except Exception as e:  # noqa: BLE001 — top-level guard for a clean CLI exit
        print(f"[cba] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    r = result["run"]
    print(f"[cba] wrote {cfg.output_path} "
          f"({r['chunks_ok']}/{r['chunks']} chunks ok, {r['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
