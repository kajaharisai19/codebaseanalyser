from __future__ import annotations

import json
from collections import Counter

from .config import BudgetConfig
from .extractor import ChunkResult
from .llm_client import LLMClient
from .schema import PROJECT_SCHEMA_HINT, ProjectOverview
from .tokenizer import Tokenizer

SUMMARY_SYSTEM = (
    "You are a software architect. Synthesize a project-level overview from "
    "per-file summaries. Base every statement on the provided summaries only. "
    "Respond with a single JSON object and nothing else."
)


def _digest(res: ChunkResult) -> str:
    """One compact line of aggregated knowledge for a file."""
    if not res.ok or res.data is None:
        return f"- {res.rel_path}: [extraction failed]"
    d = res.data
    note = f" | noteworthy: {'; '.join(d.noteworthy[:2])}" if d.noteworthy else ""
    return f"- {res.rel_path}: {d.purpose}{note}"


def _reduce_prompt(digests_block: str, is_partial: bool) -> str:
    unit = "per-file summaries" if not is_partial else "partial overviews"
    return (
        f"Below are {unit} for a codebase.\n\n{digests_block}\n\n"
        f"Produce a JSON object with EXACTLY this shape:\n{PROJECT_SCHEMA_HINT}\n"
        "Base it only on the information above. No markdown fences."
    )


class Aggregator:
    def __init__(self, client: LLMClient, tokenizer: Tokenizer, budget: BudgetConfig):
        self.client = client
        self.tok = tokenizer
        self.budget = budget

    def merge(self, results: list[ChunkResult]) -> dict:
        """Combine per-chunk results into the files section + parser-derived stats."""
        files = {}
        for r in results:
            entry = files.setdefault(
                r.rel_path,
                {"rel_path": r.rel_path, "extractions": [], "ok": True, "errors": []},
            )
            if r.ok and r.data is not None:
                entry["extractions"].append(r.data.model_dump())
            else:
                entry["ok"] = False
                entry["errors"].append(r.error)
        return files

    def _batch_digests(self, digests: list[str], reserve: int) -> list[list[str]]:
        """Group digest lines into budget-sized batches."""
        limit = int(self.budget.input_budget * 0.85) - reserve
        batches, cur, cur_tok = [], [], 0
        for d in digests:
            t = self.tok.count(d) + 1
            if cur and cur_tok + t > limit:
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(d)
            cur_tok += t
        if cur:
            batches.append(cur)
        return batches

    def _summarize_block(self, block: str, is_partial: bool) -> ProjectOverview:
        from .extractor import _extract_json  # reuse tolerant parser

        resp = self.client.complete(SUMMARY_SYSTEM, _reduce_prompt(block, is_partial))
        try:
            return ProjectOverview.model_validate(_extract_json(resp.text))
        except Exception:
            return ProjectOverview(summary=resp.text[:1000])

    def project_summary(self, results: list[ChunkResult]) -> ProjectOverview:
        digests = [_digest(r) for r in results]
        reserve = self.tok.count(_reduce_prompt("", False)) + 200
        batches = self._batch_digests(digests, reserve)

        if len(batches) == 1:
            return self._summarize_block("\n".join(batches[0]), is_partial=False)

        partials = [
            self._summarize_block("\n".join(b), is_partial=False) for b in batches
        ]
        partial_block = "\n\n".join(
            f"[batch {i+1}] {json.dumps(p.model_dump())}" for i, p in enumerate(partials)
        )
        return self._summarize_block(partial_block, is_partial=True)

    @staticmethod
    def parser_stats(parsed_files) -> dict:
        by_lang = Counter(p.language for p in parsed_files)
        return {
            "file_count": len(parsed_files),
            "total_types": sum(len(p.types) for p in parsed_files),
            "total_methods": sum(p.method_count for p in parsed_files),
            "languages": dict(by_lang),
        }
