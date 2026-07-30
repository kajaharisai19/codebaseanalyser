from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .chunker import Chunk
from .llm_client import BudgetExceededError, LLMClient
from .schema import FILE_SCHEMA_HINT, FileExtraction

SYSTEM_PROMPT = (
    "You are a senior software engineer analyzing source code. "
    "Extract ONLY facts present in the provided code — never invent methods, "
    "parameters, or behavior. Report methods that actually appear. "
    "Respond with a single JSON object and nothing else."
)

_USER_TEMPLATE = """FILE: {path}
LANGUAGE: {language}
{part_note}

STRUCTURAL SKELETON (ground truth extracted by a parser — use to stay accurate):
{skeleton}

SOURCE:
```{language}
{code}
```

Return a JSON object with EXACTLY this shape:
{schema}

Rules:
- "methods": one entry per method/function actually defined in the SOURCE above.
- "signature": copy the real signature; do not invent parameters.
- "dependencies": key types/services this code collaborates with.
- "noteworthy": framework/security/validation/caching/transaction concerns if present, else [].
- Do not include markdown fences in your response."""


@dataclass
class ChunkResult:
    chunk_id: str
    rel_path: str
    ok: bool
    data: FileExtraction | None
    error: str | None
    prompt_tokens: int = 0
    attempts: int = 0


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, depth = text.find("{"), 0
        for i in range(start, len(text)) if start >= 0 else []:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
        raise


class Extractor:
    def __init__(self, client: LLMClient, max_retries: int = 1):
        self.client = client
        self.max_retries = max_retries

    def _build_user(self, chunk: Chunk) -> str:
        part_note = (
            f"(part {chunk.part + 1} of {chunk.n_parts} — this file was split)"
            if chunk.n_parts > 1 else ""
        )
        return _USER_TEMPLATE.format(
            path=chunk.rel_path, language=chunk.language, part_note=part_note,
            skeleton=chunk.parsed.skeleton(), code=chunk.text,
            schema=FILE_SCHEMA_HINT,
        )

    def extract(self, chunk: Chunk) -> ChunkResult:
        user = self._build_user(chunk)
        last_err = ""
        prompt_tokens = 0
        for attempt in range(1, self.max_retries + 2):
            try:
                suffix = "" if attempt == 1 else (
                    "\n\nYour previous reply was not valid JSON for the schema. "
                    "Reply with ONLY the JSON object."
                )
                resp = self.client.complete(SYSTEM_PROMPT, user + suffix)
                prompt_tokens = resp.prompt_tokens
                data = FileExtraction.model_validate(_extract_json(resp.text))
                return ChunkResult(
                    chunk_id=chunk.chunk_id, rel_path=chunk.rel_path, ok=True,
                    data=data, error=None, prompt_tokens=prompt_tokens, attempts=attempt,
                )
            except BudgetExceededError as e:
                # non-retryable: chunker should have prevented this
                return ChunkResult(chunk.chunk_id, chunk.rel_path, False, None,
                                   f"budget: {e}", prompt_tokens, attempt)
            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                last_err = f"{type(e).__name__}: {e}"
            except Exception as e:  # network / backend errors
                last_err = f"{type(e).__name__}: {e}"
        return ChunkResult(chunk.chunk_id, chunk.rel_path, False, None,
                           last_err, prompt_tokens, self.max_retries + 1)
