# Codebase Analyzer

An LLM-driven pipeline that reads a codebase, extracts structural knowledge, and
emits a single well-structured, machine-readable JSON document describing the
project: an overview, per-file/method signatures and descriptions, and
complexity/noteworthy aspects.

```
load → parse → chunk → extract → aggregate
```

*Approximate time spent: ~1 day of development. (The ~11-hour local `qwen2.5` run
that produced the committed output was unattended and isn't counted in that.)*

---

## Quick start

```bash
pip install -r requirements.txt

# Offline sanity check — no server or API key required (mock backend):
python -m codebase_analyzer.cli ./repo --backend mock --model mock -o output/mock.json

# Real extraction with a local Ollama model (free, offline):
#   1. install & start ollama, then: ollama pull qwen2.5
python -m codebase_analyzer.cli ./repo --backend ollama --model qwen2.5 \
    -o output/codebase_knowledge.json

# Or a cloud model (needs the matching API key in the environment):
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY
python -m codebase_analyzer.cli ./repo --backend anthropic --model claude-opus-4-8
python -m codebase_analyzer.cli https://github.com/codejsha/spring-rest-sakila --backend openai --model gpt-4o
```

`--max-files N` limits the run for quick iteration. `--workers N` sets extract-stage
concurrency (default 4). The first positional argument may be a **local path or a
git URL** (cloned to a temp dir automatically).

Set `PYTHONPATH=src` (or `pip install -e .`) so `codebase_analyzer` is importable.

### Performance / concurrency
The extract stage is embarrassingly parallel (one independent LLM call per file),
so it runs across a thread pool (`--workers`). The aggregate pass stays sequential.
- **Cloud backends (anthropic/openai):** near-linear speedup — the bottleneck is
  network latency, so more workers ≈ proportionally faster.
- **Local Ollama on a single GPU:** modest speedup (~20–30%). One model instance on
  one GPU largely serializes compute, so the GPU — not request latency — is the
  bottleneck; multi-GPU or a smaller model helps more.
- Keep the model warm on the **server** side to avoid reload cost:
  `OLLAMA_KEEP_ALIVE=30m ollama serve`.

---

## Approach & methodology

The task is **read code → extract facts → aggregate facts**. That is
deterministic batch ETL, not a problem that needs dynamic planning or tool
selection, so the core is a plain linear pipeline. Each of the five stages is a
small, independently testable module.

### 1. Load — `loader.py`
Clones (git URL) or reads (local path) the repo, walks source files by extension,
and prunes test/build/vendor directories (`target/`, `build/`, `node_modules/`,
`test/`, …) in-place during the walk. It also grabs the repo's own README as
extra project context. Works against any repo, not just the target.

### 2. Parse — `parsers/`
Extracts **structural ground truth before the LLM is ever called**:
package/module, imports, type declarations with annotations, and method
signatures. This does double duty:
- a **compact skeleton** primes the extraction prompt (fewer hallucinations,
  higher accuracy);
- **counts** (types, methods) are true regardless of whether the LLM succeeds,
  so the JSON always carries non-LLM facts.

Per-language strategy, auto-selected — no CLI flag needed:
| Language | Strategy | Fidelity |
|---|---|---|
| Java | regex + brace tracking (`java_parser.py`) | high for signatures/annotations; not a full grammar |
| Python | stdlib `ast` (`python_parser.py`) | exact |
| everything else | broad regex fallback (`generic_parser.py`) | best-effort function list |

Primary language is auto-detected by **extension frequency** across the repo.

### 3. Chunk — `chunker.py`
Token-bounded and **verified with a real tokenizer**, never char-count estimation.
Strategy, in order:
1. **one chunk per file** — the default unit; matches 1-class-per-file
   conventions and makes error attribution trivial (we know exactly which file
   failed);
2. if a file exceeds budget → **syntactic split** on class/method boundaries via
   LangChain's language-aware `RecursiveCharacterTextSplitter.from_language`,
   using the tokenizer as its length function;
3. if a single syntactic piece is still too big → **hard-slice by token index**
   as a last resort, so a chunk can *never* silently exceed the cap.

### 4. Extract — `extractor.py` + `schema.py`
One strict-JSON-schema LLM call per chunk. The prompt carries the parser skeleton
alongside the source and instructs the model to report **only methods present in
the source** (anti-hallucination). The schema covers purpose, methods
(name/signature/description), dependencies/collaborators, complexity notes, and
noteworthy aspects (security, validation, caching, transactional boundaries, …).
The response is parsed tolerantly (handles code fences), **validated against a
Pydantic model**, and **retried once** with a corrective instruction on failure.
Failures are attributed per chunk, never fatal to the run.

### 5. Aggregate — `aggregator.py`
Merges per-chunk results into the `files` section, then a **second LLM pass
synthesizes the project overview** (purpose, architecture style, key modules,
recurring patterns, complexity) **from the aggregated per-file digests, not raw
source**. That pass has its own budget problem at scale, handled with
**map-reduce**: digests are batched under the token budget, each batch summarized
to a partial overview, and the partials reduced to the final overview. Also emits
**parser-derived stats** (file/type/method counts) as LLM-independent ground truth.

### LLM backend abstraction — `llm_client.py`
Pluggable across `mock` / `ollama` / `anthropic` / `openai` behind one
`.complete(system, user)` contract. The **token-budget gate lives in the base
class and runs immediately before every concrete call**, so no backend can bypass
it (raises `BudgetExceededError` instead of silently truncating).

### Token-budget enforcement — `config.py` + `tokenizer.py`
A per-model budget is computed **once at startup**:
```
input_budget = (context_window − max_output_tokens − fixed_prompt_overhead − safety_margin) × approx_factor
```
The check covers the **whole prompt** (system + skeleton + schema + code +
any retrieved context), not just raw code size. The `fixed_prompt_overhead` is
*measured* from the real system prompt + schema hint at startup, not guessed.

**Why LangChain here:** the chunker uses LangChain's language-aware splitter for
syntactic-boundary splitting across ~15 languages, driven by the real tokenizer
as its length function. (LangChain/LlamaIndex earn a larger role in the planned
RAG phase — semantic related-file retrieval and retrieval-based summary synthesis.)

---

## Output format

Single JSON document (`output/codebase_knowledge.json`):
```jsonc
{
  "schema_version": "1.0",
  "repo": { "root", "primary_language", "has_readme" },
  "run": { "backend", "model", "input_token_budget", "elapsed_seconds",
           "chunks", "chunks_ok", "chunks_failed" },
  "parser_stats": { "file_count", "total_types", "total_methods", "languages" },
  "project_overview": { "summary", "architecture_style", "key_modules",
                        "recurring_patterns", "complexity_assessment" },
  "files": [ { "rel_path", "ok",
               "extractions": [ { "purpose", "methods":[{name,signature,description}],
                                  "dependencies", "complexity_notes", "noteworthy" } ],
               "errors": [] } ],
  "errors": [ { "chunk", "error" } ]   // per-chunk failure attribution
}
```

---

## Best practices followed
- **Real tokenizer**, not char estimates, for every budget decision.
- **Whole-prompt** budgeting with a startup-computed per-model budget and a
  hard gate before every call.
- **Structural ground truth before the LLM** — parser counts don't depend on LLM
  success, and prime the prompt to reduce hallucination.
- **Strict schema validation + retry** with **per-chunk error attribution**.
- **File = chunk unit** for clean error attribution; graceful degradation (one
  bad file doesn't sink the run).
- **Pluggable backends** incl. a **mock** for CI/offline verification.
- Deterministic settings (`temperature=0`) for reproducibility.

---

## Assumptions & limitations
- **Parser accuracy trade-off.** Java is parsed with **regex + brace tracking**,
  not a full grammar (Python's stdlib has no Java AST, and a real multi-language
  library like tree-sitter was intentionally out of scope for this phase). It
  reliably recovers package, imports, type declarations, annotations, and method
  signatures, but does **not** fully resolve generics, may miss methods with
  unusual formatting, and builds no real type graph. Python uses the exact stdlib
  `ast`; all other languages use a best-effort regex fallback. Swapping in
  tree-sitter per language is the clean upgrade path and would not change the
  pipeline's interfaces.
- **Tokenizer approximation caveat.** Exact token counts are only available for
  OpenAI-family models (via `tiktoken`). For Claude and local models we use
  `cl100k_base` as a **proxy** and deliberately budget to **~85% of nominal**
  (`approx_factor`) rather than the literal limit, to absorb the mismatch.
- **LLM fidelity.** Descriptions and the project overview are model-generated;
  the prompt constrains the model to source-present facts and the schema is
  validated, but natural-language descriptions are not formally verified. Smaller
  local models (e.g. `qwen2.5` 7B) produce good but occasionally shallow prose;
  a cloud model raises quality at cost.
- **Chunk-level context.** Extraction is per-file, so cross-file relationships
  beyond imports are not captured in this phase (the planned RAG layer adds
  semantic related-file retrieval for exactly this).
- **Scope.** This is Phase 1 — the linear pipeline that fully satisfies the
  assignment on its own. Planned extensions (Ollama+cloud validation hybrid,
  LangGraph routing/retry, RAG context + Q&A, broader multi-language parsing)
  build on top without changing these interfaces.

---

## Layout
```
src/codebase_analyzer/
  config.py        # per-model token budget
  tokenizer.py     # real tokenizer wrappers
  llm_client.py    # mock/ollama/anthropic/openai + budget gate
  loader.py        # repo walk + README
  parsers/         # java (regex), python (ast), generic (fallback) + detection
  chunker.py       # file-first, syntactic split, hard-slice fallback
  schema.py        # pydantic extraction schema
  extractor.py     # per-chunk extract + validate + retry
  aggregator.py    # merge + map-reduce project summary
  pipeline.py      # 5-stage orchestration
  cli.py           # entrypoint
output/codebase_knowledge.json   # produced by a real run
```
