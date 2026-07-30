from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


# Nominal context windows (input+output) per known model. For models whose exact
# tokenizer we cannot run locally (Claude, Ollama), counts are approximate and we
# deliberately budget to a fraction of nominal (see `approx_factor`).
_MODEL_CONTEXT: Dict[str, int] = {
    # OpenAI (exact tokenizer available via tiktoken)
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_000_000,
    # Anthropic (approximate — no local exact tokenizer)
    "claude-opus-4-8": 200_000,
    "claude-sonnet-5": 200_000,
    "claude-haiku-4-5": 200_000,
    # Ollama local models (approximate)
    "qwen2.5": 32_768,
    "qwen2.5-coder": 32_768,
    "gemma3": 8_192,
    "llama3.2": 131_072,
    "mock": 32_768,
}

# Model families whose token counts we can only approximate. For these we scale
# the usable budget down (tokenizer-mismatch caveat from the design).
_APPROXIMATE_FAMILIES = ("claude", "qwen", "gemma", "llama", "mock")


def _is_approximate(model: str) -> bool:
    m = model.lower()
    return any(fam in m for fam in _APPROXIMATE_FAMILIES)


@dataclass
class BudgetConfig:
    model: str
    context_window: int
    max_output_tokens: int
    fixed_prompt_overhead: int  # system prompt + schema instructions, measured
    safety_margin: int
    approx_factor: float  # <1.0 for models we can only approximate

    @property
    def input_budget(self) -> int:
        """Max tokens allowed for the *entire* input prompt of a single call."""
        raw = (
            self.context_window
            - self.max_output_tokens
            - self.fixed_prompt_overhead
            - self.safety_margin
        )
        return max(0, int(raw * self.approx_factor))

    def code_budget(self, dynamic_overhead: int = 0) -> int:
        """Budget left for code/context after subtracting per-call dynamic parts
        (e.g. the parsed skeleton, or RAG-retrieved context)."""
        return max(0, self.input_budget - dynamic_overhead)


def build_budget(
    model: str,
    max_output_tokens: int = 2_048,
    fixed_prompt_overhead: int = 900,
    safety_margin: int = 512,
) -> BudgetConfig:
    """Compute the token budget for `model`. Unknown models fall back to a
    conservative 8k window so we never assume more headroom than we have."""
    key = _resolve_key(model)
    context = _MODEL_CONTEXT.get(key, 8_192)
    approx = 0.85 if _is_approximate(model) else 1.0
    return BudgetConfig(
        model=model,
        context_window=context,
        max_output_tokens=max_output_tokens,
        fixed_prompt_overhead=fixed_prompt_overhead,
        safety_margin=safety_margin,
        approx_factor=approx,
    )


def _resolve_key(model: str) -> str:
    """Map a concrete model id (e.g. 'qwen2.5:latest') to a context-window key."""
    m = model.lower()
    if m in _MODEL_CONTEXT:
        return m
    # strip ollama ':tag' and version suffixes, longest-prefix match
    base = m.split(":")[0]
    if base in _MODEL_CONTEXT:
        return base
    for known in sorted(_MODEL_CONTEXT, key=len, reverse=True):
        if base.startswith(known) or m.startswith(known):
            return known
    return base


@dataclass
class AnalyzerConfig:

    repo_path: str
    output_path: str = "output/codebase_knowledge.json"
    backend: str = "mock"            # mock | ollama | anthropic | openai
    model: str = "mock"
    max_files: int = 0               # 0 = all
    workers: int = 4                 # concurrent extract-stage LLM calls
    source_extensions: tuple = field(
        default_factory=lambda: (
            ".java", ".py", ".js", ".ts", ".go", ".rb", ".cpp", ".c", ".cs",
            ".kt", ".rs", ".php", ".scala", ".swift",
        )
    )
    budget: BudgetConfig = field(default=None)  # populated in __post_init__

    def __post_init__(self):
        if self.budget is None:
            self.budget = build_budget(self.model)
