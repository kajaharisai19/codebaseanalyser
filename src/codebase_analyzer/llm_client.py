"""LLM backend abstraction with a token-budget gate.

Every backend implements the same `.complete(system, user)` contract. The gate is
enforced in the base class immediately before the concrete call, so no backend
can bypass it. Backends:
  - mock     : deterministic, offline; echoes a schema-valid stub. For pipeline
               verification without a server or key.
  - ollama   : local models via the HTTP API (free, offline-capable).
  - anthropic: Claude, if ANTHROPIC_API_KEY is set.
  - openai   : GPT, if OPENAI_API_KEY is set.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from .config import BudgetConfig
from .tokenizer import get_tokenizer


class BudgetExceededError(RuntimeError):
    """Raised when a prompt would exceed the model's computed input budget."""


@dataclass
class LLMResponse:
    text: str
    backend: str
    model: str
    prompt_tokens: int


class LLMClient(ABC):
    """Base class enforcing the token-budget gate around every call."""

    def __init__(self, model: str, budget: BudgetConfig):
        self.model = model
        self.budget = budget
        self._tok = get_tokenizer(model)

    def count_prompt_tokens(self, system: str, user: str) -> int:
        # +8 accounts for chat role/formatting tokens the API adds around content.
        return self._tok.count(system) + self._tok.count(user) + 8

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> LLMResponse:
        """Budget-gate then dispatch to the concrete backend."""
        n = self.count_prompt_tokens(system, user)
        if n > self.budget.input_budget:
            raise BudgetExceededError(
                f"prompt is {n} tokens; budget is {self.budget.input_budget} "
                f"(model={self.model}). Chunk smaller."
            )
        text = self._complete(system, user, max_tokens or self.budget.max_output_tokens)
        return LLMResponse(text=text, backend=self.name, model=self.model, prompt_tokens=n)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def _complete(self, system: str, user: str, max_tokens: int) -> str: ...


class MockClient(LLMClient):
    name = "mock"

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        # Deterministic, schema-shaped stub so the pipeline runs fully offline.
        # Pull the file path out of the user prompt if present for realism.
        path = "unknown"
        for line in user.splitlines():
            if line.startswith("FILE:"):
                path = line.split("FILE:", 1)[1].strip()
                break
        return json.dumps(
            {
                "purpose": f"[mock] stub summary for {path}",
                "methods": [],
                "dependencies": [],
                "complexity_notes": "[mock] not analyzed",
                "noteworthy": [],
            }
        )


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, model: str, budget: BudgetConfig, host: str = "http://localhost:11434"):
        super().__init__(model, budget)
        self.host = host.rstrip("/")

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",  # ask Ollama to constrain output to JSON
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: str, budget: BudgetConfig):
        super().__init__(model, budget)
        import anthropic  # lazy import

        self._client = anthropic.Anthropic()

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAIClient(LLMClient):
    name = "openai"

    def __init__(self, model: str, budget: BudgetConfig):
        super().__init__(model, budget)
        import openai  # lazy import

        self._client = openai.OpenAI()

    def _complete(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


def make_client(backend: str, model: str, budget: BudgetConfig) -> LLMClient:
    backend = backend.lower()
    if backend == "mock":
        return MockClient(model, budget)
    if backend == "ollama":
        return OllamaClient(model, budget)
    if backend == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return AnthropicClient(model, budget)
    if backend == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        return OpenAIClient(model, budget)
    raise ValueError(f"unknown backend: {backend}")
