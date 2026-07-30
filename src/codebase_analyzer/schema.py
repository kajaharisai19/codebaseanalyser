from __future__ import annotations

from pydantic import BaseModel, Field


class MethodDoc(BaseModel):
    name: str
    signature: str = ""
    description: str = ""


class FileExtraction(BaseModel):
    """Per-file (per-chunk) extracted knowledge."""

    purpose: str = Field(default="", description="What this file/class is for")
    methods: list[MethodDoc] = Field(default_factory=list)
    dependencies: list[str] = Field(
        default_factory=list, description="Key collaborators / dependencies"
    )
    complexity_notes: str = Field(default="")
    noteworthy: list[str] = Field(
        default_factory=list,
        description="Security, validation, caching, transactional boundaries, etc.",
    )


class ProjectOverview(BaseModel):
    """Project-level synthesis produced by the aggregation pass."""

    summary: str = ""
    architecture_style: str = ""
    key_modules: list[str] = Field(default_factory=list)
    recurring_patterns: list[str] = Field(default_factory=list)
    complexity_assessment: str = ""


# JSON-schema instruction text injected into prompts (kept terse to save tokens).
FILE_SCHEMA_HINT = (
    '{"purpose": str, "methods": [{"name": str, "signature": str, '
    '"description": str}], "dependencies": [str], "complexity_notes": str, '
    '"noteworthy": [str]}'
)

PROJECT_SCHEMA_HINT = (
    '{"summary": str, "architecture_style": str, "key_modules": [str], '
    '"recurring_patterns": [str], "complexity_assessment": str}'
)
