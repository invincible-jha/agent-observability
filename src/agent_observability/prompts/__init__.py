"""agent_observability.prompts — Prompt template management and versioning.

Provides a registry for storing, versioning, rendering, and diffing
prompt templates.  Backed by in-memory storage by default; extension
points exist for persistent backends.

Public surface
--------------
PromptRegistry
    Central store for prompt templates with version tracking.
PromptVersion
    Metadata for a single stored version of a prompt template.
PromptTemplate
    A resolved prompt template ready for rendering.
PromptDiff
    The result of comparing two versions of the same prompt.
PromptRegistryError
    Base exception for all errors raised by this sub-package.
PromptNotFoundError
    Raised when the requested prompt name does not exist.
PromptVersionNotFoundError
    Raised when the requested version number does not exist.

Example
-------
::

    from agent_observability.prompts import PromptRegistry

    registry = PromptRegistry()
    registry.register(
        "greet",
        "Hello, {name}! You are {age} years old.",
        variables=["name", "age"],
    )
    text = registry.render("greet", name="Alice", age=30)
    # "Hello, Alice! You are 30 years old."
"""
from __future__ import annotations

from agent_observability.prompts.prompt_registry import (
    PromptDiff,
    PromptNotFoundError,
    PromptRegistryError,
    PromptTemplate,
    PromptVersion,
    PromptVersionNotFoundError,
    PromptRegistry,
)

__all__ = [
    "PromptDiff",
    "PromptNotFoundError",
    "PromptRegistry",
    "PromptRegistryError",
    "PromptTemplate",
    "PromptVersion",
    "PromptVersionNotFoundError",
]
