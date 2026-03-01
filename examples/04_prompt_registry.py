#!/usr/bin/env python3
"""Example: Prompt Registry

Demonstrates versioned prompt management using PromptRegistry —
track, compare, and diff prompt templates across versions.

Usage:
    python examples/04_prompt_registry.py

Requirements:
    pip install agent-observability
"""
from __future__ import annotations

import agent_observability
from agent_observability import (
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
)


SYSTEM_PROMPT_V1 = """\
You are a helpful assistant. Answer questions concisely.
"""

SYSTEM_PROMPT_V2 = """\
You are a helpful, accurate, and concise assistant.
Always cite sources when making factual claims.
Format responses in markdown when appropriate.
"""


def main() -> None:
    print(f"agent-observability version: {agent_observability.__version__}")

    # Step 1: Create a prompt registry
    registry = PromptRegistry()

    # Step 2: Register versioned prompts
    template = PromptTemplate(
        name="system-prompt",
        description="Main system prompt for the assistant",
    )
    registry.register(template)

    v1 = registry.add_version(
        name="system-prompt",
        content=SYSTEM_PROMPT_V1.strip(),
        author="ai-team",
        message="Initial system prompt",
    )
    print(f"Version 1 registered: id={v1.version_id}")

    v2 = registry.add_version(
        name="system-prompt",
        content=SYSTEM_PROMPT_V2.strip(),
        author="ai-team",
        message="Added citation and formatting instructions",
    )
    print(f"Version 2 registered: id={v2.version_id}")

    # Step 3: Retrieve the latest version
    latest = registry.get_latest("system-prompt")
    print(f"\nLatest prompt version: {latest.version_id}")
    print(f"  Content preview: {latest.content[:60]}...")

    # Step 4: List all versions
    versions = registry.list_versions("system-prompt")
    print(f"\nAll versions ({len(versions)} total):")
    for version in versions:
        print(f"  [{version.version_id}] by {version.author}: '{version.message}'")

    # Step 5: Diff between versions
    try:
        diff = registry.diff("system-prompt", v1.version_id, v2.version_id)
        print(f"\nDiff v1 -> v2:")
        print(f"  Lines added: {diff.lines_added}")
        print(f"  Lines removed: {diff.lines_removed}")
        print(f"  Has changes: {diff.has_changes}")
    except Exception as error:
        print(f"Diff error: {error}")

    # Step 6: Retrieve by specific version
    retrieved_v1 = registry.get_version("system-prompt", v1.version_id)
    print(f"\nRetrieved v1 content length: {len(retrieved_v1.content)} chars")


if __name__ == "__main__":
    main()
