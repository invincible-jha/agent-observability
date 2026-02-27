"""Simulated agent workload generator for observability benchmarks.

Produces reproducible sequences of synthetic agent actions (LLM calls,
tool invocations, memory reads, reasoning steps) without any external
dependencies or network calls.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class ActionKind(str, Enum):
    """Types of agent actions in a synthetic workload."""

    LLM_CALL = "llm_call"
    TOOL_INVOKE = "tool_invoke"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    REASONING_STEP = "reasoning_step"
    AGENT_DELEGATE = "agent_delegate"


@dataclass
class SyntheticAction:
    """A single unit of work in the simulated agent workload.

    Parameters
    ----------
    kind:
        The type of agent action.
    name:
        Human-readable label for this specific action instance.
    model:
        LLM model identifier (only relevant for LLM_CALL actions).
    input_tokens:
        Simulated input token count.
    output_tokens:
        Simulated output token count.
    cost_usd:
        Simulated USD cost for this action.
    tool_name:
        Tool name (only relevant for TOOL_INVOKE actions).
    success:
        Whether this action succeeds.
    """

    kind: ActionKind
    name: str
    model: str = "gpt-4o-mini"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_name: str = ""
    success: bool = True


# Deterministic model roster and tool roster
_MODELS = [
    ("gpt-4o-mini", "openai"),
    ("claude-haiku-4", "anthropic"),
    ("gemini-2.0-flash", "google"),
]

_TOOLS = [
    "web_search",
    "code_interpreter",
    "file_reader",
    "database_query",
    "send_email",
]


def generate_workload(
    n_actions: int = 1000,
    seed: int = 42,
    failure_rate: float = 0.05,
) -> list[SyntheticAction]:
    """Generate a reproducible sequence of synthetic agent actions.

    Parameters
    ----------
    n_actions:
        Total number of actions to generate.
    seed:
        Random seed for reproducibility.
    failure_rate:
        Fraction of actions that simulate a failure.

    Returns
    -------
    list of SyntheticAction
    """
    rng = random.Random(seed)
    actions: list[SyntheticAction] = []
    kind_weights = [0.35, 0.20, 0.15, 0.10, 0.15, 0.05]
    kind_choices = list(ActionKind)

    for index in range(n_actions):
        kind = rng.choices(kind_choices, weights=kind_weights, k=1)[0]
        model_name, _provider = rng.choice(_MODELS)
        tool_name = rng.choice(_TOOLS) if kind == ActionKind.TOOL_INVOKE else ""
        input_tokens = rng.randint(50, 4096) if kind == ActionKind.LLM_CALL else 0
        output_tokens = rng.randint(10, 512) if kind == ActionKind.LLM_CALL else 0
        cost_usd = round((input_tokens * 0.00000015) + (output_tokens * 0.0000006), 8)
        success = rng.random() > failure_rate
        actions.append(
            SyntheticAction(
                kind=kind,
                name=f"{kind.value}_{index:05d}",
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                tool_name=tool_name,
                success=success,
            )
        )
    return actions


def workload_batches(
    actions: list[SyntheticAction],
    batch_size: int,
) -> Iterator[list[SyntheticAction]]:
    """Yield successive batches from an action list.

    Parameters
    ----------
    actions:
        The full action sequence.
    batch_size:
        Size of each batch.
    """
    for offset in range(0, len(actions), batch_size):
        yield actions[offset : offset + batch_size]


__all__ = [
    "ActionKind",
    "SyntheticAction",
    "generate_workload",
    "workload_batches",
]
