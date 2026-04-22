"""
Prompt Forcing — stepwise prompt construction for GoT inference.

At each inference step, the LLM sees:
  • system + user messages (same as training — format_prompt)
  • all previously applied steps formatted in the training template,
    where the states shown are *ground-truth* from StateExecutor (never
    the model's own outputs)
  • a trailing "Step <t>: " that primes the model to emit the next operation

This file builds that partial prefix. Generation + state-executor-in-the-loop
live in run_inference.py.

Public API
----------
build_partial_completion(trace) → str
    Render all applied steps + prime the next step header.

format_step_line(step) → str
    Render one step (3 lines) in the training template.

extract_operation(generated_text) → str
    Pull the operation off the first line of the model's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.teacher_forcing import (
    SYSTEM_PROMPT,
    format_prompt,
    format_trace_completion,
    _format_state,
    _format_subgraph,
)


def build_few_shot_messages(
    graph_str: str,
    algorithm: str,
    source,
    demos: list[dict],
) -> list[dict]:
    """
    Build a chat message list with N demo (user, assistant) pairs prepended
    to the target user message.

    Each demo dict must have the full trace schema:
        {graph, algorithm, source, steps: [...]}

    Returns a list suitable for `tokenizer.apply_chat_template(..., add_generation_prompt=True)`.
    The caller then appends `build_partial_completion(trace)` to prime the model.
    """
    # Target prompt gives us [system, user_target]
    base = format_prompt(graph_str, algorithm, source)
    system_msg = base[0]
    target_user = base[1]

    messages: list[dict] = [system_msg]
    for demo in demos:
        demo_base = format_prompt(demo["graph"], demo["algorithm"], demo["source"])
        demo_user = demo_base[1]
        demo_assistant = {
            "role": "assistant",
            "content": format_trace_completion(demo["steps"]),
        }
        messages.append(demo_user)
        messages.append(demo_assistant)

    messages.append(target_user)
    return messages


def format_step_line(step: dict) -> str:
    """Render one step as the 3-line block used during training."""
    return (
        f"Step {step['step']}: {step['operation']}\n"
        f"State: {_format_state(step['state'])}\n"
        f"Subgraph: {_format_subgraph(step['induced_subgraph'])}"
    )


def build_partial_completion(trace: list[dict]) -> str:
    """
    Render all recorded steps (via the training formatter) and append
    `Step <next>: ` to prime the model.

    Empty trace → just `Step 0: `.
    """
    rendered = format_trace_completion(trace)
    next_index = len(trace)
    if rendered:
        return f"{rendered}\n\nStep {next_index}: "
    return f"Step {next_index}: "


def extract_operation(generated_text: str) -> str:
    """
    Given text generated after a `Step <t>: ` prime, return just the
    operation string on the first line (stripped).
    """
    return generated_text.split("\n", 1)[0].strip()


__all__ = [
    "SYSTEM_PROMPT",
    "format_prompt",
    "format_step_line",
    "build_partial_completion",
    "build_few_shot_messages",
    "extract_operation",
]
