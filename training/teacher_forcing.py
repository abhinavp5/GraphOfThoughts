"""
Teacher Forcing — trace formatting for SFT training.

Converts raw JSON trace dicts into text suitable for causal LM training.
Every step includes the *ground-truth* state from the State Executor,
ensuring the model never conditions on its own (potentially wrong)
predictions during training.

Public API
----------
format_prompt(graph_str, algorithm, source) → list[dict]
    Build system + user chat messages.

format_trace_completion(trace_steps) → str
    Format the full step-by-step trace as the assistant's response text.

format_full_chat(trace_dict) → list[dict]
    Convenience: build the complete [system, user, assistant] message list.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.linearization import subgraph_to_str


# ---------------------------------------------------------------------------
# System prompts (per algorithm)
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "bfs": (
        "You are a BFS algorithm executor. Given a graph and a source node, "
        "produce the step-by-step BFS execution trace.\n"
        "At each step, output exactly one operation from the following set:\n"
        "  enqueue(node)            — add node to the back of the queue\n"
        "  dequeue(node)            — remove node from the front of the queue\n"
        "  mark_visited(node)       — record node as visited\n"
        "  set_parent(child, parent)— record that child was reached from parent\n"
        "No other operation names are permitted.\n"
        # TODO: extend with check_visited(node) and terminate() if Option B is adopted.
        "Format each step as:\n"
        "Step <t>: <operation>\n"
        "State: visited=<list> frontier=<list> distances={} parent=<dict>\n"
        "Subgraph: <linearized edges or (empty)>\n"
    ),
    "dfs": (
        "You are a DFS algorithm executor. Given a graph and a source node, "
        "produce the step-by-step DFS execution trace.\n"
        "At each step, output exactly one operation from the following set:\n"
        "  push(node)               — push node onto the stack\n"
        "  pop(node)                — pop node from the stack\n"
        "  visit(node)              — record node as visited\n"
        "  set_parent(child, parent)— record that child was reached from parent\n"
        "No other operation names are permitted.\n"
        "Format each step as:\n"
        "Step <t>: <operation>\n"
        "State: visited=<list> frontier=<list> distances={} parent=<dict>\n"
        "Subgraph: <linearized edges or (empty)>\n"
    ),
    "dijkstra": (
        "You are a Dijkstra algorithm executor. Given a weighted graph and a source node, "
        "produce the step-by-step Dijkstra execution trace.\n"
        "At each step, output exactly one operation from the following set:\n"
        "  init_source(node)        — set distance(node)=0 and enqueue it\n"
        "  settle(node)             — mark node's distance as final\n"
        "  relax(u, v, new_dist)    — update distance(v) via u and record predecessor\n"
        "No other operation names are permitted.\n"
        "Format each step as:\n"
        "Step <t>: <operation>\n"
        "State: visited=<list> frontier=<list> distances=<dict> parent=<dict>\n"
        "Subgraph: <linearized edges or (empty)>\n"
    ),
}


def _system_prompt_for(algorithm: str) -> str:
    key = (algorithm or "").strip().lower()
    if key not in _PROMPTS:
        raise ValueError(f"Unknown algorithm {algorithm!r}. Expected one of {sorted(_PROMPTS)}")
    return _PROMPTS[key]


# ---------------------------------------------------------------------------
# State formatting helpers
# ---------------------------------------------------------------------------

def _format_state(state: dict) -> str:
    """Format a state dict as a compact single-line string."""
    visited = state.get("visited", [])
    frontier = state.get("frontier", [])
    distances = state.get("distances", {})
    parent = state.get("parent", {})

    parts = [
        f"visited={visited}",
        f"frontier={frontier}",
    ]

    # Only include distances/parent when they contain data
    if distances:
        parts.append(f"distances={{{_fmt_dict(distances)}}}")
    else:
        parts.append("distances={}")

    if parent:
        parts.append(f"parent={{{_fmt_dict(parent)}}}")
    else:
        parts.append("parent={}")

    return " ".join(parts)


def _fmt_dict(d: dict) -> str:
    """Format a dict as 'k1: v1, k2: v2, ...'."""
    items = []
    for k, v in d.items():
        items.append(f"{k}: {v}")
    return ", ".join(items)


def _format_subgraph(induced_subgraph: list) -> str:
    """Format induced subgraph H_t using the standard linearization."""
    if not induced_subgraph:
        return "(empty)"
    # induced_subgraph is a list of [u, v, weight] or (u, v, weight)
    edges = [tuple(e) for e in induced_subgraph]
    return subgraph_to_str(edges)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def format_prompt(graph_str: str, algorithm: str, source) -> list[dict]:
    """
    Build system + user chat messages for a trace.

    Returns a list of dicts with 'role' and 'content' keys,
    compatible with HuggingFace tokenizer.apply_chat_template().
    """
    user_content = (
        f"Graph: {graph_str}\n"
        f"Algorithm: {algorithm.upper()}\n"
        f"Source: {source}\n\n"
        f"Execute the algorithm step by step."
    )
    return [
        {"role": "system", "content": _system_prompt_for(algorithm)},
        {"role": "user", "content": user_content},
    ]


def format_trace_completion(steps: list[dict]) -> str:
    """
    Format a full list of step records as the assistant's completion text.

    Each step record must have keys: step, operation, state, induced_subgraph.
    The state shown at each step is the *ground-truth* state from the
    State Executor (teacher forcing).
    """
    lines = []
    for step in steps:
        lines.append(f"Step {step['step']}: {step['operation']}")
        lines.append(f"State: {_format_state(step['state'])}")
        lines.append(f"Subgraph: {_format_subgraph(step['induced_subgraph'])}")
        lines.append("")  # blank line between steps

    return "\n".join(lines).rstrip()


def format_full_chat(trace_dict: dict) -> list[dict]:
    """
    Build the complete [system, user, assistant] message list for one trace.

    Parameters
    ----------
    trace_dict : dict
        A single trace from the JSON dataset with keys:
        graph, algorithm, source, steps.

    Returns
    -------
    list[dict]
        Chat messages ready for tokenizer.apply_chat_template().
    """
    messages = format_prompt(
        graph_str=trace_dict["graph"],
        algorithm=trace_dict["algorithm"],
        source=trace_dict["source"],
    )
    completion = format_trace_completion(trace_dict["steps"])
    messages.append({"role": "assistant", "content": completion})
    return messages
