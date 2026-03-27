"""
Negative Sampling — inject errors into traces and teach CORRECTION recovery.

Corrupts a fraction of training traces by inserting a wrong operation,
then requiring the model to emit a CORRECTION token before continuing
with the correct operation.  This teaches the model to recover from
mistakes at inference time.

Public API
----------
corrupt_trace(trace_dict, rng=None) → dict
    Return a *new* trace dict with one corruption + CORRECTION step injected.

maybe_corrupt(trace_dict, ratio=0.2, rng=None) → dict
    With probability `ratio`, corrupt the trace; otherwise return it unchanged.

CORRECTION_TOKEN : str
    The special token string added to the tokenizer.
"""

from __future__ import annotations

import copy
import random
from typing import Any

CORRECTION_TOKEN = "<|CORRECTION|>"


# ---------------------------------------------------------------------------
# Corruption strategies
# ---------------------------------------------------------------------------

def _wrong_node_operation(step: dict, all_nodes: list, rng: random.Random) -> str:
    """
    Replace the node argument(s) in an operation with a wrong but valid node.
    """
    op = step["operation"]

    # Parse operation name and args
    if "(" not in op:
        return op

    op_name = op[:op.index("(")]
    raw_args = op[op.index("(") + 1: op.rindex(")")]
    args = [a.strip() for a in raw_args.split(",")]

    if not args or not all_nodes:
        return op

    # Pick a node different from the first argument
    original_node = args[0]
    candidates = [str(n) for n in all_nodes if str(n) != original_node]
    if not candidates:
        return op

    args[0] = rng.choice(candidates)
    return f"{op_name}({', '.join(args)})"


def _invalid_node_operation(step: dict, max_node: int, rng: random.Random) -> str:
    """
    Replace the node argument with an out-of-range node ID.
    """
    op = step["operation"]

    if "(" not in op:
        return op

    op_name = op[:op.index("(")]
    raw_args = op[op.index("(") + 1: op.rindex(")")]
    args = [a.strip() for a in raw_args.split(",")]

    if not args:
        return op

    # Use a node ID guaranteed to be outside the graph
    fake_node = max_node + rng.randint(10, 50)
    args[0] = str(fake_node)
    return f"{op_name}({', '.join(args)})"


# ---------------------------------------------------------------------------
# Main corruption logic
# ---------------------------------------------------------------------------

def corrupt_trace(
    trace_dict: dict,
    rng: random.Random | None = None,
) -> dict:
    """
    Return a new trace dict with one corruption + CORRECTION injected.

    The corruption is inserted at a randomly chosen step (not the first
    or last).  The corrupted step is followed by a CORRECTION step that
    contains the original correct operation.

    The ground-truth state is NOT advanced by the corrupted operation —
    the state shown after the error remains the pre-error ground truth.

    Parameters
    ----------
    trace_dict : dict
        Original trace with keys: graph, algorithm, source, steps.

    Returns
    -------
    dict
        New trace dict with the corruption inserted.
    """
    if rng is None:
        rng = random.Random()

    trace = copy.deepcopy(trace_dict)
    steps = trace["steps"]

    if len(steps) < 4:
        # Too short to corrupt meaningfully
        return trace

    # Choose corruption point: avoid step 0 (init) and last step
    corrupt_idx = rng.randint(2, len(steps) - 2)
    target_step = steps[corrupt_idx]

    # Extract all node IDs from the graph string for corruption
    graph_str = trace.get("graph", "")
    all_nodes = _extract_nodes(graph_str)
    max_node = max(all_nodes) if all_nodes else 20

    # Pick corruption strategy
    strategy = rng.choice(["wrong_node", "invalid_node"])

    if strategy == "wrong_node":
        corrupted_op = _wrong_node_operation(target_step, all_nodes, rng)
    else:
        corrupted_op = _invalid_node_operation(target_step, max_node, rng)

    # If corruption didn't actually change anything, force invalid node
    if corrupted_op == target_step["operation"]:
        corrupted_op = _invalid_node_operation(target_step, max_node, rng)

    # Build the corrupted step (uses pre-error state)
    pre_error_state = steps[corrupt_idx - 1]["state"]
    pre_error_subgraph = steps[corrupt_idx - 1]["induced_subgraph"]

    corrupted_step = {
        "step": target_step["step"],
        "operation": corrupted_op,
        "state": copy.deepcopy(pre_error_state),
        "induced_subgraph": copy.deepcopy(pre_error_subgraph),
        "is_error": True,
    }

    # Build the CORRECTION step (restores the correct operation)
    correction_step = {
        "step": target_step["step"],
        "operation": CORRECTION_TOKEN,
        "state": copy.deepcopy(pre_error_state),
        "induced_subgraph": copy.deepcopy(pre_error_subgraph),
        "is_correction": True,
    }

    # Insert corrupted step + correction before the real step
    new_steps = (
        steps[:corrupt_idx]
        + [corrupted_step, correction_step]
        + steps[corrupt_idx:]  # original correct step follows
    )

    # Re-number steps sequentially
    for i, step in enumerate(new_steps):
        step["step"] = i

    trace["steps"] = new_steps
    trace["corrupted"] = True
    trace["corruption_index"] = corrupt_idx
    return trace


def maybe_corrupt(
    trace_dict: dict,
    ratio: float = 0.2,
    rng: random.Random | None = None,
) -> dict:
    """
    With probability `ratio`, corrupt the trace; otherwise return unchanged.
    """
    if rng is None:
        rng = random.Random()

    if rng.random() < ratio:
        return corrupt_trace(trace_dict, rng=rng)
    return trace_dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_nodes(graph_str: str) -> list[int]:
    """Extract unique integer node IDs from a linearized graph string."""
    import re
    nodes = set()
    for match in re.finditer(r"\((\d+)\)", graph_str):
        nodes.add(int(match.group(1)))
    return sorted(nodes)
