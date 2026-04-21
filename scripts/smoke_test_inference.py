"""
Smoke test for the inference loop — no ML dependencies required.

Validates everything except the LLM call: graph reconstruction,
prompt-building, state-executor-in-the-loop, and output format.
The "model" is a stub that replays the gold operation at each step.

If gold == predicted after running this script, the scaffolding works
and the pipeline is ready for a real LLM on a proper env.

Usage::

    python scripts/smoke_test_inference.py data/traces/test_bfs.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference.prompt_forcing import (
    build_partial_completion,
    format_prompt,
)
from inference.run_inference import reconstruct_graph
from solvers.state_executor import StateExecutor


def _canonicalize_state(state: dict) -> dict:
    """Normalize state dict for equality comparison (keys stringified)."""
    return {
        "visited": list(state.get("visited", [])),
        "frontier": list(state.get("frontier", [])),
        "distances": {str(k): v for k, v in state.get("distances", {}).items()},
        "parent": {
            str(k): (str(v) if v is not None else None)
            for k, v in state.get("parent", {}).items()
        },
    }


def _canonicalize_subgraph(subgraph: list) -> set:
    """Canonicalize induced subgraph edges as a set of tuples."""
    out = set()
    for edge in subgraph:
        u, v, w = edge[0], edge[1], edge[2] if len(edge) > 2 else None
        out.add((int(u), int(v), w))
    return out


def run_sample(sample: dict, verbose: bool = False) -> tuple[bool, list[str]]:
    """Run the inference loop with gold ops as the 'model' output."""
    graph_str = sample["graph"]
    algorithm = sample["algorithm"]
    source = sample["source"]
    gold_steps = sample["steps"]

    G = reconstruct_graph(graph_str)
    executor = StateExecutor(G, source)
    messages = format_prompt(graph_str, algorithm, source)

    errors = []
    for t, gold in enumerate(gold_steps):
        # Build the prompt prefix the LLM would see
        partial = build_partial_completion(executor.trace)

        # "Model" returns the gold op
        model_op = gold["operation"]

        # Apply to the state executor
        record = executor.apply(model_op)

        # Validate vs gold
        if record["operation"] != gold["operation"]:
            errors.append(f"step {t}: op mismatch {record['operation']!r} vs {gold['operation']!r}")

        got_state = _canonicalize_state(record["state"])
        want_state = _canonicalize_state(gold["state"])
        if got_state != want_state:
            errors.append(f"step {t}: state mismatch\n  got:  {got_state}\n  want: {want_state}")

        got_sub = _canonicalize_subgraph(record["induced_subgraph"])
        want_sub = _canonicalize_subgraph(gold["induced_subgraph"])
        if got_sub != want_sub:
            errors.append(f"step {t}: subgraph mismatch\n  got:  {got_sub}\n  want: {want_sub}")

        if verbose and not errors:
            print(f"  [{t}] {model_op}  (partial_len={len(partial)})")

    # Sanity check: partial completion at the end should render all gold steps
    final_partial = build_partial_completion(executor.trace)
    if f"Step {len(gold_steps)}: " not in final_partial:
        errors.append("final partial completion missing trailing step prime")

    return len(errors) == 0, errors


def main() -> int:
    trace_path = Path(sys.argv[1] if len(sys.argv) > 1
                      else ROOT / "data/traces/test_bfs.json")
    print(f"Loading {trace_path}")
    with trace_path.open() as f:
        dataset = json.load(f)

    n_samples = min(3, len(dataset))
    print(f"Running {n_samples} samples in mock-model mode")

    all_ok = True
    for i in range(n_samples):
        sample = dataset[i]
        ok, errors = run_sample(sample, verbose=(i == 0))
        status = "PASS" if ok else "FAIL"
        print(f"  sample {i}: {status}  ({sample['algorithm']}, "
              f"source={sample['source']}, steps={len(sample['steps'])})")
        if not ok:
            all_ok = False
            for err in errors[:5]:
                print(f"    - {err}")

    if all_ok:
        print("\nAll samples replay cleanly. Inference scaffolding is ready for a real LLM.")
        return 0
    print("\nFAIL — fix before running a real model.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
