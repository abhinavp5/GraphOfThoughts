"""
Shared helpers for benchmark integrations (NLGraph / GLBench).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON root type in {path!r}: {type(data)}")


def dump_json(path: str | Path, payload) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)


def normalize_benchmark_record(
    record: dict,
    *,
    graph_keys: list[str],
    algorithm_keys: list[str],
    source_keys: list[str],
    steps_keys: list[str],
) -> dict:
    """
    Convert one benchmark record into the GoT trace schema expected by run_inference:
      {graph, algorithm, source, steps}
    """

    def _pick(keys: list[str], default=None):
        for k in keys:
            if k in record and record[k] is not None:
                return record[k]
        return default

    graph = _pick(graph_keys)
    algorithm = _pick(algorithm_keys, "bfs")
    source = _pick(source_keys, 0)
    steps = _pick(steps_keys)

    # If benchmark provides operation strings only, synthesize minimal step objects.
    if steps and isinstance(steps, list) and steps and isinstance(steps[0], str):
        steps = [
            {
                "step": i,
                "operation": op,
                "state": {"visited": [], "frontier": [], "distances": {}, "parent": {}},
                "induced_subgraph": [],
            }
            for i, op in enumerate(steps)
        ]

    if not isinstance(steps, list):
        steps = []

    if graph is None:
        raise ValueError(
            "Record missing graph field. Expected one of keys: "
            + ", ".join(graph_keys)
        )

    return {
        "graph": graph,
        "algorithm": str(algorithm).lower(),
        "source": source,
        "steps": steps,
        "benchmark_meta": {
            "id": record.get("id"),
            "task": record.get("task") or record.get("task_type"),
        },
    }
