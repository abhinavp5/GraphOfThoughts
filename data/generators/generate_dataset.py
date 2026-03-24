"""
Dataset generation CLI.

Generates a JSON trace dataset for a chosen algorithm and graph family.

Usage
-----
python -m data.generators.generate_dataset \\
    --algorithm bfs \\
    --family erdos_renyi \\
    --n 20 \\
    --count 1000 \\
    --out data/traces/bfs_er.json

Each entry in the output JSON array:
    {
        "graph":     "<linearized graph string>",
        "algorithm": "bfs",
        "source":    <source node id>,
        "steps":     [ { "step": t, "operation": o_t, "state": s_t,
                          "induced_subgraph": [[u, v, w], ...] }, ... ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Allow running as `python data/generators/generate_dataset.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generators.graph_families import FAMILIES, get_generator
from data.generators.hard_cases import HARD_CASES, get_hard_case_generator
from data.linearization import graph_to_str
from solvers.bfs import bfs
from solvers.dfs import dfs
from solvers.dijkstra import dijkstra


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGORITHMS = {
    "bfs": bfs,
    "dfs": dfs,
    "dijkstra": dijkstra,
}

ALL_FAMILIES = list(FAMILIES) + list(HARD_CASES)


# ---------------------------------------------------------------------------
# Single sample generation
# ---------------------------------------------------------------------------

def generate_sample(
    algorithm: str,
    family: str,
    n: int,
    weighted: bool,
    seed: int | None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
) -> dict:
    """Generate one (graph, trace) sample."""
    solver = ALGORITHMS[algorithm]

    # Build graph
    if family in FAMILIES:
        gen = get_generator(family)
        if family == "grid":
            rows = grid_rows or max(2, int(n ** 0.5))
            cols = grid_cols or max(2, (n + rows - 1) // rows)
            G = gen(rows, cols, weighted=weighted, seed=seed)
        elif family == "barabasi_albert":
            G = gen(n, m=2, weighted=weighted, seed=seed)
        else:
            G = gen(n, weighted=weighted, seed=seed)
    else:
        gen = get_hard_case_generator(family)
        G = gen(weighted=weighted, seed=seed)

    # Pick random source node
    rng = random.Random(seed)
    source = rng.choice(list(G.nodes()))

    # Run solver
    trace = solver(G, source)

    # Serialise induced_subgraph tuples as lists for JSON
    for step in trace:
        step["induced_subgraph"] = [list(e) for e in step["induced_subgraph"]]

    return {
        "graph": graph_to_str(G),
        "algorithm": algorithm,
        "source": source,
        "steps": trace,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate JSON trace datasets for graph algorithm training."
    )
    parser.add_argument(
        "--algorithm", required=True, choices=list(ALGORITHMS),
        help="Graph algorithm to trace."
    )
    parser.add_argument(
        "--family", required=True, choices=ALL_FAMILIES,
        help="Graph family to sample from."
    )
    parser.add_argument(
        "--n", type=int, default=20,
        help="Number of nodes per graph (default: 20)."
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of samples to generate (default: 100)."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSON file path."
    )
    parser.add_argument(
        "--weighted", action="store_true",
        help="Attach random integer edge weights."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed (each sample gets seed+i)."
    )
    parser.add_argument(
        "--grid-rows", type=int, default=None,
        help="Grid rows (only used when --family grid)."
    )
    parser.add_argument(
        "--grid-cols", type=int, default=None,
        help="Grid cols (only used when --family grid)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = []
    for i in range(args.count):
        sample_seed = (args.seed + i) if args.seed is not None else None
        sample = generate_sample(
            algorithm=args.algorithm,
            family=args.family,
            n=args.n,
            weighted=args.weighted,
            seed=sample_seed,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
        )
        dataset.append(sample)
        if (i + 1) % max(1, args.count // 10) == 0:
            print(f"  {i + 1}/{args.count} samples generated", flush=True)

    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nSaved {len(dataset)} traces to {out_path}")


if __name__ == "__main__":
    main()
