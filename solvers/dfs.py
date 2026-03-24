"""
DFS solver with step-by-step state logging via StateExecutor.

Returns a list of step records:
    [{ "step": t, "operation": o_t, "state": s_t, "induced_subgraph": H_t }, ...]
"""

from __future__ import annotations

from typing import Any

from solvers.state_executor import StateExecutor


def dfs(graph, source: Any) -> list[dict]:
    """
    Run iterative DFS from `source` on `graph`, recording every operation.

    Operations emitted (in order):
      - push(source)              — seed the stack
      - set_parent(source, None)  — source has no parent
      - pop(u)                    — u is taken from the top
      - visit(u)                  — u is marked visited
      - push(v)                   — v is a newly discovered neighbour
      - set_parent(v, u)          — v was reached from u
    """
    executor = StateExecutor(graph, source)

    visited: set = set()
    stack: list = []
    # Track which parent pushed each node (last write wins for stack duplicates)
    pushed_by: dict = {}

    stack.append(source)
    pushed_by[source] = None
    executor.apply(f"push({source})")
    executor.apply(f"set_parent({source}, None)")

    while stack:
        u = stack.pop()
        executor.apply(f"pop({u})")

        if u in visited:
            continue

        executor.apply(f"visit({u})")
        visited.add(u)

        # Push neighbours in reverse-sorted order so smallest is processed first
        for v in sorted(graph.neighbors(u), reverse=True):
            if v not in visited:
                stack.append(v)
                pushed_by[v] = u
                executor.apply(f"push({v})")
                executor.apply(f"set_parent({v}, {u})")

    return executor.trace
