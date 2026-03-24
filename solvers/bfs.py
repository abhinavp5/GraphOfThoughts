"""
BFS solver with step-by-step state logging via StateExecutor.

Returns a list of step records:
    [{ "step": t, "operation": o_t, "state": s_t, "induced_subgraph": H_t }, ...]
"""

from __future__ import annotations

from collections import deque
from typing import Any

from solvers.state_executor import StateExecutor


def bfs(graph, source: Any) -> list[dict]:
    """
    Run BFS from `source` on `graph`, recording every operation.

    Operations emitted (in order):
      - enqueue(source)           — seed the queue
      - set_parent(source, None)  — source has no parent
      - dequeue(u)                — u is taken from the front
      - visit(u)                  — u is marked visited
      - enqueue(v)                — v is a newly discovered neighbour
      - set_parent(v, u)          — v was reached from u
    """
    executor = StateExecutor(graph, source)

    visited = set()
    queue: deque = deque()

    # Initialise
    queue.append(source)
    executor.apply(f"enqueue({source})")
    executor.apply(f"set_parent({source}, None)")

    while queue:
        u = queue.popleft()
        executor.apply(f"dequeue({u})")

        if u in visited:
            continue

        executor.apply(f"visit({u})")
        visited.add(u)

        for v in sorted(graph.neighbors(u)):  # sorted for determinism
            if v not in visited and v not in queue:
                queue.append(v)
                executor.apply(f"enqueue({v})")
                executor.apply(f"set_parent({v}, {u})")

    return executor.trace
