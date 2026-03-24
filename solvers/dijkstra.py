"""
Dijkstra's algorithm with step-by-step state logging via StateExecutor.

Requires a weighted graph (edge attribute "weight"). Falls back to weight=1
for unweighted graphs.

Returns a list of step records:
    [{ "step": t, "operation": o_t, "state": s_t, "induced_subgraph": H_t }, ...]
"""

from __future__ import annotations

import heapq
from typing import Any

from solvers.state_executor import StateExecutor


def dijkstra(graph, source: Any) -> list[dict]:
    """
    Run Dijkstra's from `source` on `graph`, recording every operation.

    Operations emitted (in order):
      - init_source(source)       — distance 0, enqueue source
      - settle(u)                 — u extracted from heap, distance finalised
      - relax(u, v, new_dist)     — edge relaxation updates distance to v
    """
    executor = StateExecutor(graph, source)

    dist: dict = {source: 0}
    settled: set = set()
    heap: list = [(0, source)]   # (distance, node)

    executor.apply(f"init_source({source})")

    while heap:
        d, u = heapq.heappop(heap)

        if u in settled:
            continue

        executor.apply(f"settle({u})")
        settled.add(u)

        for v in graph.neighbors(u):
            weight = graph[u][v].get("weight", 1)
            new_dist = d + weight
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))
                executor.apply(f"relax({u}, {v}, {new_dist})")

    return executor.trace
