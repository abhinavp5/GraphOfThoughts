"""
State Executor — deterministic, non-neural source of ground truth.

Maintains algorithmic state s_t, applies operations o_t, derives the
induced subgraph H_t, and produces JSON-serializable step records:

    { "step": t, "operation": o_t, "state": s_t, "induced_subgraph": H_t }
"""

from __future__ import annotations

import copy
from typing import Any

# Operations emitted by the BFS solver. Import this wherever vocabulary
# validation is needed (dataset loading, decoding, negative sampling).
# TODO: consider extending with check_visited and terminate if BFS is
#       modified to emit explicit state-query and halting operations (Option B).
VALID_OPS: frozenset[str] = frozenset({"enqueue", "dequeue", "set_parent", "mark_visited"})


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

def initial_state(source: Any) -> dict:
    """Return a blank algorithmic state seeded at `source`."""
    return {
        "visited": [],          # ordered list of visited nodes
        "frontier": [],         # queue (BFS) / stack (DFS) / heap entries (Dijkstra)
        "distances": {},        # node -> best-known distance (Dijkstra only)
        "parent": {},           # node -> parent node in the traversal tree
    }


# ---------------------------------------------------------------------------
# Induced subgraph H_t
# ---------------------------------------------------------------------------

def derive_induced_subgraph(state: dict, graph) -> list[tuple]:
    """
    Derive H_t — the predecessor tree encoded as a list of (u, v, weight) tuples.

    Edges are taken from the `parent` mapping: parent[v] = u means the
    traversal reached v via u.  Edge weight is looked up from `graph`
    (None for unweighted graphs).
    """
    edges = []
    for node, par in state["parent"].items():
        if par is None:
            continue
        weight = None
        if graph.has_edge(par, node):
            edge_data = graph[par][node]
            weight = edge_data.get("weight", None)
        edges.append((par, node, weight))
    return edges


# ---------------------------------------------------------------------------
# State executor
# ---------------------------------------------------------------------------

class StateExecutor:
    """
    Applies a sequence of operations to maintain ground-truth graph state.

    Usage::

        executor = StateExecutor(graph, source)
        record = executor.apply("enqueue(0)")   # step 0 — init
        record = executor.apply("mark_visited(1)")  # step 1
        trace  = executor.trace                  # all records so far
    """

    def __init__(self, graph, source: Any):
        self.graph = graph
        self.source = source
        self._state: dict = initial_state(source)
        self._step: int = 0
        self.trace: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, operation: str) -> dict:
        """Apply `operation`, record the resulting state, return the record."""
        self._execute(operation)
        record = self._snapshot(operation)
        self.trace.append(record)
        self._step += 1
        return record

    def current_state(self) -> dict:
        return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Operation dispatch
    # ------------------------------------------------------------------

    def _execute(self, operation: str) -> None:
        """Parse and apply a single operation string."""
        op, args = _parse_op(operation)

        if op not in VALID_OPS:
            raise ValueError(f"Unknown operation '{operation}'. VALID_OPS={set(VALID_OPS)}")

        if op == "enqueue":
            node = args[0]
            if node not in self._state["frontier"]:
                self._state["frontier"].append(node)

        elif op == "dequeue":
            node = args[0]
            if node in self._state["frontier"]:
                self._state["frontier"].remove(node)

        elif op == "mark_visited":
            node = args[0]
            if node not in self._state["visited"]:
                self._state["visited"].append(node)
            if node in self._state["frontier"]:
                self._state["frontier"].remove(node)

        elif op == "set_parent":
            child, parent = args[0], args[1]
            self._state["parent"][child] = parent

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _snapshot(self, operation: str) -> dict:
        state_copy = copy.deepcopy(self._state)
        state_copy["visited"] = list(state_copy["visited"])
        state_copy["frontier"] = list(state_copy["frontier"])
        state_copy["distances"] = {str(k): v for k, v in state_copy["distances"].items()}
        state_copy["parent"] = {str(k): (str(v) if v is not None else None)
                                 for k, v in state_copy["parent"].items()}
        return {
            "step": self._step,
            "operation": operation,
            "state": state_copy,
            "induced_subgraph": derive_induced_subgraph(self._state, self.graph),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_op(operation: str) -> tuple[str, list]:
    """
    Parse 'op_name(arg1, arg2, ...)' into (op_name, [arg1, arg2, ...]).
    Arguments are cast to int or float when possible.
    """
    operation = operation.strip()
    if "(" not in operation:
        return operation, []
    op = operation[:operation.index("(")]
    raw_args = operation[operation.index("(") + 1: operation.rindex(")")]
    if not raw_args.strip():
        return op, []
    parsed = []
    for a in raw_args.split(","):
        a = a.strip()
        if a == "None":
            parsed.append(None)
            continue
        try:
            parsed.append(int(a))
        except ValueError:
            try:
                parsed.append(float(a))
            except ValueError:
                parsed.append(a)
    return op, parsed
