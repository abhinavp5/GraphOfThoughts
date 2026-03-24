"""
Hard-case graph generators.

These graphs are designed to stress-test traversal algorithms:

- bridge_graph      — All paths between cliques pass through single bridge edges
- bottleneck_graph  — All paths route through one high-traffic choke-point node
- high_girth_graph  — Long cycles that require the model to maintain long-term state
"""

from __future__ import annotations

import random

import networkx as nx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_weights(graph: nx.Graph, weight_range: tuple[int, int], seed: int | None) -> nx.Graph:
    rng = random.Random(seed)
    lo, hi = weight_range
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(lo, hi)
    return graph


# ---------------------------------------------------------------------------
# Hard-case generators
# ---------------------------------------------------------------------------

def bridge_graph(
    num_cliques: int = 4,
    clique_size: int = 4,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> nx.Graph:
    """
    Chain of cliques connected by single bridge edges.

    Structure: K_clique_size — bridge — K_clique_size — bridge — ...

    Removing any bridge edge disconnects the graph, forcing the traversal
    to cross every bridge exactly once.
    """
    G = nx.Graph()
    offset = 0
    prev_connector = None

    for i in range(num_cliques):
        nodes = list(range(offset, offset + clique_size))
        # Build complete subgraph (clique)
        for u in nodes:
            for v in nodes:
                if u < v:
                    G.add_edge(u, v)
        # Connect to previous clique via a single bridge edge
        if prev_connector is not None:
            G.add_edge(prev_connector, nodes[0])
        prev_connector = nodes[-1]
        offset += clique_size

    if weighted:
        _add_weights(G, weight_range, seed)
    return G


def bottleneck_graph(
    num_arms: int = 6,
    arm_length: int = 4,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> nx.Graph:
    """
    Star-shaped graph where all arms connect through a single bottleneck node.

    Node 0 is the bottleneck. Each arm is a path of `arm_length` nodes
    attached at node 0.  Every inter-arm path must pass through node 0.
    """
    G = nx.Graph()
    bottleneck = 0
    G.add_node(bottleneck)
    offset = 1

    for _ in range(num_arms):
        arm_nodes = list(range(offset, offset + arm_length))
        # Connect arm root to bottleneck
        G.add_edge(bottleneck, arm_nodes[0])
        # Build the arm path
        for j in range(len(arm_nodes) - 1):
            G.add_edge(arm_nodes[j], arm_nodes[j + 1])
        offset += arm_length

    if weighted:
        _add_weights(G, weight_range, seed)
    return G


def high_girth_graph(
    n: int = 20,
    d: int = 3,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> nx.Graph:
    """
    High-girth regular graph: long cycles, no short shortcuts.

    Uses NetworkX's LCF notation or random regular graph construction.
    Falls back to a cycle graph extended with skip edges when the random
    regular graph construction fails (which can happen for small n).

    Parameters
    ----------
    n : number of nodes (must be even and n*d must be even)
    d : degree of every node (3 gives reasonable girth)
    """
    rng = random.Random(seed)

    # Adjust n to satisfy regularity constraints
    if (n * d) % 2 != 0:
        n += 1

    try:
        G = nx.random_regular_graph(d, n, seed=seed)
    except nx.NetworkXError:
        # Fallback: build a cycle + long-range chords to approximate high girth
        G = nx.cycle_graph(n)
        skip = max(2, n // 4)
        for i in range(0, n, skip):
            u = i
            v = (i + skip) % n
            if not G.has_edge(u, v):
                G.add_edge(u, v)

    if weighted:
        _add_weights(G, weight_range, seed)
    return G


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

HARD_CASES: dict[str, callable] = {
    "bridge": bridge_graph,
    "bottleneck": bottleneck_graph,
    "high_girth": high_girth_graph,
}


def get_hard_case_generator(name: str):
    if name not in HARD_CASES:
        raise KeyError(f"Unknown hard case '{name}'. Available: {list(HARD_CASES)}")
    return HARD_CASES[name]
