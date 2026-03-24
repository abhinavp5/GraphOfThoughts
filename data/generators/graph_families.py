"""
Standard graph family generators.

Each function returns a NetworkX graph (undirected by default) with integer
node labels and optional random edge weights stored as the "weight" attribute.

Families
--------
- erdos_renyi(n, p, ...)         Erdős–Rényi G(n,p)
- barabasi_albert(n, m, ...)     Barabási–Albert preferential attachment
- random_tree(n, ...)            Uniformly random labelled tree (Prüfer)
- grid(rows, cols, ...)          2-D grid, nodes re-labeled as integers
"""

from __future__ import annotations

import random

import networkx as nx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_weights(graph: nx.Graph, weight_range: tuple[int, int], seed: int | None) -> nx.Graph:
    """Attach random integer weights to every edge in-place, then return the graph."""
    rng = random.Random(seed)
    lo, hi = weight_range
    for u, v in graph.edges():
        graph[u][v]["weight"] = rng.randint(lo, hi)
    return graph


def _relabel_to_int(graph: nx.Graph) -> nx.Graph:
    """Re-label nodes to consecutive integers (useful for grid graphs)."""
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    return nx.relabel_nodes(graph, mapping)


# ---------------------------------------------------------------------------
# Graph family generators
# ---------------------------------------------------------------------------

def erdos_renyi(
    n: int,
    p: float = 0.3,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
    directed: bool = False,
) -> nx.Graph:
    """
    Erdős–Rényi random graph G(n, p).

    Each edge is included independently with probability p.
    Ensures the graph is connected by adding a spanning path if needed.
    """
    rng = seed
    G = nx.erdos_renyi_graph(n, p, seed=rng, directed=directed)
    # Guarantee connectivity for undirected graphs
    if not directed and not nx.is_connected(G):
        components = list(nx.connected_components(G))
        r = random.Random(seed)
        for i in range(len(components) - 1):
            u = r.choice(list(components[i]))
            v = r.choice(list(components[i + 1]))
            G.add_edge(u, v)
    if weighted:
        _add_weights(G, weight_range, seed)
    return G


def barabasi_albert(
    n: int,
    m: int = 2,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> nx.Graph:
    """
    Barabási–Albert scale-free graph.

    Each new node attaches to m existing nodes via preferential attachment.
    Always connected.
    """
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    if weighted:
        _add_weights(G, weight_range, seed)
    return G


def random_tree(
    n: int,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> nx.Graph:
    """
    Uniformly random labelled tree on n nodes (via Prüfer sequence).

    Always a tree: connected, acyclic, n-1 edges.
    """
    G = nx.random_labeled_tree(n, seed=seed)
    if weighted:
        _add_weights(G, weight_range, seed)
    return G


def grid(
    rows: int,
    cols: int,
    weighted: bool = False,
    weight_range: tuple[int, int] = (1, 10),
    seed: int | None = None,
) -> nx.Graph:
    """
    2-D grid graph with rows × cols nodes.

    Nodes are re-labeled as consecutive integers (row-major order).
    """
    G = nx.grid_2d_graph(rows, cols)
    G = _relabel_to_int(G)
    if weighted:
        _add_weights(G, weight_range, seed)
    return G


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FAMILIES: dict[str, callable] = {
    "erdos_renyi": erdos_renyi,
    "barabasi_albert": barabasi_albert,
    "random_tree": random_tree,
    "grid": grid,
}


def get_generator(name: str):
    """Return a generator function by name, raising KeyError if unknown."""
    if name not in FAMILIES:
        raise KeyError(f"Unknown graph family '{name}'. Available: {list(FAMILIES)}")
    return FAMILIES[name]
