"""
Graph-Language Linearization.

Encodes a NetworkX graph (or subgraph) as a compact token string:

    Weighted edge:    (u)-[w]->(v)
    Unweighted edge:  (u)->(v)

A full graph is space-separated edge tokens, one per edge.

Public API
----------
graph_to_str(G)                          → str
subgraph_to_str(edges)                   → str   (edges = list of (u,v,w) tuples)
edge_to_str(u, v, weight=None)           → str
"""

from __future__ import annotations

import networkx as nx


def edge_to_str(u, v, weight=None) -> str:
    """Encode a single edge as a token string."""
    if weight is not None:
        return f"({u})-[{weight}]->({v})"
    return f"({u})->({v})"


def graph_to_str(G: nx.Graph) -> str:
    """
    Linearize a NetworkX graph to a space-separated token string.

    Edge ordering is sorted by (u, v) for determinism.
    For directed graphs every arc is encoded once; for undirected graphs
    each edge is encoded once (u < v canonical ordering).
    """
    tokens = []
    if G.is_directed():
        edges = sorted(G.edges(data=True), key=lambda e: (e[0], e[1]))
        for u, v, data in edges:
            tokens.append(edge_to_str(u, v, data.get("weight")))
    else:
        edges = sorted(G.edges(data=True), key=lambda e: (min(e[0], e[1]), max(e[0], e[1])))
        for u, v, data in edges:
            tokens.append(edge_to_str(min(u, v), max(u, v), data.get("weight")))
    return " ".join(tokens)


def subgraph_to_str(edges: list[tuple]) -> str:
    """
    Linearize an induced subgraph H_t from a list of (u, v, weight) tuples.

    Weight may be None for unweighted edges.
    Sorted by (u, v) for determinism.
    """
    sorted_edges = sorted(edges, key=lambda e: (e[0], e[1]))
    tokens = [edge_to_str(u, v, w) for u, v, w in sorted_edges]
    return " ".join(tokens)
