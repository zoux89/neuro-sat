from hashlib import blake2b
from typing import Dict, List, Sequence

from pynauty import Graph, certificate

Clause = Sequence[int]


def _normalize(clauses: Sequence[Clause]):
    """Renumber variables to 1..n and drop empty and duplicate clauses."""
    var_map: Dict[int, int] = {}
    for c in clauses:
        for l in c:
            var_map.setdefault(abs(l), len(var_map) + 1)
    norm = []
    seen = set()
    for c in clauses:
        cl = frozenset((1 if l > 0 else -1) * var_map[abs(l)] for l in c)
        if cl and cl not in seen:
            seen.add(cl)
            norm.append(sorted(cl, key=abs))
    return len(var_map), norm


def lcg_certificate(clauses: Sequence[Clause]) -> bytes:
    n, norm = _normalize(clauses)
    n_nodes = 2 * n + len(norm)

    def node(lit):
        return 2 * (abs(lit) - 1) + (lit < 0)

    adj: Dict[int, List[int]] = {i: [] for i in range(n_nodes)}
    for v in range(1, n + 1):
        adj[node(v)].append(node(-v))
        adj[node(-v)].append(node(v))
    for j, c in enumerate(norm):
        cnode = 2 * n + j
        for l in c:
            adj[node(l)].append(cnode)
            adj[cnode].append(node(l))

    coloring = [c for c in (set(range(2 * n)), set(range(2 * n, n_nodes))) if c]
    g = Graph(n_nodes, directed=False, adjacency_dict=adj, vertex_coloring=coloring)
    return certificate(g)


def canonical_hash(clauses: Sequence[Clause]) -> str:
    h = blake2b(digest_size=16)
    h.update(lcg_certificate(clauses))
    return h.hexdigest()


if __name__ == "__main__":
    f1 = [[1, 2], [-1, 3], [-2, -3]]
    f2 = [[10, 20], [-10, 30], [-20, -30]]
    f3 = [[-3, -2], [4, -1], [1, 3]]
    print("renamed copy has the same hash:", canonical_hash(f1) == canonical_hash(f2))
    print("different formula has a different hash:", canonical_hash(f1) != canonical_hash(f3))
