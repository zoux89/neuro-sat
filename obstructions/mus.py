import random
from typing import List, Optional, Sequence

import numpy as np
from pysat.solvers import Cadical195

from .satutil import is_sat, n_vars_of

Clause = Sequence[int]


def extract_mus(clauses: Sequence[Clause]) -> Optional[List[List[int]]]:
    """Deletion-based MUS extraction. Returns None if the formula is satisfiable."""
    clauses = [list(c) for c in clauses]
    if is_sat(clauses):
        return None

    first = n_vars_of(clauses) + 1
    selectors = list(range(first, first + len(clauses)))
    solver = Cadical195()
    for c, s in zip(clauses, selectors, strict=True):
        solver.add_clause(c + [s])

    keep = [True] * len(clauses)
    for i in range(len(clauses)):
        assumptions = [s if j == i or not keep[j] else -s for j, s in enumerate(selectors)]
        if not solver.solve(assumptions=assumptions):
            keep[i] = False
    solver.delete()
    return [c for c, k in zip(clauses, keep, strict=True) if k]


def random_sr_unsat(
    n_vars: int, rng: random.Random, np_rng: np.random.Generator,
    p_k_2: float = 0.3, p_geo: float = 0.4,
) -> List[List[int]]:
    """Add random SR clauses (NeuroSAT's generator) until the formula is unsatisfiable."""
    solver = Cadical195()
    clauses: List[List[int]] = []
    while True:
        k_base = 1 if rng.random() < p_k_2 else 2
        k = k_base + int(np_rng.geometric(p_geo))
        vs = np_rng.choice(n_vars, size=min(n_vars, k), replace=False)
        clause = [int(v + 1) if rng.random() < 0.5 else int(-(v + 1)) for v in vs]
        solver.add_clause(clause)
        clauses.append(clause)
        if not solver.solve():
            break
    solver.delete()
    return clauses


def generate_mus(
    n_vars: int, rng: random.Random, np_rng: np.random.Generator, **sr_kwargs
) -> List[List[int]]:
    mus = extract_mus(random_sr_unsat(n_vars, rng, np_rng, **sr_kwargs))
    assert mus is not None
    return mus


if __name__ == "__main__":
    f = random_sr_unsat(20, random.Random(0), np.random.default_rng(0))
    mus = extract_mus(f)
    print(f"SR formula: {len(f)} clauses, MUS: {len(mus)} clauses, "
          f"deficiency {len(mus) - n_vars_of(mus)}")
