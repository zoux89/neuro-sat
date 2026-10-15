import random
from typing import List, Optional, Sequence

from pysat.solvers import Cadical195

Clause = Sequence[int]


def n_vars_of(clauses: Sequence[Clause]) -> int:
    return max((abs(l) for c in clauses for l in c), default=0)


def solve(clauses: Sequence[Clause]) -> Optional[List[int]]:
    """Return a model as a list of signed variables, or None if the formula is unsatisfiable."""
    s = Cadical195(bootstrap_with=[list(c) for c in clauses])
    model = s.get_model() if s.solve() else None
    s.delete()
    return model


def is_sat(clauses: Sequence[Clause]) -> bool:
    return solve(clauses) is not None


def is_minimally_unsat(clauses: Sequence[Clause]) -> bool:
    clauses = [list(c) for c in clauses]
    if is_sat(clauses):
        return False
    return all(is_sat(clauses[:i] + clauses[i + 1:]) for i in range(len(clauses)))


def make_sat_twin(
    clauses: Sequence[Clause], rng: random.Random | None = None
) -> Optional[List[List[int]]]:
    """Return the formula with one literal flipped so that it is satisfiable.

    For an MU formula F and any clause c, F without c has a model that falsifies
    every literal of c, so flipping one of them satisfies c. Returns None if no
    such flip exists, which means F was not MU.
    """
    rng = rng or random.Random()
    clauses = [list(c) for c in clauses]
    order = list(range(len(clauses)))
    rng.shuffle(order)
    for i in order:
        c = clauses[i]
        if not c:
            continue
        model = solve(clauses[:i] + clauses[i + 1:])
        if model is None:
            continue
        true_lits = set(model)
        falsified = [l for l in c if -l in true_lits]
        if not falsified:
            continue
        l = rng.choice(falsified)
        twin = [list(cl) for cl in clauses]
        twin[i] = [-l if lit == l else lit for lit in c]
        if is_sat(twin):
            return twin
    return None
