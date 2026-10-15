import random
from typing import List


def generate_mu1(n_vars: int, rng: random.Random | None = None) -> List[List[int]]:
    """Return a random minimally unsatisfiable formula with n_vars variables and n_vars + 1 clauses."""
    if n_vars < 1:
        raise ValueError("n_vars must be >= 1")
    rng = rng or random.Random()

    clauses = [set()]
    for v in range(1, n_vars + 1):
        base = clauses.pop(rng.randrange(len(clauses)))
        clauses.append(base | {v})
        clauses.append(base | {-v})

    formula = [sorted(c, key=abs) for c in clauses]
    rng.shuffle(formula)
    return formula


if __name__ == "__main__":
    f = generate_mu1(6, random.Random(0))
    print(f"n=6, m={len(f)}")
    for c in f:
        print(c)
