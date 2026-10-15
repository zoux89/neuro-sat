import argparse
import glob
import os
import random

from g4satbench.utils.utils import parse_cnf_file, write_dimacs_to

from .mus import extract_mus
from .satutil import is_sat


def compact(clauses):
    """Renumber variables to 1..n."""
    var_map, out = {}, []
    for c in clauses:
        nc = []
        for l in c:
            v = var_map.setdefault(abs(l), len(var_map) + 1)
            nc.append(v if l > 0 else -v)
        out.append(nc)
    return len(var_map), out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src_train_dir")
    p.add_argument("out_train_dir")
    p.add_argument("--drop_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()
    rng = random.Random(opts.seed)

    dirs = {k: os.path.join(opts.out_train_dir, k) for k in ("sat", "unsat")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    count = {"sat": 0, "unsat": 0}

    def emit(kind, clauses):
        n, cl = compact(clauses)
        if n == 0 or not cl:
            return
        write_dimacs_to(n, cl, os.path.join(dirs[kind], "%.5d.cnf" % count[kind]))
        count[kind] += 1

    sat_files = sorted(glob.glob(os.path.join(opts.src_train_dir, "sat", "*.cnf")))
    unsat_files = sorted(glob.glob(os.path.join(opts.src_train_dir, "unsat", "*.cnf")))
    print(f"source: {len(sat_files)} sat, {len(unsat_files)} unsat", flush=True)

    for f in sat_files:
        emit("sat", parse_cnf_file(f)[1])
    for f in unsat_files:
        emit("unsat", parse_cnf_file(f)[1])

    for i, f in enumerate(unsat_files):
        _, cl = parse_cnf_file(f)
        mus = extract_mus(cl)
        if not mus:
            continue
        emit("unsat", mus)

        j = rng.randrange(len(mus))
        sub = mus[:j] + mus[j + 1:]
        if sub and is_sat(sub):
            emit("sat", sub)

        core = {frozenset(c) for c in mus}
        noncore = [c for c in cl if frozenset(c) not in core]
        if noncore:
            k = max(1, int(opts.drop_frac * len(noncore)))
            drop = set(rng.sample(range(len(noncore)), k))
            kept = [c for c in cl if frozenset(c) in core] + \
                   [c for t, c in enumerate(noncore) if t not in drop]
            if not is_sat(kept):
                emit("unsat", kept)

        if (i + 1) % 500 == 0:
            print(f"  unsat processed {i + 1}/{len(unsat_files)}", flush=True)

    for f in sat_files:
        _, cl = parse_cnf_file(f)
        k = max(1, int(opts.drop_frac * len(cl)))
        drop = set(rng.sample(range(len(cl)), k))
        kept = [c for t, c in enumerate(cl) if t not in drop]
        if kept and is_sat(kept):
            emit("sat", kept)

    m = min(count.values())
    for kind, d in dirs.items():
        for idx in range(m, count[kind]):
            os.remove(os.path.join(d, "%.5d.cnf" % idx))
    print(f"augmented train set: {m} sat / {m} unsat (before balancing: {count})", flush=True)


if __name__ == "__main__":
    main()
