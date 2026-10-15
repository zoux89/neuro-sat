import argparse
import os
import random

import networkx as nx
import numpy as np
from g4satbench.utils.utils import VIG, clean_clauses, write_dimacs_to
from tqdm import tqdm

from .canonical import canonical_hash
from .mu1 import generate_mu1
from .mus import generate_mus
from .satutil import is_minimally_unsat, make_sat_twin, n_vars_of


def sample_obstruction(source, n_vars, rng, np_rng, verify):
    if source == "mu1":
        clauses = generate_mu1(n_vars, rng)
    else:
        clauses = generate_mus(n_vars, rng, np_rng)
    if verify and not is_minimally_unsat(clauses):
        return None
    return clauses


class Generator:
    def __init__(self, opts):
        self.opts = opts
        self.seen = set()

    def run(self):
        rng = random.Random(self.opts.seed)
        np_rng = np.random.default_rng(self.opts.seed)
        out_dir = os.path.abspath(self.opts.out_dir)
        for split in ["train", "valid", "test"]:
            n = getattr(self.opts, f"{split}_instances")
            if n <= 0:
                continue
            sat_dir = os.path.join(out_dir, split, "sat")
            unsat_dir = os.path.join(out_dir, split, "unsat")
            os.makedirs(sat_dir, exist_ok=True)
            os.makedirs(unsat_dir, exist_ok=True)
            print(f"Generating {self.opts.source} {split} set...")
            self.fill(n, sat_dir, unsat_dir, rng, np_rng)

    def next_source(self, rng):
        if self.opts.source == "mix":
            return rng.choice(["mu1", "mus"])
        return self.opts.source

    def fill(self, n, sat_dir, unsat_dir, rng, np_rng):
        made = 0
        attempts = 0
        max_attempts = n * 200 + 1000
        pbar = tqdm(total=n)
        while made < n and attempts < max_attempts:
            attempts += 1
            source = self.next_source(rng)
            n_vars = rng.randint(self.opts.min_n, self.opts.max_n)
            obstruction = sample_obstruction(source, n_vars, rng, np_rng, self.opts.verify)
            if obstruction is None:
                continue

            clauses = clean_clauses([list(c) for c in obstruction])
            nv = n_vars_of(clauses)
            if self.opts.require_connected and not nx.is_connected(VIG(nv, clauses)):
                continue

            key = canonical_hash(clauses)
            if key in self.seen:
                continue

            twin = make_sat_twin(clauses, rng)
            if twin is None:
                continue
            twin = clean_clauses(twin)

            self.seen.add(key)
            write_dimacs_to(nv, clauses, os.path.join(unsat_dir, "%.5d.cnf" % made))
            write_dimacs_to(n_vars_of(twin), twin, os.path.join(sat_dir, "%.5d.cnf" % made))
            made += 1
            pbar.update(1)
        pbar.close()
        if made < n:
            print(f"WARNING: only produced {made}/{n} pairs after {attempts} attempts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", type=str)
    p.add_argument("--source", choices=["mu1", "mus", "mix"], default="mu1")
    p.add_argument("--train_instances", type=int, default=0)
    p.add_argument("--valid_instances", type=int, default=0)
    p.add_argument("--test_instances", type=int, default=0)
    p.add_argument("--min_n", type=int, default=10)
    p.add_argument("--max_n", type=int, default=40)
    p.add_argument("--no_require_connected", dest="require_connected", action="store_false")
    p.add_argument("--no_verify", dest="verify", action="store_false",
                   help="skip the minimal unsatisfiability check")
    p.add_argument("--seed", type=int, default=0)
    Generator(p.parse_args()).run()


if __name__ == "__main__":
    main()
