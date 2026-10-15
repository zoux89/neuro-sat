import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from g4satbench.models.gnn import GNN
from g4satbench.utils.utils import parse_cnf_file
from pysat.solvers import Cadical195
from torch_geometric.data import Batch

from run_comparison import REPO, run_dir

sys.path.insert(0, REPO)
from obstructions.mus import extract_mus  # noqa: E402

RUN = run_dir("sr", 26, 1e-4, 1)


def flip_sensitivity(clauses, target_sat):
    """Fraction of (clause, literal) flips after which the formula's satisfiability equals target_sat."""
    nxt = max(abs(l) for c in clauses for l in c) + 1
    s = Cadical195()
    sel, flips = [], []
    for j, c in enumerate(clauses):
        sel.append(nxt)
        s.add_clause(list(c) + [nxt])
        nxt += 1
        for l in c:
            flips.append((j, nxt))
            s.add_clause([(-x if x == l else x) for x in c] + [nxt])
            nxt += 1
    base = [-x for x in sel] + [t for _, t in flips]
    hits = 0
    for k, (j, t) in enumerate(flips):
        assumptions = list(base)
        assumptions[j] = sel[j]
        assumptions[len(sel) + k] = -t
        if s.solve(assumptions=assumptions) == target_sat:
            hits += 1
    s.delete()
    return hits / len(flips)


def deletion_sensitivity(clauses):
    """Fraction of clauses of an unsatisfiable formula whose deletion makes it satisfiable."""
    base = max(abs(l) for c in clauses for l in c)
    sel = [base + 1 + j for j in range(len(clauses))]
    s = Cadical195()
    for c, t in zip(clauses, sel, strict=True):
        s.add_clause(list(c) + [t])
    hits = 0
    for j in range(len(clauses)):
        assumptions = [-t for t in sel]
        assumptions[j] = sel[j]
        if s.solve(assumptions=assumptions):
            hits += 1
    s.delete()
    return hits / len(clauses)


def backbone_fraction(clauses, n_vars):
    """Fraction of variables that take the same value in every model of a satisfiable formula."""
    s = Cadical195(bootstrap_with=[list(c) for c in clauses])
    assert s.solve()
    model = s.get_model()
    fixed = sum(1 for lit in model[:n_vars] if not s.solve(assumptions=[-lit]))
    s.delete()
    return fixed / n_vars


def load_model(ckpt, n_iter, device):
    opts = SimpleNamespace(task="satisfiability", graph="lcg", init_emb="learned",
                           model="neurosat", dim=128, n_iterations=n_iter,
                           n_mlp_layers=2, activation="relu", device=device)
    m = GNN(opts).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def per_instance_loss(model, proc_dir, split, n, label, device, bs=64):
    out = []
    for s in range(0, n, bs):
        datas = [torch.load(os.path.join(proc_dir, split, "%.5d_lcg.pt" % i), weights_only=False)
                 for i in range(s, min(n, s + bs))]
        p = model(Batch.from_data_list(datas).to(device)).cpu()
        out.append(F.binary_cross_entropy(p, torch.full_like(p, float(label)), reduction="none"))
    return torch.cat(out).numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return np.corrcoef(ra, rb)[0, 1]


def quintiles(x, loss, name):
    bins = np.digitize(x, np.quantile(x, [0.2, 0.4, 0.6, 0.8]))
    print(f"    {name:12s} " + "  ".join(
        f"Q{q + 1}: x={x[bins == q].mean():.3f} loss={loss[bins == q].mean():.3f}" for q in range(5)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=os.path.expanduser("~/g4satbench/easy/sr/train"))
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--mid_epoch", type=int, default=14)
    p.add_argument("--n_iterations", type=int, default=26)
    p.add_argument("--out", default=os.path.join(REPO, "experiments", "results", "sensitivity_sr_train.json"))
    o = p.parse_args()

    N = o.limit
    rows = {k: np.zeros(N) for k in ["n", "m", "flip_F", "flip_S", "del_F", "core_frac", "backbone_S"]}
    t0 = time.time()
    for i in range(N):
        nv, Fc = parse_cnf_file(os.path.join(o.data, "unsat", "%.5d.cnf" % i))
        _, Sc = parse_cnf_file(os.path.join(o.data, "sat", "%.5d.cnf" % i))
        rows["n"][i], rows["m"][i] = nv, len(Fc)
        rows["flip_F"][i] = flip_sensitivity(Fc, True)
        rows["flip_S"][i] = flip_sensitivity(Sc, False)
        rows["del_F"][i] = deletion_sensitivity(Fc)
        rows["core_frac"][i] = len(extract_mus(Fc)) / len(Fc)
        rows["backbone_S"][i] = backbone_fraction(Sc, nv)
        if (i + 1) % 500 == 0:
            print(f"  solver stats {i + 1}/{N}  ({time.time() - t0:.0f}s)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = os.path.join(o.data, "processed")
    ckpts = {"mid": os.path.join(RUN, "checkpoints", f"model_{o.mid_epoch}.pt"),
             "best": os.path.join(RUN, "checkpoints", "model_best.pt")}
    for tag, ck in ckpts.items():
        m = load_model(ck, o.n_iterations, device)
        rows[f"loss_F_{tag}"] = per_instance_loss(m, proc, "unsat", N, 0, device)
        rows[f"loss_S_{tag}"] = per_instance_loss(m, proc, "sat", N, 1, device)
        rows[f"loss_pair_{tag}"] = rows[f"loss_F_{tag}"] + rows[f"loss_S_{tag}"]

    with open(o.out, "w") as f:
        json.dump({k: v.tolist() for k, v in rows.items()}, f)
    print("wrote", o.out)

    print("\n=== sensitivity distribution (SR train) ===")
    for k in ["n", "m", "flip_F", "flip_S", "del_F", "core_frac", "backbone_S"]:
        v = rows[k]
        print(f"  {k:11s} mean={v.mean():.3f}  p10={np.quantile(v, .1):.3f}  "
              f"p50={np.median(v):.3f}  p90={np.quantile(v, .9):.3f}")

    for tag in ckpts:
        lf, ls, lp = rows[f"loss_F_{tag}"], rows[f"loss_S_{tag}"], rows[f"loss_pair_{tag}"]
        print(f"\n=== checkpoint: {tag}   mean loss  F={lf.mean():.3f}  S={ls.mean():.3f} ===")
        print("  Spearman(sensitivity, loss):")
        print(f"    UNSAT side   flip_F={spearman(rows['flip_F'], lf):+.3f}  del_F={spearman(rows['del_F'], lf):+.3f}  "
              f"core_frac={spearman(rows['core_frac'], lf):+.3f}  n={spearman(rows['n'], lf):+.3f}  "
              f"m={spearman(rows['m'], lf):+.3f}")
        print(f"    SAT side     flip_S={spearman(rows['flip_S'], ls):+.3f}  "
              f"backbone={spearman(rows['backbone_S'], ls):+.3f}  "
              f"n={spearman(rows['n'], ls):+.3f}  m={spearman(rows['m'], ls):+.3f}")
        print(f"    pair         flip_F={spearman(rows['flip_F'], lp):+.3f}  flip_S={spearman(rows['flip_S'], lp):+.3f}  "
              f"core_frac={spearman(rows['core_frac'], lp):+.3f}  n={spearman(rows['n'], lp):+.3f}")
        print("  loss by sensitivity quintile:")
        quintiles(rows["flip_F"], lf, "flip_F  (F)")
        quintiles(rows["del_F"], lf, "del_F   (F)")
        quintiles(rows["core_frac"], lf, "core    (F)")
        quintiles(rows["flip_S"], ls, "flip_S  (S)")
        quintiles(rows["backbone_S"], ls, "backbone(S)")
        quintiles(rows["n"], lp, "n     (pair)")
    print(f"\n  Spearman(mid-checkpoint loss, best-checkpoint loss) per pair = "
          f"{spearman(rows['loss_pair_mid'], rows['loss_pair_best']):+.3f}")


if __name__ == "__main__":
    main()
