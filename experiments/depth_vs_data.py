import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from g4satbench.models.gnn import GNN
from torch_geometric.data import Batch

from run_comparison import REPO, run_dir

RUN = run_dir("sr", 26, 1e-4, 1)


@torch.no_grad()
def acc(model, proc, split, idx, label, device, bs=32):
    hits = 0
    for s in range(0, len(idx), bs):
        datas = [torch.load(os.path.join(proc, split, "%.5d_lcg.pt" % i), weights_only=False)
                 for i in idx[s:s + bs]]
        p = model(Batch.from_data_list(datas).to(device)).cpu()
        hits += ((p > 0.5).float() == label).sum().item()
    return hits / len(idx)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", default=os.path.expanduser("~/g4satbench/easy/sr/train"))
    ap.add_argument("--valid", default=os.path.expanduser("~/g4satbench/easy/sr/valid"))
    ap.add_argument("--sens", default=os.path.join(REPO, "experiments", "results", "sensitivity_sr_train.json"))
    ap.add_argument("--iters", type=int, nargs="+", default=[16, 26, 40, 64, 100])
    ap.add_argument("--k", type=int, default=600, help="pairs per sensitivity group")
    ap.add_argument("--n_valid", type=int, default=400)
    o = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = SimpleNamespace(task="satisfiability", graph="lcg", init_emb="learned",
                           model="neurosat", dim=128, n_iterations=26,
                           n_mlp_layers=2, activation="relu", device=device)
    model = GNN(opts).to(device)
    model.load_state_dict(torch.load(os.path.join(RUN, "checkpoints", "model_best.pt"),
                                     map_location=device, weights_only=False)["state_dict"])
    model.eval()

    with open(o.sens) as f:
        d = {k: np.array(v) for k, v in json.load(f).items()}
    order_S = np.argsort(d["flip_S"])
    lo_S, hi_S = order_S[:o.k].tolist(), order_S[-o.k:].tolist()
    order_F = np.argsort(d["flip_F"])
    lo_F, hi_F = order_F[:o.k].tolist(), order_F[-o.k:].tolist()
    vidx = list(range(o.n_valid))
    ptr, pva = os.path.join(o.train, "processed"), os.path.join(o.valid, "processed")

    print(f"baseline trained with 26 rounds, {o.k} pairs per group\n")
    hdr = (f"{'iters':>5s} | {'SAT lo-sens':>11s} {'SAT hi-sens':>11s} | {'UNSAT lo-sens':>13s} "
           f"{'UNSAT hi-sens':>13s} | {'valid SAT':>9s} {'valid UNSAT':>11s} {'valid all':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for it in o.iters:
        model.opts.n_iterations = it
        s_lo = acc(model, ptr, "sat", lo_S, 1.0, device)
        s_hi = acc(model, ptr, "sat", hi_S, 1.0, device)
        f_lo = acc(model, ptr, "unsat", lo_F, 0.0, device)
        f_hi = acc(model, ptr, "unsat", hi_F, 0.0, device)
        v_s = acc(model, pva, "sat", vidx, 1.0, device)
        v_u = acc(model, pva, "unsat", vidx, 0.0, device)
        print(f"{it:5d} | {s_lo:11.3f} {s_hi:11.3f} | {f_lo:13.3f} {f_hi:13.3f} | "
              f"{v_s:9.3f} {v_u:11.3f} {(v_s + v_u) / 2:9.3f}")


if __name__ == "__main__":
    main()
