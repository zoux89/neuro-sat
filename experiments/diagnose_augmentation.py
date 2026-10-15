import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from g4satbench.models.gnn import GNN
from g4satbench.utils.utils import parse_cnf_file
from torch_geometric.data import Batch

from run_comparison import run_dir


def load_model(ckpt, n_iter, device):
    opts = SimpleNamespace(task="satisfiability", graph="lcg", init_emb="learned",
                           model="neurosat", dim=128, n_iterations=n_iter,
                           n_mlp_layers=2, activation="relu", device=device)
    m = GNN(opts).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def score(model, proc_dir, split, indices, label, device, bs=64):
    probs = []
    for s in range(0, len(indices), bs):
        datas = []
        for i in indices[s:s + bs]:
            d = torch.load(os.path.join(proc_dir, split, "%.5d_lcg.pt" % i), weights_only=False)
            d.y = torch.tensor(float(label))
            datas.append(d)
        probs.append(model(Batch.from_data_list(datas).to(device)).cpu())
    p = torch.cat(probs)
    y = torch.full_like(p, float(label))
    loss = F.binary_cross_entropy(p, y, reduction="none")
    acc = ((p > 0.5).float() == y).float()
    return p.numpy(), loss.numpy(), acc.numpy()


def struct_stats(cnf_dir, indices):
    n, m = [], []
    for i in indices:
        nv, cl = parse_cnf_file(os.path.join(cnf_dir, "%.5d.cnf" % i))
        n.append(nv)
        m.append(len(cl))
    n, m = np.array(n), np.array(m)
    return n.mean(), m.mean(), (m / n).mean()


def check_layout(aug_root, n_base):
    for i in range(3):
        _, mus = parse_cnf_file(os.path.join(aug_root, "unsat", "%.5d.cnf" % (n_base + 2 * i)))
        _, kept = parse_cnf_file(os.path.join(aug_root, "unsat", "%.5d.cnf" % (n_base + 2 * i + 1)))
        _, sub = parse_cnf_file(os.path.join(aug_root, "sat", "%.5d.cnf" % (n_base + i)))
        _, orig = parse_cnf_file(os.path.join(aug_root, "unsat", "%.5d.cnf" % i))
        print(f"  i={i}: |orig|={len(orig)} |F-noncore|={len(kept)} |MUS|={len(mus)} |MUS-1|={len(sub)}")
        assert len(sub) == len(mus) - 1 and len(kept) < len(orig) and len(mus) <= len(kept)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aug_root", default=os.path.expanduser("~/g4satbench/easy/sr-aug/train"))
    p.add_argument("--sr_valid", default=os.path.expanduser("~/g4satbench/easy/sr/valid"))
    p.add_argument("--ckpt", default=os.path.join(run_dir("sr", 26, 1e-4, 1), "checkpoints", "model_best.pt"))
    p.add_argument("--n_iterations", type=int, default=26)
    p.add_argument("--n_base", type=int, default=3000, help="SR pairs in the source split")
    p.add_argument("--limit", type=int, default=1000, help="instances scored per subset")
    o = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(o.ckpt, o.n_iterations, device)
    proc = os.path.join(o.aug_root, "processed")
    B, L = o.n_base, o.limit
    check_layout(o.aug_root, B)

    subsets = [
        ("SR original (SAT)", "sat", list(range(0, L)), 1),
        ("SR original (UNSAT)", "unsat", list(range(0, L)), 0),
        ("MUS - 1 clause (SAT)", "sat", list(range(B, B + L)), 1),
        ("SAT - 20% cl. (SAT)", "sat", list(range(2 * B, 2 * B + L)), 1),
        ("MUS (UNSAT)", "unsat", [B + 2 * i for i in range(L)], 0),
        ("F - 20% noncore (UNSAT)", "unsat", [B + 2 * i + 1 for i in range(L)], 0),
    ]
    print(f"\nmodel: {o.ckpt}\n")
    hdr = f"{'subset':26s} {'acc':>6s} {'loss':>6s} {'|p-.5|':>7s} {'n':>6s} {'m':>7s} {'m/n':>6s}"
    print(hdr)
    print("-" * len(hdr))
    losses = {}
    for name, split, idx, lab in subsets:
        pr, lo, ac = score(model, proc, split, idx, lab, device)
        n, m, r = struct_stats(os.path.join(o.aug_root, split), idx)
        losses[name] = lo
        print(f"{name:26s} {ac.mean():6.3f} {lo.mean():6.3f} {np.abs(pr - 0.5).mean():7.3f} "
              f"{n:6.1f} {m:7.1f} {r:6.2f}")

    vproc = os.path.join(o.sr_valid, "processed")
    nv = len(glob.glob(os.path.join(o.sr_valid, "sat", "*.cnf")))
    for split, lab in (("sat", 1), ("unsat", 0)):
        pr, lo, ac = score(model, vproc, split, list(range(nv)), lab, device)
        print(f"{'SR valid (' + split.upper() + ')':26s} {ac.mean():6.3f} {lo.mean():6.3f} "
              f"{np.abs(pr - 0.5).mean():7.3f}")

    print("\nshare of total loss over the subsets:")
    tot = sum(lo.sum() for lo in losses.values())
    for name, lo in losses.items():
        print(f"  {name:26s} {100 * lo.sum() / tot:5.1f}%")


if __name__ == "__main__":
    main()
