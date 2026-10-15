import argparse
import json
import os
from types import SimpleNamespace

import torch
from g4satbench.data.dataloader import get_dataloader
from g4satbench.models.gnn import GNN

from run_comparison import REPO, parse_valid_curve, train_one

ARMS = [("sr", "sr"), ("obs-mu1", "obs-mu1")]
TARGETS = ["sr", "3-sat"]


def eval_opts(n_iter, device):
    return SimpleNamespace(task="satisfiability", label="satisfiability", graph="lcg",
                           init_emb="learned", model="neurosat", dim=128,
                           n_iterations=n_iter, n_mlp_layers=2, activation="relu",
                           data_fetching="parallel", batch_size=64, device=device)


@torch.no_grad()
def accuracy(model, loader, device):
    hit = tot = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        hit += ((pred > 0.5).float() == data.y).sum().item()
        tot += data.num_graphs
    return hit / tot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=os.path.expanduser("~/g4satbench/easy"))
    p.add_argument("--out", default=os.path.join(REPO, "experiments", "results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[11])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_iterations", type=int, default=26)
    p.add_argument("--lr", type=float, default=1e-4)
    o = p.parse_args()
    os.makedirs(o.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eo = eval_opts(o.n_iterations, device)
    loaders = {t: get_dataloader(os.path.join(o.data_root, t, "test"), ["sat", "unsat"], None, eo, "test")
               for t in TARGETS}
    model = GNN(eo).to(device)

    results = {}
    for seed in o.seeds:
        results[seed] = {}
        for train_src, valid_src in ARMS:
            rd, log, best_ckpt = train_one(train_src, o.data_root, valid_src, o.epochs,
                                           o.batch_size, o.n_iterations, o.lr, seed)
            own_valid = dict(parse_valid_curve(log))
            best_epoch = torch.load(best_ckpt, map_location="cpu", weights_only=False)["epoch"]

            curve = {t: [] for t in TARGETS}
            for e in range(o.epochs):
                sd = torch.load(os.path.join(rd, "checkpoints", f"model_{e}.pt"),
                                map_location=device, weights_only=False)["state_dict"]
                model.load_state_dict(sd)
                model.eval()
                for t in TARGETS:
                    curve[t].append(accuracy(model, loaders[t], device))
                print(f"[seed {seed}] {train_src:8s} epoch {e:2d}  own-valid {own_valid.get(e, float('nan')):.3f}  "
                      + "  ".join(f"{t}-test {curve[t][-1]:.3f}" for t in TARGETS), flush=True)

            results[seed][train_src] = {
                "valid_on": valid_src,
                "best_epoch_on_own_valid": best_epoch,
                "own_valid_curve": [own_valid.get(e) for e in range(o.epochs)],
                "target_test_curve": curve,
                "target_test_at_best": {t: curve[t][best_epoch] for t in TARGETS},
            }
            with open(os.path.join(o.out, "replication.json"), "w") as f:
                json.dump(results, f, indent=2)

    report(results)
    plot(results, o.out)


def report(results):
    print("\n=== model selected on its own validation data, tested on the targets ===")
    hdr = (f"{'seed':>4s} {'train / valid':22s} {'best ep':>7s} {'own valid':>9s} "
           + " ".join(f"{t + ' test':>11s}" for t in TARGETS))
    print(hdr)
    print("-" * len(hdr))
    for seed, arms in results.items():
        for src, r in arms.items():
            be = r["best_epoch_on_own_valid"]
            print(f"{seed:>4d} {src + ' / ' + r['valid_on']:22s} {be:>7d} {r['own_valid_curve'][be]:>9.3f} "
                  + " ".join(f"{r['target_test_at_best'][t]:>11.3f}" for t in TARGETS))

    print("\n=== first epoch at which SR test accuracy reaches each threshold ===")
    ths = [0.55, 0.60, 0.70, 0.75]
    print(f"{'seed':>4s} {'train':10s} " + " ".join(f"{'>=' + str(th):>7s}" for th in ths))
    for seed, arms in results.items():
        for src, r in arms.items():
            c = r["target_test_curve"]["sr"]
            cells = [next((str(e) for e, a in enumerate(c) if a >= th), "-") for th in ths]
            print(f"{seed:>4d} {src:10s} " + " ".join(f"{x:>7s}" for x in cells))


def plot(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6 * len(TARGETS), 4.5), sharey=True)
    for ax, t in zip(axes, TARGETS, strict=True):
        for seed, arms in results.items():
            for src, r in arms.items():
                ax.plot(r["target_test_curve"][t], marker="o", ms=3,
                        label=f"{src} (seed {seed})", color="C0" if src == "sr" else "C3",
                        alpha=0.9 if len(results) == 1 else 0.6)
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.set_xlabel("training epoch")
        ax.set_title(f"{t} test accuracy after each epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend()
    path = os.path.join(out, "replication.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
