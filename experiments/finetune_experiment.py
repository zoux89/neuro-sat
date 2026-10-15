import argparse
import json
import os
import shutil

from run_comparison import G4SB, PY, REPO, eval_one, fmt, parse_valid_curve, run_dir, sh


def obs_ckpt(src, n_iter, lr, seed):
    return os.path.join(run_dir(src, n_iter, lr, seed), "checkpoints", "model_best.pt")


def finetune_run_dir(init, n_iter, lr, seed):
    """train_model.py nests a run started from --checkpoint inside the checkpoint's run directory."""
    sr_dir = run_dir("sr", n_iter, lr, seed)
    if init == "scratch":
        return sr_dir
    return os.path.join(run_dir(init, n_iter, lr, seed),
                        os.path.relpath(sr_dir, os.path.join(G4SB, "runs")))


def run(init, budget, data_root, epochs, batch_size, n_iter, lr, seed):
    rd = finetune_run_dir(init, n_iter, lr, seed)
    if os.path.isdir(rd):
        shutil.rmtree(rd)

    cmd = [PY, "train_model.py", "satisfiability",
           os.path.join(data_root, "sr", "train") + "/",
           "--train_splits", "sat", "unsat",
           "--valid_dir", os.path.join(data_root, "sr", "valid") + "/",
           "--valid_splits", "sat", "unsat",
           "--label", "satisfiability", "--graph", "lcg", "--model", "neurosat",
           "--n_iterations", str(n_iter), "--epochs", str(epochs),
           "--lr", str(lr), "--weight_decay", "1e-08",
           "--scheduler", "ReduceLROnPlateau",
           "--batch_size", str(batch_size), "--seed", str(seed)]
    if budget is not None:
        cmd += ["--train_sample_size", str(budget)]
    if init != "scratch":
        cmd += ["--checkpoint", obs_ckpt(init, n_iter, lr, seed)]
    sh(cmd, cwd=G4SB)

    curve = parse_valid_curve(os.path.join(rd, "log.txt"))

    def epochs_to(th):
        return next((e for e, a in curve if a >= th), None)

    return {
        "curve": curve,
        "sr_test_acc": eval_one(rd, os.path.join(data_root, "sr", "test"), n_iter, batch_size),
        "best_valid_acc": max((a for _, a in curve), default=None),
        "epochs_to_0.60": epochs_to(0.60),
        "epochs_to_0.70": epochs_to(0.70),
        "epochs_to_0.75": epochs_to(0.75),
    }


def plot(results, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for init, r in results.items():
        if r["curve"]:
            xs, ys = zip(*r["curve"], strict=True)
            plt.plot(xs, ys, marker="o", ms=3, label=init)
    plt.xlabel("SR fine-tuning epoch")
    plt.ylabel("SR validation accuracy")
    plt.title(f"Obstruction pretraining, then SR fine-tuning (SR budget={tag})")
    plt.axhline(0.5, ls="--", c="gray", lw=0.8)
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(out, f"finetune_{tag}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=os.path.expanduser("~/g4satbench/easy"))
    p.add_argument("--out", default=os.path.join(REPO, "experiments", "results"))
    p.add_argument("--budget", type=int, default=0, help="SR training pairs to use, 0 for all")
    p.add_argument("--inits", nargs="+", default=["scratch", "obs-mus", "obs-mix"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_iterations", type=int, default=26)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=123)
    opts = p.parse_args()

    budget = opts.budget or None
    tag = "full" if budget is None else str(budget)

    os.makedirs(opts.out, exist_ok=True)
    results = {}
    for init in opts.inits:
        results[init] = run(init, budget, opts.data_root, opts.epochs,
                            opts.batch_size, opts.n_iterations, opts.lr, opts.seed)
        with open(os.path.join(opts.out, f"finetune_{tag}.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== SR budget = {tag}, validated on SR ===")
    hdr = f"{'init':10s} {'best_val':>9s} {'>=0.60':>7s} {'>=0.70':>7s} {'>=0.75':>7s} {'SR test':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for init, r in results.items():
        print(f"{init:10s} {fmt(r['best_valid_acc'], 9)} {fmt(r['epochs_to_0.60'], 7)} "
              f"{fmt(r['epochs_to_0.70'], 7)} {fmt(r['epochs_to_0.75'], 7)} {fmt(r['sr_test_acc'], 8)}")

    plot(results, opts.out, tag)


if __name__ == "__main__":
    main()
