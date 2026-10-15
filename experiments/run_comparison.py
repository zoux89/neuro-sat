import argparse
import json
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G4SB = os.path.join(REPO, "G4SATBench")
PY = sys.executable

SOURCES = ["sr", "obs-mu1", "obs-mus", "obs-mix"]
ACC_RE = re.compile(r"Overall accuracy\s*\|\s*([0-9.]+)")
EPOCH_RE = re.compile(r"EPOCH #(\d+)")


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def run_dir(dataset, n_iter, lr, seed):
    return os.path.join(
        G4SB, "runs",
        f"train_task=satisfiability_difficulty=easy_dataset={dataset}_splits=sat_unsat",
        f"graph=lcg_init_emb=learned_model=neurosat_n_iterations={n_iter}"
        f"_lr={lr:.0e}_weight_decay=1e-08_seed={seed}")


def fmt(x, width, prec=3):
    if isinstance(x, float):
        s = f"{x:.{prec}f}"
    else:
        s = "-" if x is None else str(x)
    return s.rjust(width)


def gen_data(data_root, sizes, min_n, max_n, seed):
    tr, va, te = sizes
    gens = os.path.join(G4SB, "g4satbench", "generators")

    def done(path, split="train"):
        d = os.path.join(path, split, "sat")
        return os.path.isdir(d) and len(os.listdir(d)) >= (tr if split == "train" else te)

    common = ["--min_n", str(min_n), "--max_n", str(max_n), "--seed", str(seed),
              "--train_instances", str(tr), "--valid_instances", str(va),
              "--test_instances", str(te)]

    sr_dir = os.path.join(data_root, "sr")
    if not done(sr_dir):
        sh([PY, os.path.join(gens, "sr.py"), sr_dir + "/"] + common)

    for src in ["mu1", "mus", "mix"]:
        d = os.path.join(data_root, f"obs-{src}")
        if not done(d):
            sh([PY, "-m", "obstructions.generate_dataset", d + "/", "--source", src] + common,
               cwd=REPO)

    tsat = os.path.join(data_root, "3-sat")
    if not done(tsat, "test"):
        sh([PY, os.path.join(gens, "3-sat.py"), tsat + "/",
            "--test_instances", str(te), "--min_n", str(min_n),
            "--max_n", str(max_n), "--seed", str(seed)])


def parse_valid_curve(log_path):
    """Return (epoch, validation accuracy) pairs from a train_model.py log.

    Each epoch logs the training accuracy first and the validation accuracy second.
    """
    curve = []
    epoch = None
    accs = []
    with open(log_path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                if epoch is not None and len(accs) >= 2:
                    curve.append((epoch, accs[1]))
                epoch = int(m.group(1))
                accs = []
                continue
            a = ACC_RE.search(line)
            if a:
                accs.append(float(a.group(1)))
    if epoch is not None and len(accs) >= 2:
        curve.append((epoch, accs[1]))
    return curve


def train_one(src, data_root, valid_src, epochs, batch_size, n_iter, lr, seed):
    """Train with train_model.py into a fresh run directory (its logger appends to log.txt)."""
    rd = run_dir(src, n_iter, lr, seed)
    if os.path.isdir(rd):
        shutil.rmtree(rd)
    sh([PY, "train_model.py", "satisfiability", os.path.join(data_root, src, "train") + "/",
        "--train_splits", "sat", "unsat",
        "--valid_dir", os.path.join(data_root, valid_src, "valid") + "/",
        "--valid_splits", "sat", "unsat",
        "--label", "satisfiability", "--graph", "lcg", "--model", "neurosat",
        "--n_iterations", str(n_iter), "--epochs", str(epochs),
        "--lr", str(lr), "--weight_decay", "1e-08",
        "--scheduler", "ReduceLROnPlateau",
        "--batch_size", str(batch_size), "--seed", str(seed)], cwd=G4SB)
    return rd, os.path.join(rd, "log.txt"), os.path.join(rd, "checkpoints", "model_best.pt")


def eval_one(rd, test_dir, n_iter, batch_size):
    """Evaluate model_best of a run and return the accuracy from the log eval_model.py writes."""
    test_dir = os.path.abspath(test_dir)
    difficulty, dataset = test_dir.split(os.path.sep)[-3:-1]
    sh([PY, "eval_model.py", "satisfiability", test_dir + "/",
        os.path.join(rd, "checkpoints", "model_best.pt"),
        "--test_splits", "sat", "unsat", "--label", "satisfiability",
        "--graph", "lcg", "--model", "neurosat", "--n_iterations", str(n_iter),
        "--batch_size", str(batch_size)], cwd=G4SB)
    log = os.path.join(
        rd,
        f"eval_task=satisfiability_difficulty={difficulty}_dataset={dataset}"
        f"_splits=sat_unsat_n_iterations={n_iter}_checkpoint=model_best.txt")
    if not os.path.exists(log):
        return None
    with open(log) as f:
        accs = ACC_RE.findall(f.read())
    return float(accs[-1]) if accs else None


def print_summary(results):
    print("\n=== SUMMARY ===")
    hdr = (f"{'train set':10s} {'valid_on':10s} {'best_val':>9s} {'own test':>9s} "
           f"{'SR test':>8s} {'3SAT test':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for src, r in results.items():
        print(f"{src:10s} {r['valid_on']:10s} {fmt(r['best_valid_acc'], 9)} "
              f"{fmt(r['own_test_acc'], 9)} {fmt(r['sr_test_acc'], 8)} {fmt(r['3sat_test_acc'], 10)}")


def plot(results, out, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    for src, r in results.items():
        if r["valid_curve"]:
            xs, ys = zip(*r["valid_curve"], strict=True)
            plt.plot(xs, ys, marker="o", ms=3, label=src)
    plt.xlabel("epoch")
    plt.ylabel("SR validation accuracy")
    plt.title("NeuroSAT: obstruction vs SR training (validated on SR)")
    plt.axhline(0.5, ls="--", c="gray", lw=0.8)
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(out, f"valid_curves{tag}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=os.path.expanduser("~/g4satbench/easy"))
    p.add_argument("--out", default=os.path.join(REPO, "experiments", "results"))
    p.add_argument("--train_instances", type=int, default=5000)
    p.add_argument("--valid_instances", type=int, default=500)
    p.add_argument("--test_instances", type=int, default=500)
    p.add_argument("--min_n", type=int, default=10)
    p.add_argument("--max_n", type=int, default=40)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_iterations", type=int, default=26)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--valid_on", default="sr",
                   help="dataset whose valid split is used for validation (ignored with --self_valid)")
    p.add_argument("--self_valid", action="store_true",
                   help="validate each model on its own training set's valid split")
    p.add_argument("--only", nargs="+", default=None, help="train only these sources")
    p.add_argument("--tag", default="", help="suffix for the results file and plot")
    p.add_argument("--skip_data", action="store_true")
    opts = p.parse_args()

    os.makedirs(opts.out, exist_ok=True)
    if not opts.skip_data:
        gen_data(opts.data_root,
                 (opts.train_instances, opts.valid_instances, opts.test_instances),
                 opts.min_n, opts.max_n, opts.seed)

    results = {}
    for src in opts.only or SOURCES:
        valid_src = src if opts.self_valid else opts.valid_on
        rd, log, _ = train_one(src, opts.data_root, valid_src, opts.epochs,
                               opts.batch_size, opts.n_iterations, opts.lr, opts.seed)
        curve = parse_valid_curve(log)
        tests = {key: eval_one(rd, os.path.join(opts.data_root, ds, "test"),
                               opts.n_iterations, opts.batch_size)
                 for key, ds in [("own_test_acc", src), ("sr_test_acc", "sr"),
                                 ("3sat_test_acc", "3-sat")]}
        results[src] = {
            "valid_on": valid_src,
            "valid_curve": curve,
            **tests,
            "best_valid_acc": max((a for _, a in curve), default=None),
            "epochs_to_90": next((e for e, a in curve if a >= 0.90), None),
            "epochs_to_95": next((e for e, a in curve if a >= 0.95), None),
        }
        with open(os.path.join(opts.out, f"results{opts.tag}.json"), "w") as f:
            json.dump(results, f, indent=2)

    print_summary(results)
    plot(results, opts.out, opts.tag)


if __name__ == "__main__":
    main()
