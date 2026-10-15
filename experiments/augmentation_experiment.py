import argparse
import glob
import json
import math
import os
import statistics

from run_comparison import REPO, eval_one, parse_valid_curve, train_one

TESTS = [("easy", "sr"), ("easy", "3-sat"), ("medium", "sr"), ("medium", "3-sat")]


def n_pairs(data_root, dataset):
    return len(glob.glob(os.path.join(data_root, "easy", dataset, "train", "sat", "*.cnf")))


def evaluate(rd, o):
    out = {}
    for diff, ds in TESTS:
        out[f"{diff}/{ds}"] = eval_one(rd, os.path.join(o.data_root, diff, ds, "test"),
                                       o.n_iterations, o.eval_batch_size)
    curve = parse_valid_curve(os.path.join(rd, "log.txt"))
    out["best_valid"] = max((a for _, a in curve), default=None)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=os.path.expanduser("~/g4satbench"))
    p.add_argument("--out", default=os.path.join(REPO, "experiments", "results"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--base_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--n_iterations", type=int, default=26)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--skip_epoch_matched", action="store_true")
    o = p.parse_args()
    os.makedirs(o.out, exist_ok=True)

    nb, na = n_pairs(o.data_root, "sr"), n_pairs(o.data_root, "sr-aug")
    spe_b = math.ceil(nb / o.batch_size)
    spe_a = math.ceil(na / o.batch_size)
    aug_epochs = max(1, round(o.base_epochs * spe_b / spe_a))
    arms = [("baseline", "sr", o.base_epochs),
            ("aug_compute_matched", "sr-aug", aug_epochs)]
    if not o.skip_epoch_matched:
        arms.append(("aug_epoch_matched_more_compute", "sr-aug", o.base_epochs))
    meta = {"n_pairs": {"sr": nb, "sr-aug": na},
            "steps": {name: ep * (spe_b if ds == "sr" else spe_a) for name, ds, ep in arms},
            "epochs": {name: ep for name, _, ep in arms}}
    print(json.dumps(meta, indent=2), flush=True)

    results = {name: {} for name, _, _ in arms}
    path = os.path.join(o.out, "augmentation.json")
    for seed in o.seeds:
        for name, ds, ep in arms:
            rd, _, _ = train_one(ds, os.path.join(o.data_root, "easy"), "sr", ep,
                                 o.batch_size, o.n_iterations, o.lr, seed)
            results[name][str(seed)] = evaluate(rd, o)
            with open(path, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=2)

    print("\n=== mean ± std over seeds ===")
    cols = [f"{d}/{s}" for d, s in TESTS]
    print(f"{'arm':32s} {'steps':>6s} " + " ".join(f"{c:>15s}" for c in cols))
    for name, _, _ in arms:
        cells = []
        for c in cols:
            vals = [r[c] for r in results[name].values() if r.get(c) is not None]
            if len(vals) >= 2:
                cells.append(f"{statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}")
            elif vals:
                cells.append(f"{vals[0]:.3f}")
            else:
                cells.append("-")
        print(f"{name:32s} {meta['steps'][name]:>6d} " + " ".join(f"{x:>15s}" for x in cells))


if __name__ == "__main__":
    main()
