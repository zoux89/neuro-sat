# Obstruction training for NeuroSAT

This repository tests whether the obstruction-training method of Abu-Khzam, Abd El-Wahab and Yosri ([arXiv:2006.04689](https://arxiv.org/abs/2006.04689)) works for SAT as well as for vertex cover, using NeuroSAT ([Selsam et al.](https://arxiv.org/abs/1802.03685)). Models, data generators and the training loop come from [G4SATBench](https://github.com/zhaoyu-li/G4SATBench). A copy of it is included in this repository with a few small fixes so it runs on current versions of torch and PyG.

Training NeuroSAT only on obstructions does not transfer to SR, the random formula distribution NeuroSAT is normally trained on. Obstructions are useful for pretraining: with 500 SR training pairs, a model pretrained on MUS cores reaches 0.75 SR test accuracy, against 0.53 for a model trained from scratch on the same pairs. Adding obstruction-derived formulas to the SR training set lowered accuracy at equal compute. The [analysis](#analysis) section explains both negative results. All experiments use G4SATBench's easy setting and one or two seeds.

## Background

Abu-Khzam et al. train a GNN for minimum vertex cover on obstructions, the graphs that are minimal in the minor order among those with vertex cover larger than k. They report the same solution quality as [Dai et al.](https://arxiv.org/abs/1704.01665) with about 200 times fewer training iterations.

The same idea applies to SAT. Deleting a clause never makes a satisfiable formula unsatisfiable, so the satisfiable formulas form a deletion-closed family. Its obstructions are the minimally unsatisfiable (MU) formulas, which are unsatisfiable but become satisfiable when any single clause is removed.

| Vertex cover (Abu-Khzam et al.) | SAT (this repository) |
| --- | --- |
| Minor order (deletion and contraction) | Clause deletion |
| Minor-closed family vc(G) ≤ k | Satisfiable formulas |
| Minimal graphs with vc(G) > k | Minimally unsatisfiable formulas |
| Obstruction sets grown with the Dinneen-Lai rules | MU(1) formulas grown by clause splitting |
| Isomorphic duplicates removed with nauty | nauty canonical hash of the literal-clause graph |

## Obstruction datasets

The deficiency of a formula is its number of clauses minus its number of variables. Three training sets are built from MU formulas:

- `obs-mu1` contains MU formulas of deficiency 1. They are generated from the empty clause by repeatedly replacing a clause C with C ∨ v and C ∨ ¬v for a fresh variable v, which produces every MU(1) formula.
- `obs-mus` contains unsatisfiable SR formulas shrunk to a minimal unsatisfiable subset (MUS) by deletion-based extraction. Their deficiency is typically between 3 and 6.
- `obs-mix` takes half of its formulas from each of the other two.

Each MU formula F is paired with a satisfiable twin that differs from it in one literal, following NeuroSAT's SR pairing. For any clause c of F, removing c leaves a satisfiable formula whose models all falsify c, so flipping one literal of c makes the formula satisfiable. A SAT solver confirms that every MU formula is minimally unsatisfiable and every twin is satisfiable, and isomorphic duplicates are dropped.

## Repository layout

```
neuro-sat/
├── obstructions/
│   ├── mu1.py                      MU(1) generator (clause splitting)
│   ├── mus.py                      MUS extraction from SR formulas
│   ├── satutil.py                  solving, MU check, one-literal SAT twin
│   ├── canonical.py                nauty canonical hash of the literal-clause graph
│   ├── generate_dataset.py         writes obstruction datasets in G4SATBench format
│   └── augment_monotone.py         derives extra labelled formulas from an SR train split
├── experiments/
│   ├── run_comparison.py           SR vs obstruction training, transfer to SR and 3-SAT
│   ├── finetune_experiment.py      obstruction pretraining followed by SR fine-tuning
│   ├── replicate_abukhzam.py       the paper's protocol with per-epoch target accuracy
│   ├── augmentation_experiment.py  SR vs SR with derived formulas
│   ├── run_aug.sh                  medium test sets, augmentation, 3-seed comparison
│   ├── diagnose_augmentation.py    baseline accuracy and loss on each derived subset
│   ├── sensitivity_vs_loss.py      boundary sensitivity vs per-instance loss
│   ├── depth_vs_data.py            accuracy vs number of message-passing rounds
│   └── results/                    JSON results and plots
├── compat/                         replacements for torch_scatter and kmeans_pytorch
├── patches/g4satbench.patch        our changes to G4SATBench
└── G4SATBench/                     G4SATBench at commit 6863b33 with the patch applied
```

## Setup

Tested on Fedora with Python 3.14, torch 2.11.0 built for CUDA 13.0, and an RTX 5070 Laptop GPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Install torch first from the index that matches your CUDA version. `pynauty` is built from source and needs a C compiler. `requirements.txt` installs `compat/` and `G4SATBench/` in editable mode, so run it from the repository root.

G4SATBench pins torch 1.13, torch_scatter 2.1 and SSH-only forks of cnfgen and kmeans_pytorch, none of which install with Python 3.14 and current CUDA. The `compat/` package replaces `torch_scatter` with `torch_geometric.utils.scatter` and `kmeans_pytorch` with a simple k-means. The [patch](patches/g4satbench.patch) to G4SATBench makes three changes:

- `Cadical` is imported as `Cadical195`, its current name in python-sat.
- `torch.load` is called with `weights_only=False`, because torch 2.6 changed the default.
- `3-sat.py` reads clauses from cnfgen 0.9 directly, since `_compress_clause` was removed.

## Reproducing the results

Datasets are written to `~/g4satbench/<difficulty>/<dataset>/{train,valid,test}/{sat,unsat}`, and every script accepts `--data_root` to change this. Training runs and checkpoints are written to `G4SATBench/runs/`. Run the commands from the repository root in the order below, because later steps load checkpoints written by earlier ones.

```bash
# SR, obstruction and 3-SAT datasets (3000/400/400 pairs, 10 to 40 variables),
# one model per training set, all validated on SR
python experiments/run_comparison.py --train_instances 3000 --valid_instances 400 \
    --test_instances 400 --epochs 30

# obstruction models selected on their own validation split, tested on SR and 3-SAT
python experiments/run_comparison.py --skip_data --self_valid \
    --only obs-mu1 obs-mus obs-mix --epochs 30 --tag _selfvalid

# obstruction pretraining, then fine-tuning on all SR pairs or on 500 of them
python experiments/finetune_experiment.py --budget 0   --inits scratch obs-mus obs-mix --epochs 25
python experiments/finetune_experiment.py --budget 500 --inits scratch obs-mus obs-mix --epochs 25

# Abu-Khzam et al.'s protocol with SR and MU(1)
python experiments/replicate_abukhzam.py --seeds 11

# medium test sets, augmented SR train split, baseline vs augmented over 3 seeds
sh experiments/run_aug.sh

# diagnostics on the seed-1 SR baseline from the augmentation run
python experiments/diagnose_augmentation.py
python experiments/sensitivity_vs_loss.py
python experiments/depth_vs_data.py
```

## Results

All numbers are accuracies on balanced SAT/UNSAT test sets. Easy test sets have 10 to 40 variables and 400 pairs, medium test sets have 40 to 200 variables and 200 pairs. Unless stated otherwise, models are trained on 3000 pairs for 30 epochs with 26 message-passing rounds.

### Training on obstructions only

When all four models are validated on SR, the SR model reaches 0.78 validation accuracy within 30 epochs and the three obstruction models stay at 0.50 throughout ([validation curves](experiments/results/valid_curves.png)). Selecting each obstruction model on its own validation split gives the following test accuracies:

| Training set | Own test set | SR test | 3-SAT test |
| --- | --- | --- | --- |
| SR | 0.79 | 0.79 | 0.80 |
| obs-mu1 | 1.00 | 0.50 | 0.50 |
| obs-mus | 0.78 | 0.50 | 0.73 |
| obs-mix | 0.94 | 0.50 | 0.50 |

The models learn their own distribution but nothing that helps on SR. Following Abu-Khzam et al.'s protocol more closely, with each model selected on its own validation data and target accuracy recorded after every epoch, gives the same outcome on a different seed. The SR model reaches 0.78 on SR and 0.81 on 3-SAT, while the MU(1) model is at 0.50 on both after every epoch ([per-epoch curves](experiments/results/replication.png)).

### Obstruction pretraining followed by SR fine-tuning

Models are initialised randomly or from an obstruction model, then fine-tuned on SR for 25 epochs and validated on SR. With all 3000 SR pairs:

| Initialisation | Epochs to reach 0.70 | Epochs to reach 0.75 | Best SR valid | SR test |
| --- | --- | --- | --- | --- |
| Random | 14 | 18 | 0.78 | 0.80 |
| obs-mus | 1 | 4 | 0.78 | 0.81 |
| obs-mix | 9 | 24 | 0.77 | 0.78 |

With 500 SR pairs:

| Initialisation | Best SR valid | SR test |
| --- | --- | --- |
| Random | 0.54 | 0.53 |
| obs-mus | 0.73 | 0.75 |
| obs-mix | 0.58 | 0.56 |

Pretraining on MUS cores reduces the fine-tuning needed to reach 0.75 SR validation accuracy from 18 epochs to 4 ([curves](experiments/results/finetune_full.png)). With 500 pairs, the randomly initialised model stays near chance, while the obs-mus model reaches 0.75, which is 0.05 below the random-init model trained on all 3000 pairs ([curves](experiments/results/finetune_500.png)). Pretraining on the mix, which includes MU(1) formulas, helps much less.

### Monotone augmentation

[`augment_monotone.py`](obstructions/augment_monotone.py) adds formulas whose labels follow from monotonicity under clause deletion. For each unsatisfiable SR training formula F it adds the MUS of F (UNSAT), the MUS with one clause removed (SAT), and F with 20% of its non-core clauses removed (UNSAT). For each satisfiable SR formula it adds the formula with 20% of its clauses removed (SAT). The 3000 SR pairs become 9000 SAT and 9000 UNSAT formulas. All runs are validated on the original SR validation split.

| Training set | Gradient steps | Seeds | Easy SR | Easy 3-SAT | Medium SR | Medium 3-SAT |
| --- | --- | --- | --- | --- | --- | --- |
| SR, 30 epochs | 720 | 2 | 0.79 | 0.79 | 0.58 | 0.71 |
| Augmented, 10 epochs | 710 | 2 | 0.73 | 0.71 | 0.57 | 0.59 |
| Augmented, 30 epochs | 2130 | 1 | 0.81 | 0.80 | 0.59 | 0.64 |

Values are means over the available seeds ([raw results](experiments/results/augmentation.json)). The run was planned with three seeds, but its training process received SIGTERM partway through seed 2, so seed 3 and the second seed of the 30-epoch augmented run are missing. At equal compute, augmentation lowers accuracy on all four test sets. With three times the compute it is within 0.02 of the baseline on three test sets and 0.07 lower on medium 3-SAT.

## Analysis

### Obstruction training

On SR, the obstruction models predict a single class. The obs-mu1 model labels all 800 SR test formulas satisfiable, and the obs-mus model scores 0.504. The training sets differ from SR in clause density, and their SAT twins often contain a pure literal, which tells the model the formula is satisfiable:

| Training set | Clauses per variable | SAT formulas with a pure literal | UNSAT formulas with a pure literal |
| --- | --- | --- | --- |
| SR | 6.07 | 0.2% | 0.2% |
| obs-mu1 | 1.04 | 22.7% | 0% |
| obs-mus | 1.76 | 8.0% | 0% |
| obs-mix | 1.17 | 15.9% | 0% |
| 3-SAT (test set) | 4.6 | 6.0% | 3.3% |

An MU formula cannot contain a pure literal. Setting a pure literal true satisfies every clause it occurs in, and the remaining clauses form a proper subset of the formula, which is satisfiable. When the twin construction flips the only occurrence of a literal with a given sign, the twin gains a pure literal and is satisfiable for that reason alone, which a GNN can detect from the neighbourhood of one variable. In obs-mu1, a flip can also turn a clause into a copy of its sibling from the same split, and removing the duplicate leaves the twin with one clause fewer than its MU partner.

The bigger problem is density. SR formulas near the satisfiability threshold have about 6 clauses per variable. An MU formula of deficiency d has only d more clauses than variables, so the obstruction sets have 1 to 1.8. Models trained only on sparse formulas output a constant on dense ones. Pretraining still helps because fine-tuning on SR supplies the dense formulas, and the obs-mus weights are a better starting point for that than a random initialisation.

The obs-mus model reaches 0.73 on 3-SAT while staying at 0.50 on SR, and we don't know why. One difference is clause length: MUS cores average 3.15 literals per clause, 3-SAT has exactly 3, and SR averages 4.2.

### Monotone augmentation

[`diagnose_augmentation.py`](experiments/diagnose_augmentation.py) scores the SR-only baseline (seed 1) on each part of the augmented training set:

| Subset | Label | Baseline accuracy | Mean loss | Share of total loss | Clauses per variable | With a pure literal |
| --- | --- | --- | --- | --- | --- | --- |
| SR original | SAT | 0.68 | 0.46 | 18% | 6.07 | 0.2% |
| SR original | UNSAT | 0.92 | 0.39 | 15% | 6.07 | 0.2% |
| MUS minus one clause | SAT | 0.75 | 0.45 | 18% | 1.50 | 39% |
| SAT minus 20% of clauses | SAT | 0.90 | 0.18 | 7% | 4.88 | 0.5% |
| MUS | UNSAT | 0.64 | 0.63 | 25% | 1.56 | 0% |
| UNSAT minus 20% of non-core clauses | UNSAT | 0.88 | 0.43 | 17% | 5.15 | 0.5% |

The baseline's errors are mostly on satisfiable SR formulas (0.66 on the SAT half of SR validation, 0.92 on the UNSAT half), and neither kind of added SAT formula targets them. Dropping 20% of a satisfiable formula's clauses moves it away from the threshold, and the baseline already classifies 0.90 of these correctly, so they contribute 7% of the loss. The MUS-derived formulas account for 43% of the loss, but they have a quarter of SR's clause density, and 39% of the satisfiable ones contain a pure literal for the same reason as the obstruction twins. Two thirds of the augmented set is derived formulas, so in the run with equal compute each SR pair is seen 10 times instead of 30.

### Boundary sensitivity and message-passing depth

[`sensitivity_vs_loss.py`](experiments/sensitivity_vs_loss.py) uses a solver to measure how close each SR training formula is to the SAT/UNSAT boundary, for example the fraction of single-literal flips that change its label, and compares this with the baseline's per-instance loss. At the best checkpoint, the Spearman correlation between these measures and the loss is between +0.14 and +0.32. Loss on satisfiable formulas rises from 0.22 in the least sensitive fifth to 0.65 in the most sensitive fifth. Formula size correlates more strongly with loss, at +0.28 for satisfiable and +0.60 for unsatisfiable formulas.

[`depth_vs_data.py`](experiments/depth_vs_data.py) evaluates the same baseline, trained with 26 message-passing rounds, at other round counts:

| Rounds at test time | SR valid SAT | SR valid UNSAT | SR valid overall |
| --- | --- | --- | --- |
| 16 | 0.90 | 0.22 | 0.56 |
| 26 | 0.66 | 0.92 | 0.79 |
| 40 | 0.76 | 0.78 | 0.77 |
| 64 | 0.80 | 0.76 | 0.78 |
| 100 | 0.83 | 0.69 | 0.76 |

More rounds move predictions from UNSAT toward SAT without raising overall accuracy. On the most boundary-sensitive satisfiable training formulas, accuracy rises from 0.55 at 26 rounds to 0.77 at 100, while accuracy on the most sensitive unsatisfiable ones falls from 0.88 to 0.67.

## Limitations

- Most results come from one seed and the augmentation results from two. All training used the easy setting.
- Only NeuroSAT on the literal-clause graph was tested.
- The SAT twins come from a single literal flip. Twins without pure literals, or obstructions closer to SR's clause density, were not tried.
- MU formulas come with exact unsat-core labels, which could be used for G4SATBench's `core_variable` task. This was not tried.

## References

- F. N. Abu-Khzam, M. M. Abd El-Wahab, N. Yosri. Graph Minors Meet Machine Learning: the Power of Obstructions. [arXiv:2006.04689](https://arxiv.org/abs/2006.04689), 2020.
- D. Selsam, M. Lamm, B. Bünz, P. Liang, L. de Moura, D. L. Dill. Learning a SAT Solver from Single-Bit Supervision. ICLR 2019. [arXiv:1802.03685](https://arxiv.org/abs/1802.03685).
- Z. Li, J. Guo, X. Si. G4SATBench: Benchmarking and Advancing SAT Solving with Graph Neural Networks. TMLR 2024. [arXiv:2309.16941](https://arxiv.org/abs/2309.16941).
- H. Dai, E. B. Khalil, Y. Zhang, B. Dilkina, L. Song. Learning Combinatorial Optimization Algorithms over Graphs. NeurIPS 2017. [arXiv:1704.01665](https://arxiv.org/abs/1704.01665).

The included copy of G4SATBench is distributed under the [MIT license](G4SATBench/LICENSE).
