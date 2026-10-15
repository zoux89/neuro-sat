#!/bin/sh
set -eu
PY=${PY:-python}
DATA=${DATA:-$HOME/g4satbench}
cd "$(dirname "$0")/.."

echo "== $(date) medium test sets =="
$PY G4SATBench/g4satbench/generators/sr.py    "$DATA/medium/sr/"    --test_instances 200 --min_n 40 --max_n 200 --seed 7
$PY G4SATBench/g4satbench/generators/3-sat.py "$DATA/medium/3-sat/" --test_instances 200 --min_n 40 --max_n 200 --seed 7

echo "== $(date) augmenting the SR train split =="
rm -rf "$DATA/easy/sr-aug"
$PY -m obstructions.augment_monotone "$DATA/easy/sr/train" "$DATA/easy/sr-aug/train" --seed 0

echo "== $(date) training =="
$PY experiments/augmentation_experiment.py --data_root "$DATA" --seeds 1 2 3 --base_epochs 30

echo "== $(date) done =="
