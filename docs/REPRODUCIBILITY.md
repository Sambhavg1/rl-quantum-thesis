# Reproducibility

## Environment

```bash
conda env create -f ../environment.yml
conda activate quantum
```

## AME discovery (Track 2)

`code/ame_simple.py` is fully command-line driven:

```bash
python code/ame_simple.py --n 4 --d 3 --episodes 10000 --seed 42 --run_name ame43_s42
python code/ame_simple.py --n 3 --d 2 --episodes  5000 --seed 42 --run_name ame32_s42
python code/ame_simple.py --n 5 --d 2 --episodes 15000 --seed 42 --run_name ame52_s42
python code/ame_simple.py --n 6 --d 2 --episodes 20000 --seed 42 --run_name ame62_s42
```

Each run writes a per-run directory containing the per-episode CSV, the best
sequence, the best state in Dirac notation, the circuit, a JSON summary and
config, and the training and entropy-climb plots.

## Cluster-state preparation (Track 1)

```bash
python code/cluster_ppo.py --n 6  --episodes 50000  --seed 42 --run_name cluster6_ideal_s42
python code/cluster_ppo.py --n 6  --episodes 50000  --seed 42 --noisy --run_name cluster6_noisy_s42
python code/cluster_ppo.py --n 10 --episodes 100000 --seed 42 --run_name cluster10_ideal_s42
```

## Verifying a reported state

```bash
python code/replay_analyse.py --n <n> --d <d> --seq "<gate-sequence>"
```

This reconstructs the state from |0...0> and recomputes every
balanced-bipartition entropy, with no dependence on the reward logged during
training. Run it on any state before it goes into a figure or a table.

## Determinism

Runs seed `numpy`, `random`, and TensorFlow. Exact bit-for-bit reproducibility
across machines is not guaranteed (GPU kernels, BLAS threading, and TensorFlow
versions all affect floating-point results), but seeded runs on the same
machine and versions are repeatable.

## Harder cases

The AME(4,6), (5,2) and (6,2) scripts and the non-Clifford and noise ablations
are in `../experiments/`; see `experiments/README.md`. Several hard-code cluster
paths and need adjustment before running.
