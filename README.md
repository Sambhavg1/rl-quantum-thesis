# Deep Reinforcement Learning for Quantum State Preparation and AME Discovery

Companion code and results for the MSc thesis *Deep Reinforcement Learning for
Quantum Optimisation* (Department of Physics, IIT Madras, 2026).

A PPO agent is used to discover gate sequences for two families of quantum
states:

**Track 1 — Cluster-state preparation.** A fidelity-rewarded agent prepares
linear cluster states on *n* qubits, with an optional gate-level noise model.
Driver: `code/cluster_ppo.py`.

**Track 2 — Absolutely Maximally Entangled (AME) state discovery.** A
*target-free* agent searches for AME states on *n* qudits of local dimension
*d* by maximising the minimum von Neumann entropy across all balanced
bipartitions. No target state is supplied; the agent is given only the
entanglement objective. Driver: `code/ame_simple.py`.

The target-free formulation is the central methodological contribution.

## Repository structure

```
rl-quantum-thesis/
├── code/                 The two thesis drivers and the verification tool
│   ├── ame_simple.py        Track 2: target-free AME discovery (CLI)
│   ├── cluster_ppo.py       Track 1: cluster-state preparation (CLI)
│   ├── replay_analyse.py    Rebuild a logged sequence and recompute all
│   │                        bipartition entropies (independent verification)
│   └── README.md
├── experiments/          Research scripts for the harder cases (AME(4,6),
│                         (5,2), (6,2), non-Clifford-cost studies, noise
│                         variants). Exploratory, not the clean drivers.
│   └── README.md
├── results/              Curated results reported in the thesis
│   ├── ame/                 Per-configuration reports, circuits, and plots
│   ├── cluster/             Cluster-state training and comparison plots
│   └── README.md
├── figures/              Figures reproduced in the thesis
├── thesis/
│   └── thesis.pdf
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── DATA_NOTES.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── LICENSE
```

## Quickstart

```bash
conda env create -f environment.yml
conda activate quantum

# Track 2: discover AME(4,3)
python code/ame_simple.py --n 4 --d 3 --episodes 10000 --seed 42 --run_name ame43_s42

# Track 1: prepare a 6-qubit cluster state
python code/cluster_ppo.py --n 6 --episodes 50000 --seed 42 --run_name cluster6_ideal_s42

# Verify any logged sequence, independently of training
python code/replay_analyse.py --n 2 --d 2 --seq "F_0-SUM_0_1"
```

Both drivers are command-line driven; run with `--help` for all options. See
`docs/REPRODUCIBILITY.md` for the exact commands behind the reported runs.

## Results

`results/` holds the curated outputs reported in the thesis: for each AME
configuration a text report, a circuit, and stability / length / top-sequence
plots, plus the cluster-state training and DQN-versus-PPO comparison figures.
See `results/README.md` for the scope of what is included.

## Software environment

Versions are pinned in `requirements.txt` and `environment.yml`. Core stack:
Python 3.11, TensorFlow 2.18, NumPy, Matplotlib. The environments use the
original `gym` API; see `code/README.md` for the note on migrating to
`gymnasium`.

## Citation

A machine-readable entry is in `CITATION.cff`.

## Licence

Code is released under the MIT Licence (`LICENSE`). The thesis document in
`thesis/` is the author's own work, shared for reference.
