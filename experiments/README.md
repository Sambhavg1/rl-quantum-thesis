# Experiments

Research scripts used for the harder cases and for ablations, beyond the two
clean drivers in `code/`. These are working scripts: less uniform, often
machine- or cluster-specific, and kept here for transparency and reuse rather
than as polished entry points. For the canonical, documented runs use
`code/ame_simple.py` and `code/cluster_ppo.py`.

## Subfolders

- **`ame46/`** — AME(4,6), the hardest case studied. The golden AME(4,6) state
  is non-stabilizer, so a Clifford-only agent cannot reach it; these scripts
  add non-Clifford resources and explore circuit decompositions. Variants cover
  a tuned single-non-Clifford "gold gate", a controlled-phase gate set, a
  matrix-decomposition route, and scaled multi-angle runs.

- **`scaling/`** — Specialised drivers for other `(n, d)`: parallelised and
  GPU AME(4,3) variants, a topology study, AME(5,2) (convergence and masked
  action space), AME(6,2), and an N=4 approximate-target experiment, with their
  submission scripts.

- **`nc_penalty_study/`** — Ablation on the number of non-Clifford gates
  (1, 3, 10), each with and without an explicit gate-count penalty in the
  reward. Used to study the cost of leaving the Clifford set.

- **`cluster_variants/`** — Cluster-state drivers with alternative noise models
  (heterogeneous per-gate error, fixed-rate error) used for the robustness
  experiments.

## Caveats

Some scripts hard-code cluster paths (for example `/scratch/...`) or
machine-specific settings; adjust before running. Scripts are provided as-is.
