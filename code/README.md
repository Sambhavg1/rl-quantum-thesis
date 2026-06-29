# Code

The two thesis drivers and the verification tool.

## Files

| File | Track | What it does |
|------|-------|--------------|
| `cluster_ppo.py` | 1 | Linear cluster-state preparation on *n* qubits. Per-step reward is the gain in fidelity to the target cluster state. Optional gate-level noise model via `--noisy`. CLI driven; auto-scales network width with *n*. |
| `ame_simple.py` | 2 | Target-free AME discovery on *n* qudits of dimension *d*. Reward is the minimum von Neumann entropy across all balanced bipartitions; no target state is supplied. Clifford gate set (X, Z, F, SUM). Saves CSV, Dirac state, circuit, and plots. |
| `replay_analyse.py` | — | Rebuilds a state from a logged hyphen-separated gate sequence and reports the entropy of every balanced bipartition. Use it to verify any reported state independently of the reward logged during training. |

## Command-line use

```bash
python ame_simple.py  --n 4 --d 3 --episodes 10000 --seed 42 --run_name ame43_s42
python cluster_ppo.py --n 6        --episodes 50000 --seed 42 --run_name cluster6_ideal_s42
python cluster_ppo.py --n 6        --episodes 50000 --seed 42 --noisy --run_name cluster6_noisy_s42
python replay_analyse.py --n 4 --d 2 --seq "F_0-SUM_0_1-X_2"
```

Each run writes its outputs to a per-run directory (CSV trajectory, best
sequence, Dirac state, circuit, summary, and plots).

## Gate-name convention

Single-qudit gates `X_i`, `Z_i`, `F_i` on qudit *i*; two-qudit `SUM_i_j`
(qudit CNOT, control *i*, target *j*). `replay_analyse.py` uses the same
convention, so any `Gate_Sequence` logged by `ame_simple.py` can be replayed
directly.

## Note on the gym / gymnasium API

The drivers subclass `gym.Env` and `step` returns the 4-tuple
`(obs, reward, done, info)`. The pinned `requirements.txt` installs
`gym==0.26.2` so the code runs as committed. To migrate to `gymnasium`,
replace `import gym` with `import gymnasium as gym`, split `done` into
`terminated` / `truncated`, and update `reset` to return `(obs, info)`.
