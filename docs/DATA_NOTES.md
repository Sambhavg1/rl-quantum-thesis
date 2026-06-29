# Data notes

## Scope of `results/`

`results/` contains the curated per-configuration outputs reported in the
thesis for the AME configurations that exist, where the agent converges to the
maximum minimum-bipartition entropy. For these runs the success condition
terminates the episode at the AME state, so the saved state is the reported
state.

## Held for a verified release

For non-existent targets (AME(4,2), AME(7,2)) the success condition never
fires, so an episode runs to its step limit and the saved end-of-episode state
is not necessarily the highest-entanglement state seen during that episode. The
peak value and the saved state must therefore be reconciled before these
plateau results are published. `code/replay_analyse.py` reconstructs a state
from its logged gate sequence and recomputes every balanced-bipartition
entropy, which is the check used for that reconciliation. The non-stabilizer
AME(4,6) case is held on the same basis (the gate set used must match the
reported claim).

## Verification convention

Treat any logged sequence as a record of which gates were applied, not as a
certified state. Reconstruct and recompute before reporting:

```bash
python code/replay_analyse.py --n <n> --d <d> --seq "<gate-sequence>"
```
