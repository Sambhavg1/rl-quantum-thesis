#!/usr/bin/env python3
# =====================================================================
# replay_analyse.py  --  deterministic replay + bipartition analysis
#
# Purpose
# -------
# Given a logged hyphen-separated gate sequence (as written to the
# results CSVs by ame_ppo.py), rebuild the final state from |0...0>
# and report the von Neumann entropy across EVERY balanced
# bipartition. This is the tool used to verify a reported AME / plateau
# state, independently of the reward that was logged during training.
#
# Why this matters
# ----------------
# The training loop logs an episode-level scalar (max entropy seen).
# To trust any reported state you must reconstruct it from its gate
# sequence and recompute the bipartition entropies directly. If the
# reconstructed H_min disagrees with the logged reward, the logged
# "best" sequence is not the state that produced the headline number
# (e.g. a final-episode vs best-episode logging mismatch). Run this on
# every state you intend to put in the thesis / repo.
#
# Gate-name convention (matches ame_ppo.py): X_i, Z_i, F_i, SUM_i_j
#
# >>> ACTION REQUIRED <<<
# Confirm yourself that this reproduces the bipartition numbers you
# report for AME(4,2), AME(7,2) and AME(4,6). Do not take it on trust.
# =====================================================================

import argparse
import numpy as np
from functools import reduce
from itertools import combinations


def build_single_qudit_gates(d):
    """Single-qudit Clifford generators X, Z, F over local dimension d."""
    omega = np.exp(2j * np.pi / d)
    X = np.zeros((d, d), dtype=np.complex128)
    Z = np.zeros((d, d), dtype=np.complex128)
    F = np.zeros((d, d), dtype=np.complex128)
    for i in range(d):
        X[(i + 1) % d, i] = 1.0
        Z[i, i] = omega ** i
        for k in range(d):
            F[i, k] = omega ** (i * k)
    F /= np.sqrt(d)
    return {'X': X, 'Z': Z, 'F': F}


def apply_single(state, mat, q, n, d):
    """Apply a single-qudit gate `mat` to qudit q of an (d,)*n tensor."""
    state = np.tensordot(mat, state, axes=([1], [q]))
    return np.moveaxis(state, 0, q)


def apply_sum(state, q_ctrl, q_tgt, n, d):
    """Apply SUM (qudit CNOT): |c,t> -> |c, t+c mod d>, in-place style."""
    new = np.zeros_like(state)
    it = np.ndindex(*state.shape)
    for idx in it:
        c = idx[q_ctrl]
        t = idx[q_tgt]
        new_idx = list(idx)
        new_idx[q_tgt] = (t + c) % d
        new[tuple(new_idx)] += state[idx]
    return new


def replay(seq_str, n, d):
    """Rebuild the state from |0...0> by applying the logged sequence."""
    gates = build_single_qudit_gates(d)
    state = np.zeros((d,) * n, dtype=np.complex128)
    state[(0,) * n] = 1.0 + 0j
    for name in seq_str.split('-'):
        head, *rest = name.split('_')
        if head == 'SUM':
            i, j = int(rest[0]), int(rest[1])
            state = apply_sum(state, i, j, n, d)
        else:
            q = int(rest[0])
            state = apply_single(state, gates[head], q, n, d)
        nrm = np.linalg.norm(state.ravel())
        if nrm > 1e-12:
            state /= nrm
    return state


def bipartition_entropies(state, n, d):
    """von Neumann entropy across every size-(n//2) bipartition."""
    k = n // 2
    out = {}
    for keep in combinations(range(n), k):
        trace = tuple(i for i in range(n) if i not in set(keep))
        psi = np.transpose(state, list(keep) + list(trace)).reshape(
            d ** k, d ** (n - k))
        rho = psi @ psi.conj().T
        ev = np.linalg.eigvalsh(rho)
        ev = ev[ev > 1e-12]
        out[keep] = float(-np.sum(ev * np.log(ev)))
    return out


def main():
    p = argparse.ArgumentParser(description="Replay a gate sequence and report bipartition entropies.")
    p.add_argument('--n', type=int, required=True, help='number of qudits')
    p.add_argument('--d', type=int, required=True, help='local dimension')
    p.add_argument('--seq', type=str, required=True,
                   help="hyphen-separated gate sequence, e.g. 'F_0-SUM_0_1-X_2'")
    args = p.parse_args()

    state = replay(args.seq, args.n, args.d)
    bip = bipartition_entropies(state, args.n, args.d)
    h_max = (args.n // 2) * np.log(args.d)
    h_min = min(bip.values())

    print(f"AME({args.n},{args.d})  H_max = {h_max:.6f}  (k*ln d, k = n//2)")
    print(f"min bipartition entropy  H_min = {h_min:.6f}  ({100*h_min/h_max:.2f}% of max)")
    print("-" * 60)
    for keep in sorted(bip):
        flag = "  <-- min" if abs(bip[keep] - h_min) < 1e-9 else ""
        print(f"  keep {keep}: {bip[keep]:.6f}{flag}")


if __name__ == '__main__':
    main()
