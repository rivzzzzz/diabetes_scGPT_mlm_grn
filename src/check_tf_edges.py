#!/usr/bin/env python
"""Edge-level check: for each known adipocyte TF-target edge (from literature),
look up the actual attention value between the TF and target gene in the saved
GRN attention/difference matrices -- not just the gene-level aggregate shift score
computed in grn_mlm_adipocyte_claude.py. Mirrors the pairwise-edge inspection done
in the scGPT Attention_GRN tutorial. No GPU/model needed, just reads saved arrays.
"""
import json
import numpy as np

GRN_DIR = "diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05"

# (TF, target, source/confidence note)
EDGES = [
    ("CEBPD", "PPARG", "activation; TRRUST PMID 10649448/18619497/20971808"),
    ("PPARG", "CEBPA", "reciprocal cross-activation; Lefterova et al. 2008"),
    ("CEBPA", "PPARG", "reciprocal cross-activation; Lefterova et al. 2008"),
    ("PPARG", "FABP4", "direct PPRE target; Lefterova et al. 2008 / classic"),
    ("PPARG", "CD36", "direct PPRE target; Lefterova et al. 2008"),
    ("PPARG", "PLIN1", "PPARG target; Lefterova et al. 2008"),
    ("PPARG", "LPL", "PPARG target; Lefterova et al. 2008"),
    ("PPARG", "ADIPOQ", "commonly cited, flagged lower-confidence/possibly indirect"),
]

with open(f"{GRN_DIR}/gene_panel.json") as f:
    gene_panel = json.load(f)
gene_to_idx = {g: i for i, g in enumerate(gene_panel)}
M = len(gene_panel)
print(f"Panel size: {M} genes")

attn = np.load(f"{GRN_DIR}/attention_matrices.npz")
diff = np.load(f"{GRN_DIR}/difference_matrices.npz")
conditions = list(attn.keys())
print(f"Conditions available: {conditions}")
print(f"Difference matrices available: {list(diff.keys())}")


def offdiag_values(mat):
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


def percentile_of(value, background):
    return float((background < value).mean())


def abs_percentile_of(value, background):
    return float((np.abs(background) < abs(value)).mean())


print("\n=== Raw attention value + percentile vs background (all off-diagonal pairs), per condition ===")
bg_by_cond = {c: offdiag_values(attn[c]) for c in conditions}

for tf, target, note in EDGES:
    if tf not in gene_to_idx or target not in gene_to_idx:
        print(f"\n{tf} -> {target}: SKIPPED (not in panel)")
        continue
    i, j = gene_to_idx[tf], gene_to_idx[target]
    print(f"\n{tf} -> {target}  [{note}]")
    for c in conditions:
        v_fwd = float(attn[c][i, j])  # tf row (query), target col (key)
        v_rev = float(attn[c][j, i])  # target row, tf col
        p_fwd = percentile_of(v_fwd, bg_by_cond[c])
        p_rev = percentile_of(v_rev, bg_by_cond[c])
        print(
            f"  [{c:12s}] attn[{tf}->{target}]={v_fwd:.4f} (percentile {p_fwd:6.1%})   "
            f"attn[{target}->{tf}]={v_rev:.4f} (percentile {p_rev:6.1%})"
        )

print("\n=== Difference-matrix value (condition vs No_Diabetes), |value| percentile vs background ===")
bg_diff = {d: offdiag_values(diff[d]) for d in diff.keys()}

for tf, target, note in EDGES:
    if tf not in gene_to_idx or target not in gene_to_idx:
        continue
    i, j = gene_to_idx[tf], gene_to_idx[target]
    print(f"\n{tf} -> {target}  [{note}]")
    for d in diff.keys():
        v_fwd = float(diff[d][i, j])
        v_rev = float(diff[d][j, i])
        p_fwd = abs_percentile_of(v_fwd, bg_diff[d])
        p_rev = abs_percentile_of(v_rev, bg_diff[d])
        print(
            f"  [{d:10s}] diff[{tf}->{target}]={v_fwd:+.4f} (|.| percentile {p_fwd:6.1%})   "
            f"diff[{target}->{tf}]={v_rev:+.4f} (|.| percentile {p_rev:6.1%})"
        )

print("\nDone.")
