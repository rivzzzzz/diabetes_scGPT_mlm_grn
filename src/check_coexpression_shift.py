#!/usr/bin/env python
"""Step 2: does the attention-shift finding (PPARG -> LPL/PLIN1/ADIPOQ) correspond
to an actual co-expression/correlation shift between conditions, or is it specific
to the attention mechanism? Computes real gene-gene Pearson correlation matrices
per condition, on the exact same cells/genes used in the GRN attention run, and
runs the same edge-lookup + percentile methodology as check_tf_edges.py so the
two are directly comparable.
"""
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

GRN_DIR = "diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05"

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

print("=== Loading data (same filtering as grn_mlm_adipocyte_claude.py) ===")
adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
adata.obs = adata.obs.rename({'type': 'condition', 'celltype_atac': 'cell_type'}, axis=1)
adata = adata[((adata.obs['cell_type'].isin(['Adipocyte'])) & adata.obs['condition'].isin(['No_Diabetes', 'T1D', 'T2D']))]

adata.var['gene_name'] = adata.var['gene_id']
adata.var_names = adata.var.gene_id

adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[
    (adata.obs["pct_counts_mt"] < 10) &
    (adata.obs["total_counts"] > 1500) &
    (adata.obs["n_genes_by_counts"] > 800)
].copy()
print(f"After QC: {adata.n_obs} cells")

ptprc_idx = adata.var_names.get_loc('PTPRC')
ptprc_counts = adata.X[:, ptprc_idx]
if issparse(adata.X):
    ptprc_counts = ptprc_counts.toarray().flatten()
adata = adata[ptprc_counts == 0].copy()
print(f"After PTPRC purity gate: {adata.n_obs} cells")
print(adata.obs['condition'].value_counts().to_string())

print("\n=== Restricting to the exact gene panel used in GRN extraction ===")
with open(f"{GRN_DIR}/gene_panel.json") as f:
    gene_panel = json.load(f)
gene_panel_present = [g for g in gene_panel if g in set(adata.var["gene_name"])]
print(f"Gene panel: {len(gene_panel_present)} / {len(gene_panel)} genes present")
adata = adata[:, gene_panel_present].copy()

print("\n=== Normalizing (same as GRN script, minus binning -- correlation needs continuous values) ===")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

gene_to_idx = {g: i for i, g in enumerate(gene_panel_present)}


def corr_matrix_for_condition(adata, cond):
    sub = adata[adata.obs["condition"] == cond]
    X = sub.X.toarray() if issparse(sub.X) else np.asarray(sub.X)
    # genes x cells for np.corrcoef
    C = np.corrcoef(X.T)
    C = np.nan_to_num(C, nan=0.0)
    return C, sub.n_obs


print("\n=== Computing gene-gene correlation matrices per condition ===")
conditions = ["No_Diabetes", "T1D", "T2D"]
corr = {}
ncells = {}
for c in conditions:
    corr[c], ncells[c] = corr_matrix_for_condition(adata, c)
    print(f"{c}: {ncells[c]} cells, correlation matrix {corr[c].shape}")

diff = {
    "diff_T1D": corr["T1D"] - corr["No_Diabetes"],
    "diff_T2D": corr["T2D"] - corr["No_Diabetes"],
}


def offdiag_values(mat):
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return mat[mask]


def percentile_of(value, background):
    return float((background < value).mean())


def abs_percentile_of(value, background):
    return float((np.abs(background) < abs(value)).mean())


print("\n=== Raw correlation + percentile vs background, per condition ===")
bg_by_cond = {c: offdiag_values(corr[c]) for c in conditions}

for tf, target, note in EDGES:
    if tf not in gene_to_idx or target not in gene_to_idx:
        print(f"\n{tf} -> {target}: SKIPPED (not in panel)")
        continue
    i, j = gene_to_idx[tf], gene_to_idx[target]
    print(f"\n{tf} <-> {target}  [{note}]")
    for c in conditions:
        v = float(corr[c][i, j])
        p = percentile_of(v, bg_by_cond[c])
        print(f"  [{c:12s}] corr = {v:+.4f}  (percentile {p:6.1%})")

print("\n=== Difference in correlation (condition vs No_Diabetes), |value| percentile vs background ===")
bg_diff = {d: offdiag_values(diff[d]) for d in diff.keys()}

for tf, target, note in EDGES:
    if tf not in gene_to_idx or target not in gene_to_idx:
        continue
    i, j = gene_to_idx[tf], gene_to_idx[target]
    print(f"\n{tf} <-> {target}  [{note}]")
    for d in diff.keys():
        v = float(diff[d][i, j])
        p = abs_percentile_of(v, bg_diff[d])
        print(f"  [{d:10s}] diff corr = {v:+.4f}  (|.| percentile {p:6.1%})")

print("\n=== Saving matrices for later reuse ===")
import os
out_dir = f"{GRN_DIR}/coexpression"
os.makedirs(out_dir, exist_ok=True)
np.savez(f"{out_dir}/correlation_matrices.npz", **corr)
np.savez(f"{out_dir}/correlation_difference_matrices.npz", **diff)
with open(f"{out_dir}/gene_panel.json", "w") as f:
    json.dump(gene_panel_present, f)
print(f"Saved to {out_dir}")

print("\nDone.")
