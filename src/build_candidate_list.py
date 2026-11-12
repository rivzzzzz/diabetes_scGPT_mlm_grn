#!/usr/bin/env python
"""Build a candidate edge list for potential wet-lab follow-up.

Part 1 (saved matrices): for each candidate gene, rank its partners by how much
their attention AND their real correlation shift between disease and healthy.
Edges high in BOTH are the dual-supported candidates. Candidates = the top-20
attention-shift genes from the GRN run (known to be a noisy list), plus the 10
literature markers as positive controls.

Part 2 (needs h5ad): CALCOCO1/CALCOCO2 are NOT in the model's gene panel, so no
attention exists for them. Correlation can still be computed directly from
expression, so this does that separately and reports their top shifted partners.

NOTE on indexing: the attention matrices are indexed by GRN_DIR/gene_panel.json
(2194 entries, includes <cls>); the correlation matrices by
GRN_DIR/coexpression/gene_panel.json (2193, genes only). Everything below maps by
gene NAME, never by shared index.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

GRN_DIR = Path("diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05")
COEXP_DIR = GRN_DIR / "coexpression"
T1D_DE = "64_donor_T1D_edgeR_all_genes_all_celltypes.tsv"
T2D_DE = "64_donor_T2D_edgeR_all_genes_all_celltypes.tsv"

MARKERS = ["PPARG", "CEBPA", "CEBPB", "CEBPD", "SREBF1", "FABP4", "CD36", "PLIN1", "LPL", "ADIPOQ"]
TOP20_T1D = ["DIRC3", "LRCH1", "EPB41L4B", "RNF149", "DHRS3", "FGF13", "ANKRD30A", "PRKD1",
             "AKT3", "NWD2", "NCKAP5", "MLLT3", "LINC02615", "CRYBG1", "G0S2", "DEPP1",
             "PLEKHG1", "EPB41", "SEMA6A", "DOCK3"]
TOP20_T2D = ["RNF149", "LRCH1", "DEPP1", "DHRS3", "DIRC3", "LDHA", "G0S2", "ME1", "PLS3",
             "ODC1", "FGF13", "NCKAP5", "EPB41L4B", "MLLT3", "COL4A3", "MRC1", "PPP1R14A",
             "SMAD3", "CRYBG1", "AHNAK"]
LAB_INTEREST = ["CALCOCO1", "CALCOCO2", "CALCOCO3"]

# partial, hand-curated -- NOT a TF census. Only used to flag obvious TFs in output.
# A proper systematic scan should use a real TF list (e.g. Lambert et al. 2018).
KNOWN_TFS = {
    "PPARG", "CEBPA", "CEBPB", "CEBPD", "SREBF1", "SMAD3", "MLLT3", "KLF4", "KLF5",
    "EGR1", "FOSB", "JUN", "JUNB", "FOS", "STAT3", "NR4A1", "ESRRA", "NFIA", "TCF21",
    "RXRA", "FOXO1", "MEF2C", "GATA2", "GATA3", "ZEB1", "TWIST1", "HIF1A", "ATF3",
}

TOP_N_PARTNERS = 12

candidates = sorted(set(TOP20_T1D) | set(TOP20_T2D))
print(f"Candidate genes from top-20 lists: {len(candidates)}")
print(f"Markers (positive controls): {len(MARKERS)}")

# ---------- load saved matrices ----------
with open(GRN_DIR / "gene_panel.json") as f:
    attn_panel = json.load(f)
with open(COEXP_DIR / "gene_panel.json") as f:
    corr_panel = json.load(f)
attn_idx = {g: i for i, g in enumerate(attn_panel)}
corr_idx = {g: i for i, g in enumerate(corr_panel)}
print(f"attention panel: {len(attn_panel)}   correlation panel: {len(corr_panel)}")

attn_diff = np.load(GRN_DIR / "difference_matrices.npz")
corr_diff = np.load(COEXP_DIR / "correlation_difference_matrices.npz")
corr_raw = np.load(COEXP_DIR / "correlation_matrices.npz")

# ---------- DE annotation ----------
def load_de(path):
    df = pd.read_csv(path, sep="\t")
    return df[df["celltype"] == "Adipocyte"].set_index("gene")

de = {"T1D": load_de(T1D_DE), "T2D": load_de(T2D_DE)}


def de_tag(gene):
    bits = []
    for cond in ("T1D", "T2D"):
        d = de[cond]
        if gene in d.index:
            row = d.loc[gene]
            if str(row["diffexpressed"]) != "No":
                bits.append(f"{cond}:{row['diffexpressed']}(FDR={row['FDR']:.1e})")
    return ", ".join(bits) if bits else "not DE"


def offdiag(mat):
    return mat[~np.eye(mat.shape[0], dtype=bool)]


# pre-sort |background| once so percentiles are O(log n) lookups, not O(n) scans
attn_bg = {k: np.sort(np.abs(offdiag(attn_diff[k]))) for k in attn_diff.files}
corr_bg = {k: np.sort(np.abs(offdiag(corr_diff[k]))) for k in corr_diff.files}


def pct_abs(values, sorted_abs_bg):
    """Percentile of |values| within a pre-sorted |background|. Vectorized."""
    return np.searchsorted(sorted_abs_bg, np.abs(values)) / len(sorted_abs_bg)

print("\n" + "=" * 100)
print("PART 1 -- candidate edges: partners ranked by DUAL support (attention shift AND correlation shift)")
print("=" * 100)

dual_hits = []
for gene in candidates + MARKERS:
    if gene not in attn_idx or gene not in corr_idx:
        print(f"\n### {gene}: not in both panels (attention={gene in attn_idx}, correlation={gene in corr_idx}) -- skipped")
        continue

    ai, ci = attn_idx[gene], corr_idx[gene]
    partners = [p for p in corr_panel if p != gene and p in attn_idx]
    aj = np.array([attn_idx[p] for p in partners])
    cj = np.array([corr_idx[p] for p in partners])

    # symmetric summary: stronger of the two attention directions, vectorized over partners
    a_t1d = np.maximum(np.abs(attn_diff["diff_T1D"][ai, aj]), np.abs(attn_diff["diff_T1D"][aj, ai]))
    a_t2d = np.maximum(np.abs(attn_diff["diff_T2D"][ai, aj]), np.abs(attn_diff["diff_T2D"][aj, ai]))

    df = pd.DataFrame({
        "partner": partners,
        "attn_pct_T1D": pct_abs(a_t1d, attn_bg["diff_T1D"]),
        "attn_pct_T2D": pct_abs(a_t2d, attn_bg["diff_T2D"]),
        "corr_pct_T1D": pct_abs(corr_diff["diff_T1D"][ci, cj], corr_bg["diff_T1D"]),
        "corr_pct_T2D": pct_abs(corr_diff["diff_T2D"][ci, cj], corr_bg["diff_T2D"]),
        "corr_healthy": corr_raw["No_Diabetes"][ci, cj],
        "corr_T1D": corr_raw["T1D"][ci, cj],
        "corr_T2D": corr_raw["T2D"][ci, cj],
    })
    # dual score = min of the two methods' best-condition percentile (both must be high)
    df["attn_best"] = df[["attn_pct_T1D", "attn_pct_T2D"]].max(axis=1)
    df["corr_best"] = df[["corr_pct_T1D", "corr_pct_T2D"]].max(axis=1)
    df["dual"] = df[["attn_best", "corr_best"]].min(axis=1)
    df = df.sort_values("dual", ascending=False).head(TOP_N_PARTNERS)

    tag = "TF" if gene in KNOWN_TFS else "  "
    role = "MARKER" if gene in MARKERS else "candidate"
    print(f"\n### {gene} [{tag}] ({role}) -- {de_tag(gene)}")
    print(f"{'partner':<12} {'TF':<3} {'dual':>6} {'attn%':>7} {'corr%':>7} {'r healthy':>10} {'r T1D':>7} {'r T2D':>7}  DE")
    for _, r in df.iterrows():
        ptf = "TF" if r["partner"] in KNOWN_TFS else ""
        print(f"{r['partner']:<12} {ptf:<3} {r['dual']:>6.0%} {r['attn_best']:>7.0%} {r['corr_best']:>7.0%} "
              f"{r['corr_healthy']:>10.2f} {r['corr_T1D']:>7.2f} {r['corr_T2D']:>7.2f}  {de_tag(r['partner'])}")
        if r["dual"] >= 0.95:
            dual_hits.append({
                "gene": gene, "partner": r["partner"], "dual": r["dual"],
                "gene_is_tf": gene in KNOWN_TFS, "partner_is_tf": r["partner"] in KNOWN_TFS,
                "gene_is_marker": gene in MARKERS, "partner_is_marker": r["partner"] in MARKERS,
            })

print("\n" + "=" * 100)
print(f"DUAL-SUPPORTED EDGES (both methods >=95th percentile): {len(dual_hits)}")
print("=" * 100)
hits = pd.DataFrame(dual_hits)
if len(hits):
    hits["novel_ish"] = ~(hits["gene_is_marker"] & hits["partner_is_marker"])
    hits = hits.sort_values("dual", ascending=False)
    for _, r in hits.iterrows():
        flag = "known-marker pair" if not r["novel_ish"] else "NOVEL-ish"
        print(f"  {r['gene']:<12} -- {r['partner']:<12} dual={r['dual']:.0%}  [{flag}]")
    hits.to_csv(GRN_DIR / "candidate_dual_supported_edges.csv", index=False)
    print(f"\nSaved to {GRN_DIR / 'candidate_dual_supported_edges.csv'}")

# ---------- PART 2: CALCOCO (correlation only -- not in model panel) ----------
print("\n" + "=" * 100)
print("PART 2 -- CALCOCO (lab interest): correlation only, no attention (not in model gene panel)")
print("=" * 100)

adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
adata.obs = adata.obs.rename({"type": "condition", "celltype_atac": "cell_type"}, axis=1)
adata = adata[(adata.obs["cell_type"].isin(["Adipocyte"]))
              & adata.obs["condition"].isin(["No_Diabetes", "T1D", "T2D"])]
adata.var["gene_name"] = adata.var["gene_id"]
adata.var_names = adata.var.gene_id
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
adata = adata[(adata.obs["pct_counts_mt"] < 10) & (adata.obs["total_counts"] > 1500)
              & (adata.obs["n_genes_by_counts"] > 800)].copy()
ptprc = adata.X[:, adata.var_names.get_loc("PTPRC")]
if issparse(adata.X):
    ptprc = ptprc.toarray().flatten()
adata = adata[ptprc == 0].copy()
print(f"Cells after same filtering as GRN run: {adata.n_obs}")

present_lab = [g for g in LAB_INTEREST if g in set(adata.var["gene_name"])]
print(f"CALCOCO genes present in data: {present_lab}")
keep = [g for g in corr_panel if g in set(adata.var["gene_name"])] + [g for g in present_lab if g not in corr_panel]
adata = adata[:, keep].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
gidx = {g: i for i, g in enumerate(keep)}

cond_corr = {}
for cond in ["No_Diabetes", "T1D", "T2D"]:
    sub = adata[adata.obs["condition"] == cond]
    X = sub.X.toarray() if issparse(sub.X) else np.asarray(sub.X)
    cond_corr[cond] = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
    print(f"{cond}: {sub.n_obs} cells")

for g in present_lab:
    gi = gidx[g]
    d_t1d = cond_corr["T1D"][gi] - cond_corr["No_Diabetes"][gi]
    d_t2d = cond_corr["T2D"][gi] - cond_corr["No_Diabetes"][gi]
    score = np.abs(d_t1d) + np.abs(d_t2d)
    score[gi] = -np.inf
    top = np.argsort(-score)[:TOP_N_PARTNERS]
    print(f"\n### {g} -- {de_tag(g)} -- top partners by correlation SHIFT")
    print(f"{'partner':<14} {'TF':<3} {'r healthy':>10} {'r T1D':>7} {'r T2D':>7} {'dT1D':>7} {'dT2D':>7}  DE")
    for j in top:
        p = keep[j]
        ptf = "TF" if p in KNOWN_TFS else ""
        print(f"{p:<14} {ptf:<3} {cond_corr['No_Diabetes'][gi, j]:>10.2f} {cond_corr['T1D'][gi, j]:>7.2f} "
              f"{cond_corr['T2D'][gi, j]:>7.2f} {d_t1d[j]:>+7.2f} {d_t2d[j]:>+7.2f}  {de_tag(p)}")

print("\nDone.")
