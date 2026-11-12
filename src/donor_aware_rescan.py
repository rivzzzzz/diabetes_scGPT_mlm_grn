#!/usr/bin/env python
"""Donor-aware rescan -- the statistically correct version of the earlier analysis.

Earlier work pooled all cells of a condition and correlated across the pool. That
lets between-donor variation in mean expression manufacture correlation structure,
which is what produced the (now retracted) "disease shift" finding.

Here: correlation is computed WITHIN each donor, so each donor is one observation,
and conditions are compared with a Welch t-test across donors, BH-FDR corrected
across every gene pair in the panel.

Contrasts run separately (the earlier test pooled T1D+T2D, which the consult
flagged): T1D vs healthy, T2D vs healthy, and pooled diabetic vs healthy.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.sparse import issparse

GRN_DIR = Path("diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05")
MIN_CELLS = 30
FDR_ALPHA = 0.05

SAVE_DIR = Path("diabetes_scGPT_mlm_objective/save") / time.strftime("donor-rescan-%b%d-%H-%M")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Mirror stdout into run.log so this result survives the pod."""

    def __init__(self, path):
        self.log = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.log.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.log.flush()


sys.stdout = _Tee(SAVE_DIR / "run.log")
print(f"saving to {SAVE_DIR}")

FOCUS = [("ZEB1", "PPARG"), ("ZEB1", "CEBPB"), ("PPARG", "LPL"), ("PPARG", "PLIN1"),
         ("PPARG", "ADIPOQ"), ("PPARG", "FABP4"), ("PPARG", "CD36"), ("MLXIPL", "FASN")]

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
adata.obs["donor"] = adata.obs["orig.ident"].astype(str)

with open(GRN_DIR / "gene_panel.json") as f:
    panel = [g for g in json.load(f) if g in set(adata.var["gene_name"])]
adata = adata[:, panel].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
gidx = {g: i for i, g in enumerate(panel)}
G = len(panel)
print(f"{adata.n_obs} cells, {G} genes, {adata.obs['donor'].nunique()} donors")

# ---------- per-donor correlation matrices ----------
donor_ids, donor_cond, donor_n = [], [], []
counts = adata.obs["donor"].value_counts()
keep = [d for d in counts.index if counts[d] >= MIN_CELLS]
print(f"Donors with >= {MIN_CELLS} adipocytes: {len(keep)}")

corrs = np.empty((len(keep), G, G), dtype=np.float32)
for k, d in enumerate(keep):
    m = (adata.obs["donor"] == d).values
    c = np.corrcoef(X[m].T)
    corrs[k] = np.nan_to_num(c, nan=0.0).astype(np.float32)
    donor_ids.append(d)
    donor_cond.append(str(adata.obs.loc[m, "condition"].iloc[0]))
    donor_n.append(int(m.sum()))

donor_cond = np.array(donor_cond)
print(pd.Series(donor_cond).value_counts().to_string())
print(f"cells per donor: median={np.median(donor_n):.0f} min={min(donor_n)} max={max(donor_n)}")

iu = np.triu_indices(G, k=1)
n_edges = len(iu[0])
print(f"edges tested: {n_edges:,}")


def bh_fdr(p):
    n = len(p)
    o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(r, 0, 1)
    return out


results = {}
for label, grp in [("T1D vs healthy", "T1D"), ("T2D vs healthy", "T2D"), ("diabetic(pooled) vs healthy", "BOTH")]:
    a = corrs[donor_cond == "No_Diabetes"]
    b = corrs[np.isin(donor_cond, ["T1D", "T2D"])] if grp == "BOTH" else corrs[donor_cond == grp]
    na, nb = len(a), len(b)

    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    se = np.sqrt(va / na + vb / nb)
    se[se < 1e-12] = np.nan
    t = (mb - ma) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * stats.t.sf(np.abs(t), df)

    pe = np.nan_to_num(p[iu], nan=1.0)
    q = bh_fdr(pe)
    n_sig = int((q < FDR_ALPHA).sum())
    print(f"\n{'=' * 90}\n{label}   (healthy n={na}, other n={nb})\n{'=' * 90}")
    print(f"edges passing FDR<{FDR_ALPHA}: {n_sig:,} / {n_edges:,}")
    print(f"smallest raw p: {np.nanmin(pe):.3g}   smallest q: {np.nanmin(q):.3g}")

    if n_sig:
        order = np.argsort(q)[:20]
        print(f"\n{'edge':<28} {'r healthy':>10} {'r other':>9} {'diff':>8} {'q':>10}")
        for o in order:
            i, j = iu[0][o], iu[1][o]
            print(f"{panel[i] + '-' + panel[j]:<28} {ma[i, j]:>10.3f} {mb[i, j]:>9.3f} "
                  f"{mb[i, j] - ma[i, j]:>+8.3f} {q[o]:>10.3g}")

    # where our claimed findings land in THIS (correct) framework
    print(f"\n{'focus pair':<22} {'r healthy':>10} {'r other':>9} {'diff':>8} {'raw p':>10} {'q':>10}")
    # q is aligned to triu_indices order; index focus pairs directly by index math
    pos = {}
    for ga, gb in FOCUS:
        if ga in gidx and gb in gidx:
            i, j = sorted((gidx[ga], gidx[gb]))
            # index of (i,j) within triu_indices ordering
            k = int(i * G - (i * (i + 1)) // 2 + (j - i - 1))
            pos[(ga, gb)] = k
    focus_out = {}
    for (ga, gb), k in pos.items():
        i, j = sorted((gidx[ga], gidx[gb]))
        print(f"{ga + '-' + gb:<22} {ma[i, j]:>10.3f} {mb[i, j]:>9.3f} "
              f"{mb[i, j] - ma[i, j]:>+8.3f} {pe[k]:>10.3g} {q[k]:>10.3g}")
        focus_out[f"{ga}-{gb}"] = {"r_healthy": float(ma[i, j]), "r_other": float(mb[i, j]),
                                   "diff": float(mb[i, j] - ma[i, j]),
                                   "raw_p": float(pe[k]), "q": float(q[k])}

    results[label] = {"n_healthy": na, "n_other": nb, "n_edges": n_edges,
                      "n_sig": n_sig, "min_raw_p": float(np.nanmin(pe)),
                      "min_q": float(np.nanmin(q)), "focus_pairs": focus_out}

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
for k, v in results.items():
    print(f"  {k:<32} edges surviving FDR<0.05: {v['n_sig']:,}   (min q = {v['min_q']:.3g})")

with open(SAVE_DIR / "donor_rescan_results.json", "w") as f:
    json.dump({"min_cells_per_donor": MIN_CELLS, "fdr_alpha": FDR_ALPHA,
               "n_cells": int(adata.n_obs), "n_genes": G,
               "donors_used": len(keep),
               "donors_per_condition": pd.Series(donor_cond).value_counts().to_dict(),
               "contrasts": results}, f, indent=2)
print(f"\nSaved to {SAVE_DIR}")
print("Done.")
