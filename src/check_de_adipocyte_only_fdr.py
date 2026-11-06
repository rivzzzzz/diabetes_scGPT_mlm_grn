#!/usr/bin/env python
"""Close the FDR-correction caveat from check_de_overlap.py: the edgeR output's
FDR column may have been computed by correcting p-values jointly across all
celltypes (~16x more tests than needed), which would make true Adipocyte-specific
effects look falsely non-significant. Recompute Benjamini-Hochberg FDR using only
the raw PValue column restricted to the Adipocyte subset, and see if any of the
10 markers (or the top-20 attention genes) cross 0.05 under the less conservative,
within-celltype-only correction.
"""
import numpy as np
import pandas as pd

T1D_DE_FILE = "64_donor_T1D_edgeR_all_genes_all_celltypes.tsv"
T2D_DE_FILE = "64_donor_T2D_edgeR_all_genes_all_celltypes.tsv"

MARKERS = ["PPARG", "CEBPA", "CEBPB", "CEBPD", "SREBF1", "FABP4", "CD36", "PLIN1", "LPL", "ADIPOQ"]
TOP20_T1D = [
    "DIRC3", "LRCH1", "EPB41L4B", "RNF149", "DHRS3", "FGF13", "ANKRD30A", "PRKD1",
    "AKT3", "NWD2", "NCKAP5", "MLLT3", "LINC02615", "CRYBG1", "G0S2", "DEPP1",
    "PLEKHG1", "EPB41", "SEMA6A", "DOCK3",
]
TOP20_T2D = [
    "RNF149", "LRCH1", "DEPP1", "DHRS3", "DIRC3", "LDHA", "G0S2", "ME1", "PLS3",
    "ODC1", "FGF13", "NCKAP5", "EPB41L4B", "MLLT3", "COL4A3", "MRC1", "PPP1R14A",
    "SMAD3", "CRYBG1", "AHNAK",
]


def bh_fdr(pvals):
    """Benjamini-Hochberg, matching R's p.adjust(method='BH')."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p-value down
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    fdr = np.empty(n)
    fdr[order] = np.clip(ranked, 0, 1)
    return fdr


def load_and_recorrect(path):
    df = pd.read_csv(path, sep="\t")
    df_ad = df[df["celltype"] == "Adipocyte"].copy()
    df_ad["FDR_adipocyte_only"] = bh_fdr(df_ad["PValue"].values)
    df_ad = df_ad.set_index("gene")
    return df_ad


print("Loading and recorrecting (Adipocyte-only BH) ...")
t1d = load_and_recorrect(T1D_DE_FILE)
t2d = load_and_recorrect(T2D_DE_FILE)
print(f"T1D Adipocyte genes: {len(t1d)}, T2D Adipocyte genes: {len(t2d)}")

n_diff_t1d = (t1d["FDR"].round(6) != t1d["FDR_adipocyte_only"].round(6)).sum()
print(f"\nRows where original FDR != recomputed Adipocyte-only FDR (T1D): {n_diff_t1d} / {len(t1d)}")
print("(0 here means the original FDR was ALREADY computed within-Adipocyte only -- pooled-correction worry was unfounded)")

print("\n=== Markers: original (pooled?) FDR vs. Adipocyte-only-recomputed FDR ===")
print(f"{'gene':<10} {'T1D orig FDR':>13} {'T1D adi-only FDR':>18}   {'T2D orig FDR':>13} {'T2D adi-only FDR':>18}")
for g in MARKERS:
    r1 = t1d.loc[g] if g in t1d.index else None
    r2 = t2d.loc[g] if g in t2d.index else None
    f1o = f"{r1['FDR']:.2e}" if r1 is not None else "n/a"
    f1n = f"{r1['FDR_adipocyte_only']:.2e}" if r1 is not None else "n/a"
    f2o = f"{r2['FDR']:.2e}" if r2 is not None else "n/a"
    f2n = f"{r2['FDR_adipocyte_only']:.2e}" if r2 is not None else "n/a"
    print(f"{g:<10} {f1o:>13} {f1n:>18}   {f2o:>13} {f2n:>18}")

n_newly_sig_t1d = sum(1 for g in MARKERS if g in t1d.index and t1d.loc[g]['FDR'] >= 0.05 and t1d.loc[g]['FDR_adipocyte_only'] < 0.05)
n_newly_sig_t2d = sum(1 for g in MARKERS if g in t2d.index and t2d.loc[g]['FDR'] >= 0.05 and t2d.loc[g]['FDR_adipocyte_only'] < 0.05)
print(f"\nMarkers newly significant (FDR<0.05) ONLY under Adipocyte-only correction: {n_newly_sig_t1d} (T1D), {n_newly_sig_t2d} (T2D)")

print("\n=== Top-20 attention genes: same check ===")
n_newly_sig_top20_t1d = sum(1 for g in TOP20_T1D if g in t1d.index and t1d.loc[g]['FDR'] >= 0.05 and t1d.loc[g]['FDR_adipocyte_only'] < 0.05)
n_newly_sig_top20_t2d = sum(1 for g in TOP20_T2D if g in t2d.index and t2d.loc[g]['FDR'] >= 0.05 and t2d.loc[g]['FDR_adipocyte_only'] < 0.05)
print(f"Top-20 (T1D list) newly significant under Adipocyte-only correction: {n_newly_sig_top20_t1d} / 20")
print(f"Top-20 (T2D list) newly significant under Adipocyte-only correction: {n_newly_sig_top20_t2d} / 20")

print("\nDone.")
