#!/usr/bin/env python
"""Compare the GRN/attention-flagged genes against the existing pseudobulk edgeR
DE results (Adipocyte celltype only), for both T1D and T2D vs. the baseline.

Checks two groups:
  1. The 10 literature-sourced TF/target markers (PPARG, CEBPA, ... ADIPOQ)
  2. The top-20 "most attention-shifted" genes from the gene-level GRN ranking
     (checked against their own condition's DE results)
"""
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


def load_adipocyte(path):
    df = pd.read_csv(path, sep="\t")
    df = df[df["celltype"] == "Adipocyte"].set_index("gene")
    return df


def lookup(df, gene):
    if gene not in df.index:
        return None
    row = df.loc[gene]
    return {
        "logFC": float(row["logFC"]),
        "FDR": float(row["FDR"]),
        "diffexpressed": str(row["diffexpressed"]),
    }


print("Loading DE tables (Adipocyte celltype only) ...")
de_t1d = load_adipocyte(T1D_DE_FILE)
de_t2d = load_adipocyte(T2D_DE_FILE)
print(f"T1D Adipocyte genes tested: {len(de_t1d)}")
print(f"T2D Adipocyte genes tested: {len(de_t2d)}")

print("\n=== 1. Literature-sourced TF/target markers: DE status in Adipocytes ===")
print(f"{'gene':<10} {'T1D logFC':>10} {'T1D FDR':>10} {'T1D':>6}   {'T2D logFC':>10} {'T2D FDR':>10} {'T2D':>6}")
for g in MARKERS:
    r1 = lookup(de_t1d, g)
    r2 = lookup(de_t2d, g)
    f1 = f"{r1['logFC']:>10.3f} {r1['FDR']:>10.2e} {r1['diffexpressed']:>6}" if r1 else f"{'n/a':>10} {'n/a':>10} {'n/a':>6}"
    f2 = f"{r2['logFC']:>10.3f} {r2['FDR']:>10.2e} {r2['diffexpressed']:>6}" if r2 else f"{'n/a':>10} {'n/a':>10} {'n/a':>6}"
    print(f"{g:<10} {f1}   {f2}")

print("\n=== 2. Top-20 attention-shifted genes (T1D list) vs. T1D Adipocyte DE ===")
print(f"{'gene':<12} {'logFC':>10} {'FDR':>10} {'diffexpressed':>14}")
for g in TOP20_T1D:
    r = lookup(de_t1d, g)
    if r:
        print(f"{g:<12} {r['logFC']:>10.3f} {r['FDR']:>10.2e} {r['diffexpressed']:>14}")
    else:
        print(f"{g:<12} {'not tested in Adipocyte DE table':>36}")

print("\n=== 3. Top-20 attention-shifted genes (T2D list) vs. T2D Adipocyte DE ===")
print(f"{'gene':<12} {'logFC':>10} {'FDR':>10} {'diffexpressed':>14}")
for g in TOP20_T2D:
    r = lookup(de_t2d, g)
    if r:
        print(f"{g:<12} {r['logFC']:>10.3f} {r['FDR']:>10.2e} {r['diffexpressed']:>14}")
    else:
        print(f"{g:<12} {'not tested in Adipocyte DE table':>36}")

n_sig_t1d = sum(1 for g in TOP20_T1D if (r := lookup(de_t1d, g)) and r["diffexpressed"] != "No")
n_sig_t2d = sum(1 for g in TOP20_T2D if (r := lookup(de_t2d, g)) and r["diffexpressed"] != "No")
n_sig_markers_t1d = sum(1 for g in MARKERS if (r := lookup(de_t1d, g)) and r["diffexpressed"] != "No")
n_sig_markers_t2d = sum(1 for g in MARKERS if (r := lookup(de_t2d, g)) and r["diffexpressed"] != "No")

print("\n=== Summary ===")
print(f"Markers DE (FDR<0.05) in Adipocytes: {n_sig_markers_t1d}/10 (T1D), {n_sig_markers_t2d}/10 (T2D)")
print(f"Top-20 attention genes DE (FDR<0.05) in Adipocytes: {n_sig_t1d}/20 (T1D), {n_sig_t2d}/20 (T2D)")
print("\nDone.")
