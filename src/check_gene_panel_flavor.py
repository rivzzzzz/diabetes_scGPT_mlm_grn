#!/usr/bin/env python
"""Diagnostic (no training): check whether adata.X holds raw counts, and compare
which known adipocyte TF/target genes survive under the current `seurat_v3` HVG
flavor vs. the candidate `pearson_residuals` flavor, on the same cell-filtered
data used by mlm_finetune_adipocyte.py.
"""
import json
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

GENES_OF_INTEREST = [
    "PPARG", "CEBPA", "CEBPB", "CEBPD", "SREBF1",  # TFs
    "FABP4", "CD36", "PLIN1", "LPL", "ADIPOQ",      # targets
]

N_HVG = 3000

print("=== Loading data (same filtering as mlm_finetune_adipocyte.py) ===")
adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
adata.obs = adata.obs.rename({'type': 'condition', 'celltype_atac': 'cell_type'}, axis=1)
adata = adata[((adata.obs['cell_type'].isin(['Adipocyte'])) & adata.obs['condition'].isin(['No_Diabetes', 'T1D', 'T2D']))]

with open("diabetes_scGPT_mlm_objective/data_files/dev_idx_adipocyte.json", "r") as f:
    dev_idx = json.load(f)
adata = adata[dev_idx].copy()

adata.var['gene_name'] = adata.var['gene_id']
adata.var_names = adata.var.gene_id

adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[
    (adata.obs["pct_counts_mt"] < 10) &
    (adata.obs["total_counts"] > 1500) &
    (adata.obs["n_genes_by_counts"] > 800)
].copy()
print(f"Cells after QC: {adata.n_obs}, genes: {adata.n_vars}")

print("\n=== Checking whether adata.X is raw integer counts ===")
X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
is_nonneg = bool((X >= 0).all())
is_integer_valued = bool(np.allclose(X, np.round(X)))
print(f"adata.X dtype: {X.dtype}, min: {X.min():.4f}, max: {X.max():.4f}")
print(f"All non-negative: {is_nonneg}")
print(f"All integer-valued: {is_integer_valued}")
x_is_raw_counts = is_nonneg and is_integer_valued
print(f"=> adata.X appears to be RAW COUNTS: {x_is_raw_counts}")

print(f"\nadata.raw is set: {adata.raw is not None}")
print(f"adata.layers keys: {list(adata.layers.keys())}")

present_genes = [g for g in GENES_OF_INTEREST if g in set(adata.var["gene_name"])]
missing_genes = [g for g in GENES_OF_INTEREST if g not in set(adata.var["gene_name"])]
print(f"\nGenes of interest present in filtered data (pre-HVG): {present_genes}")
print(f"Genes of interest NOT in filtered data at all (check gene_id/symbol mismatch): {missing_genes}")


def hvg_panel(adata_in, flavor, n_top_genes, use_layer=None, check_values=True):
    a = adata_in.copy()
    try:
        if flavor == "pearson_residuals":
            sc.experimental.pp.highly_variable_genes(
                a, n_top_genes=n_top_genes, flavor="pearson_residuals",
                layer=use_layer, check_values=check_values,
            )
        else:
            sc.pp.highly_variable_genes(
                a, n_top_genes=n_top_genes, flavor=flavor, layer=use_layer,
            )
        hvg = set(a.var.loc[a.var["highly_variable"], "gene_name"])
        return hvg, None
    except Exception as e:
        return None, repr(e)


print("\n=== Computing seurat_v3 HVG panel (current approach, requires raw counts) ===")
hvg_seuratv3, err_sv3 = hvg_panel(adata, flavor="seurat_v3", n_top_genes=N_HVG)
if err_sv3:
    print(f"seurat_v3 FAILED: {err_sv3}")
else:
    print(f"seurat_v3 panel size: {len(hvg_seuratv3)}")

print("\n=== Computing pearson_residuals HVG panel (candidate flavor, requires raw counts) ===")
hvg_pearson, err_pr = hvg_panel(adata, flavor="pearson_residuals", n_top_genes=N_HVG)
if err_pr:
    print(f"pearson_residuals FAILED: {err_pr}")
else:
    print(f"pearson_residuals panel size: {len(hvg_pearson)}")

print("\n=== Survival of known TF/target genes ===")
print(f"{'gene':<10} {'in_data':<8} {'seurat_v3':<10} {'pearson_residuals':<18}")
for g in GENES_OF_INTEREST:
    in_data = g in present_genes
    in_sv3 = (g in hvg_seuratv3) if hvg_seuratv3 is not None else "N/A"
    in_pr = (g in hvg_pearson) if hvg_pearson is not None else "N/A"
    print(f"{g:<10} {str(in_data):<8} {str(in_sv3):<10} {str(in_pr):<18}")

print("\nDone.")
