#!/usr/bin/env python
"""In-silico perturbation probe (masking-based counterfactual).

This is NOT true perturbation prediction. There is no experimental perturbation
data for this project, so nothing here is validated against ground truth.

What it does: take real adipocyte cells, mask a target gene so the model must
predict it, record that prediction, then re-run the SAME cell with a candidate
regulator's input expression forced to 0 ("knocked out") and record how the
target's prediction moves. Same cell, same masked target, one input changed.

Interpretation limit: the model only ever saw observational data, so a large
effect means "this regulator is highly informative for predicting this target
in the model's learned joint distribution", NOT "this regulator causally drives
this target". A background of randomly chosen perturbed genes gives every effect
a percentile, so numbers are comparable to the attention/correlation percentiles
produced by check_tf_edges.py and check_coexpression_shift.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

MODEL_DIR = Path("diabetes_scGPT_mlm_objective/save/dev_diabetes_mlm_objective_claude_pearsonhvg_1-Oct26-22-11")
MODEL_FILE = MODEL_DIR / "model_e20.pt"
GRN_DIR = Path("diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05")
SCGPT_DIR = Path("scGPT_human")

# the TFs we perturb, and the genes we read predictions for
MARKERS = ["PPARG", "CEBPA", "CEBPB", "CEBPD", "SREBF1", "FABP4", "CD36", "PLIN1", "LPL", "ADIPOQ"]

N_CELLS_PER_CONDITION = 128
N_BACKGROUND_GENES = 100
BATCH_SIZE = 16
SEED = 42

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_bins = 51
mask_value = -1
pad_value = -2

set_seed(SEED)
rng = np.random.default_rng(SEED)

save_dir = Path(f"diabetes_scGPT_mlm_objective/save/perturb-probe-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
logger.info(f"save to {save_dir}")

# ---------- data: same filtering as grn_mlm_adipocyte_claude.py ----------
logger.info("Loading data")
adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
adata.obs = adata.obs.rename({"type": "condition", "celltype_atac": "cell_type"}, axis=1)
adata = adata[
    (adata.obs["cell_type"].isin(["Adipocyte"]))
    & adata.obs["condition"].isin(["No_Diabetes", "T1D", "T2D"])
]
adata.var["gene_name"] = adata.var["gene_id"]
adata.var_names = adata.var.gene_id

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
adata = adata[
    (adata.obs["pct_counts_mt"] < 10)
    & (adata.obs["total_counts"] > 1500)
    & (adata.obs["n_genes_by_counts"] > 800)
].copy()
logger.info(f"After QC: {adata.n_obs} cells")

ptprc_counts = adata.X[:, adata.var_names.get_loc("PTPRC")]
if issparse(adata.X):
    ptprc_counts = ptprc_counts.toarray().flatten()
adata = adata[ptprc_counts == 0].copy()
logger.info(f"After PTPRC purity gate: {adata.n_obs} cells")

with open(GRN_DIR / "gene_panel.json") as f:
    gene_panel = json.load(f)
gene_panel = [g for g in gene_panel if g in set(adata.var["gene_name"])]
adata = adata[:, gene_panel].copy()
logger.info(f"Gene panel: {adata.n_vars} genes")

preprocessor = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=True,
    result_log1p_key="X_log1p",
    subset_hvg=False,
    binning=n_bins,
    result_binned_key="X_binned",
)
preprocessor(adata)

# ---------- model ----------
vocab = GeneVocab.from_file(SCGPT_DIR / "vocab.json")
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
vocab.set_default_index(vocab[pad_token])

with open(SCGPT_DIR / "args.json") as f:
    model_configs = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_batch_types = len(set(adata.obs["age"].astype("category").cat.codes.values))

model = TransformerModel(
    len(vocab),
    model_configs["embsize"],
    model_configs["nheads"],
    model_configs["d_hid"],
    model_configs["nlayers"],
    vocab=vocab,
    dropout=0.2,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=True,
    do_dab=False,
    use_batch_labels=False,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=False,
    n_input_bins=n_bins,
    ecs_threshold=0,
    explicit_zero_prob=False,
    use_fast_transformer=True,
    pre_norm=False,
)
state = torch.load(MODEL_FILE, map_location=device)
missing, unexpected = model.load_state_dict(state, strict=False)
logger.info(f"Loaded {MODEL_FILE}; missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    logger.info(f"missing keys (first 10): {missing[:10]}")
if unexpected:
    logger.info(f"unexpected keys (first 10): {unexpected[:10]}")
model.to(device)
model.eval()

# ---------- tokenize ----------
genes = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)
markers_present = [g for g in MARKERS if g in genes]
logger.info(f"Markers present in panel: {markers_present}")

background_pool = [g for g in genes if g not in set(markers_present)]
background_genes = list(rng.choice(background_pool, size=N_BACKGROUND_GENES, replace=False))


def forward_predictions(tok_gene_ids, values):
    """Run the model over all cells in batches, return mlm_output [N, M] on cpu."""
    outs = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for i in range(0, tok_gene_ids.size(0), BATCH_SIZE):
            g = tok_gene_ids[i : i + BATCH_SIZE].to(device)
            v = values[i : i + BATCH_SIZE].to(device)
            out = model(
                g,
                v.float(),
                src_key_padding_mask=g.eq(vocab[pad_token]),
                batch_labels=None,
                MVC=False,
                ECS=False,
            )
            outs.append(out["mlm_output"].float().cpu())
    return torch.cat(outs, dim=0)


results = {}
for condition in ["No_Diabetes", "T1D", "T2D"]:
    sub = adata[adata.obs["condition"] == condition]
    n_take = min(N_CELLS_PER_CONDITION, sub.n_obs)
    take = rng.choice(sub.n_obs, size=n_take, replace=False)
    sub = sub[take].copy()

    counts = (
        sub.layers["X_binned"].toarray()
        if issparse(sub.layers["X_binned"])
        else sub.layers["X_binned"]
    )
    tok = tokenize_and_pad_batch(
        counts,
        gene_ids,
        max_len=len(genes) + 1,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    tok_gene_ids, tok_values = tok["genes"], tok["values"]
    logger.info(f"[{condition}] {n_take} cells, tokenized to {tuple(tok_values.shape)}")

    # map each gene of interest to its column in the tokenized sequence
    vocab_ids = tok_gene_ids[0].tolist()
    id_to_col = {gid: c for c, gid in enumerate(vocab_ids)}
    col_of = {g: id_to_col[vocab[g]] for g in markers_present + background_genes if vocab[g] in id_to_col}

    deltas = {}  # perturbed gene -> {measured gene -> mean delta}
    for n_done, pert_gene in enumerate(markers_present + background_genes, start=1):
        if pert_gene not in col_of:
            continue
        measured = [m for m in markers_present if m != pert_gene]
        measured_cols = [col_of[m] for m in measured]

        v_base = tok_values.clone()
        v_base[:, measured_cols] = mask_value  # model must predict these
        v_pert = v_base.clone()
        v_pert[:, col_of[pert_gene]] = 0  # "knock out" the regulator

        pred_base = forward_predictions(tok_gene_ids, v_base)
        pred_pert = forward_predictions(tok_gene_ids, v_pert)

        deltas[pert_gene] = {
            m: float((pred_pert[:, col_of[m]] - pred_base[:, col_of[m]]).mean())
            for m in measured
        }
        if n_done % 20 == 0:
            logger.info(f"[{condition}] {n_done}/{len(markers_present) + len(background_genes)} perturbations done")

    results[condition] = deltas

    # ---------- report ----------
    logger.info(f"\n=== [{condition}] effect of zeroing each TF, vs. background of {len(background_genes)} random genes ===")
    for pert_gene in markers_present:
        if pert_gene not in deltas:
            continue
        line = []
        for m, d in deltas[pert_gene].items():
            bg = [abs(deltas[r][m]) for r in background_genes if r in deltas and m in deltas[r]]
            pct = float(np.mean(np.array(bg) < abs(d))) if bg else float("nan")
            line.append(f"{m}: {d:+.4f} ({pct:.0%})")
        logger.info(f"  zero {pert_gene:<7} -> " + "  ".join(line))

logger.info("\n=== Directional asymmetry: zero A, measure B  vs  zero B, measure A ===")
for condition in results:
    logger.info(f"[{condition}]")
    for a, b in [("PPARG", "LPL"), ("PPARG", "PLIN1"), ("PPARG", "ADIPOQ"),
                 ("PPARG", "FABP4"), ("PPARG", "CD36"), ("PPARG", "CEBPA")]:
        d = results[condition]
        if a in d and b in d[a] and b in d and a in d[b]:
            logger.info(f"  zero {a} -> {b}: {d[a][b]:+.4f}   |   zero {b} -> {a}: {d[b][a]:+.4f}")

with open(save_dir / "perturbation_deltas.json", "w") as f:
    json.dump(results, f, indent=2)
with open(save_dir / "background_genes.json", "w") as f:
    json.dump(background_genes, f)
logger.info(f"\nSaved to {save_dir}")
logger.info("Done.")
