#!/usr/bin/env python
"""Focused profile of candidate TFs that surfaced from the data (MLXIPL, ZEB1, et al.).

Part A (saved matrices): for each TF, rank partners with STRICT support --
  - attention shift high in BOTH conditions (min, not max, of the two percentiles)
  - correlation shift high in BOTH conditions
  - correlation shift the SAME SIGN in both conditions
This is deliberately much tighter than build_candidate_list.py, which used
max-of-either-condition and produced an unusable 353-edge dump.

Part B (GPU): run the masking perturbation probe on these TFs and their top
partners to get directionality -- zero TF, measure partner, vs. the reverse.
Genes with very high mean expression are flagged, since the probe is known to
misbehave on them (zeroing an abundant transcript is off-distribution).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse

import scgpt as scg
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

GRN_DIR = Path("diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05")
COEXP_DIR = GRN_DIR / "coexpression"
MODEL_DIR = Path("diabetes_scGPT_mlm_objective/save/dev_diabetes_mlm_objective_claude_pearsonhvg_1-Oct26-22-11")
MODEL_FILE = MODEL_DIR / "model_e20.pt"
SCGPT_DIR = Path("scGPT_human")

CANDIDATE_TFS = ["MLXIPL", "ZEB1", "RORA", "TCF4", "FOXP1", "EBF1", "BHLHE40",
                 "CREB3L2", "GLIS3", "KLF4", "FOXO1", "STAT3", "EGR1", "NFIA"]
REFERENCE_TFS = ["PPARG", "CEBPA"]  # calibration: one known-strong, one known-inert
TOP_PARTNERS = 4

N_CELLS = 128
N_BACKGROUND = 60
BATCH_SIZE = 16
SEED = 42

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_bins, mask_value, pad_value = 51, -1, -2

set_seed(SEED)
rng = np.random.default_rng(SEED)
save_dir = Path(f"diabetes_scGPT_mlm_objective/save/tf-profile-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# ================= PART A =================
with open(GRN_DIR / "gene_panel.json") as f:
    attn_panel = json.load(f)
with open(COEXP_DIR / "gene_panel.json") as f:
    corr_panel = json.load(f)
attn_idx = {g: i for i, g in enumerate(attn_panel)}
corr_idx = {g: i for i, g in enumerate(corr_panel)}

attn_diff = np.load(GRN_DIR / "difference_matrices.npz")
corr_diff = np.load(COEXP_DIR / "correlation_difference_matrices.npz")
corr_raw = np.load(COEXP_DIR / "correlation_matrices.npz")


def offdiag(m):
    return m[~np.eye(m.shape[0], dtype=bool)]


attn_bg = {k: np.sort(np.abs(offdiag(attn_diff[k]))) for k in attn_diff.files}
corr_bg = {k: np.sort(np.abs(offdiag(corr_diff[k]))) for k in corr_diff.files}


def pct(v, sorted_bg):
    return np.searchsorted(sorted_bg, np.abs(v)) / len(sorted_bg)


de_t1d = pd.read_csv("64_donor_T1D_edgeR_all_genes_all_celltypes.tsv", sep="\t")
de_t1d = de_t1d[de_t1d["celltype"] == "Adipocyte"].set_index("gene")
de_t2d = pd.read_csv("64_donor_T2D_edgeR_all_genes_all_celltypes.tsv", sep="\t")
de_t2d = de_t2d[de_t2d["celltype"] == "Adipocyte"].set_index("gene")


def de_tag(g):
    out = []
    for name, d in (("T1D", de_t1d), ("T2D", de_t2d)):
        if g in d.index and str(d.loc[g, "diffexpressed"]) != "No":
            out.append(f"{name}:{d.loc[g, 'diffexpressed']}")
    return ",".join(out) if out else "notDE"


logger.info("=" * 110)
logger.info("PART A -- strict partner profiles (both conditions must agree, correlation shift same sign)")
logger.info("=" * 110)

tf_partners = {}
for tf in REFERENCE_TFS + CANDIDATE_TFS:
    if tf not in attn_idx or tf not in corr_idx:
        logger.info(f"\n### {tf}: not in panels, skipped")
        continue
    ai, ci = attn_idx[tf], corr_idx[tf]
    partners = [p for p in corr_panel if p != tf and p in attn_idx]
    aj = np.array([attn_idx[p] for p in partners])
    cj = np.array([corr_idx[p] for p in partners])

    a1 = np.maximum(np.abs(attn_diff["diff_T1D"][ai, aj]), np.abs(attn_diff["diff_T1D"][aj, ai]))
    a2 = np.maximum(np.abs(attn_diff["diff_T2D"][ai, aj]), np.abs(attn_diff["diff_T2D"][aj, ai]))
    c1 = corr_diff["diff_T1D"][ci, cj]
    c2 = corr_diff["diff_T2D"][ci, cj]

    df = pd.DataFrame({
        "partner": partners,
        "attn_strict": np.minimum(pct(a1, attn_bg["diff_T1D"]), pct(a2, attn_bg["diff_T2D"])),
        "corr_strict": np.minimum(pct(c1, corr_bg["diff_T1D"]), pct(c2, corr_bg["diff_T2D"])),
        "same_sign": np.sign(c1) == np.sign(c2),
        "r_healthy": corr_raw["No_Diabetes"][ci, cj],
        "r_T1D": corr_raw["T1D"][ci, cj],
        "r_T2D": corr_raw["T2D"][ci, cj],
        "dr_T1D": c1,
        "dr_T2D": c2,
    })
    df = df[df["same_sign"]].copy()
    df["dual_strict"] = df[["attn_strict", "corr_strict"]].min(axis=1)
    df = df.sort_values("dual_strict", ascending=False).head(TOP_PARTNERS)
    tf_partners[tf] = df["partner"].tolist()

    ref = " (reference)" if tf in REFERENCE_TFS else ""
    logger.info(f"\n### {tf}{ref} -- {de_tag(tf)}")
    logger.info(f"{'partner':<12} {'dual':>6} {'attn':>6} {'corr':>6} {'r_hlth':>7} {'r_T1D':>7} {'r_T2D':>7} {'dr_T1D':>7} {'dr_T2D':>7}  DE")
    for _, r in df.iterrows():
        logger.info(f"{r['partner']:<12} {r['dual_strict']:>6.0%} {r['attn_strict']:>6.0%} {r['corr_strict']:>6.0%} "
                    f"{r['r_healthy']:>7.2f} {r['r_T1D']:>7.2f} {r['r_T2D']:>7.2f} {r['dr_T1D']:>+7.2f} {r['dr_T2D']:>+7.2f}  {de_tag(r['partner'])}")

# ================= PART B =================
logger.info("\n" + "=" * 110)
logger.info("PART B -- perturbation probe for directionality")
logger.info("=" * 110)

interest = []
for tf, ps in tf_partners.items():
    interest.append(tf)
    interest.extend(ps)
interest = sorted(set(interest))
logger.info(f"Interest set: {len(interest)} genes")

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
panel = [g for g in attn_panel if g in set(adata.var["gene_name"])]
adata = adata[:, panel].copy()

raw_for_expr = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
mean_expr = dict(zip(panel, raw_for_expr.mean(axis=0)))
hi_cut = float(np.percentile(list(mean_expr.values()), 98))
logger.info(f"High-expression flag cutoff (98th pct of mean counts): {hi_cut:.2f}")

Preprocessor(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
             normalize_total=1e4, result_normed_key="X_normed", log1p=True,
             result_log1p_key="X_log1p", subset_hvg=False, binning=n_bins,
             result_binned_key="X_binned")(adata)

vocab = GeneVocab.from_file(SCGPT_DIR / "vocab.json")
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
vocab.set_default_index(vocab[pad_token])
with open(SCGPT_DIR / "args.json") as f:
    mc = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerModel(
    len(vocab), mc["embsize"], mc["nheads"], mc["d_hid"], mc["nlayers"], vocab=vocab,
    dropout=0.2, pad_token=pad_token, pad_value=pad_value, do_mvc=True, do_dab=False,
    use_batch_labels=False, num_batch_labels=len(set(adata.obs["age"].astype("category").cat.codes.values)),
    domain_spec_batchnorm=False, n_input_bins=n_bins, ecs_threshold=0,
    explicit_zero_prob=False, use_fast_transformer=True, pre_norm=False,
)
missing, unexpected = model.load_state_dict(torch.load(MODEL_FILE, map_location=device), strict=False)
logger.info(f"Loaded model; missing={len(missing)} unexpected={len(unexpected)}")
model.to(device).eval()

genes = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)
interest = [g for g in interest if g in set(genes)]
bg_pool = [g for g in genes if g not in set(interest)]
background = list(rng.choice(bg_pool, size=N_BACKGROUND, replace=False))


def fwd(tg, vals):
    outs = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for i in range(0, tg.size(0), BATCH_SIZE):
            g = tg[i:i + BATCH_SIZE].to(device)
            v = vals[i:i + BATCH_SIZE].to(device)
            o = model(g, v.float(), src_key_padding_mask=g.eq(vocab[pad_token]),
                      batch_labels=None, MVC=False, ECS=False)
            outs.append(o["mlm_output"].float().cpu())
    return torch.cat(outs, 0)


all_deltas = {}
for cond in ["No_Diabetes", "T1D", "T2D"]:
    sub = adata[adata.obs["condition"] == cond]
    take = rng.choice(sub.n_obs, size=min(N_CELLS, sub.n_obs), replace=False)
    sub = sub[take].copy()
    counts = sub.layers["X_binned"].toarray() if issparse(sub.layers["X_binned"]) else sub.layers["X_binned"]
    tok = tokenize_and_pad_batch(counts, gene_ids, max_len=len(genes) + 1, vocab=vocab,
                                 pad_token=pad_token, pad_value=pad_value,
                                 append_cls=True, include_zero_gene=True)
    tg, tv = tok["genes"], tok["values"]
    col = {g: c for c, g in enumerate(tg[0].tolist())}
    col_of = {g: col[vocab[g]] for g in interest + background if vocab[g] in col}

    deltas = {}
    for n, pg in enumerate(interest + background, 1):
        if pg not in col_of:
            continue
        measured = [m for m in interest if m != pg and m in col_of]
        mcols = [col_of[m] for m in measured]
        vb = tv.clone()
        vb[:, mcols] = mask_value
        vp = vb.clone()
        vp[:, col_of[pg]] = 0
        pb, pp = fwd(tg, vb), fwd(tg, vp)
        deltas[pg] = {m: float((pp[:, col_of[m]] - pb[:, col_of[m]]).mean()) for m in measured}
        if n % 30 == 0:
            logger.info(f"[{cond}] {n}/{len(interest) + len(background)}")
    all_deltas[cond] = deltas
    logger.info(f"[{cond}] done")

logger.info("\n" + "=" * 110)
logger.info("DIRECTIONALITY: zero TF -> partner  vs  zero partner -> TF  (mean over conditions)")
logger.info("hi = gene mean expression above 98th pct; probe unreliable there")
logger.info("=" * 110)
rows = []
for tf, ps in tf_partners.items():
    for p in ps:
        fwd_vals, rev_vals = [], []
        for c in all_deltas:
            d = all_deltas[c]
            if tf in d and p in d[tf]:
                fwd_vals.append(d[tf][p])
            if p in d and tf in d[p]:
                rev_vals.append(d[p][tf])
        if not fwd_vals or not rev_vals:
            continue
        f, r = float(np.mean(fwd_vals)), float(np.mean(rev_vals))
        ratio = abs(f) / abs(r) if abs(r) > 1e-9 else np.inf
        rows.append({"tf": tf, "partner": p, "zero_tf_to_partner": f, "zero_partner_to_tf": r,
                     "asymmetry_ratio": ratio,
                     "tf_hi_expr": mean_expr.get(tf, 0) > hi_cut,
                     "partner_hi_expr": mean_expr.get(p, 0) > hi_cut})

res = pd.DataFrame(rows).sort_values("asymmetry_ratio", ascending=False)
logger.info(f"{'TF':<10} {'partner':<12} {'zeroTF->p':>10} {'zerop->TF':>10} {'ratio':>7}  flags")
for _, r in res.iterrows():
    flags = []
    if r["tf_hi_expr"]:
        flags.append("TF-hi")
    if r["partner_hi_expr"]:
        flags.append("partner-hi")
    logger.info(f"{r['tf']:<10} {r['partner']:<12} {r['zero_tf_to_partner']:>+10.4f} "
                f"{r['zero_partner_to_tf']:>+10.4f} {r['asymmetry_ratio']:>7.1f}  {','.join(flags)}")

res.to_csv(save_dir / "tf_directionality.csv", index=False)
with open(save_dir / "tf_partners.json", "w") as f:
    json.dump(tf_partners, f, indent=2)
logger.info(f"\nSaved to {save_dir}")
logger.info("Done.")
