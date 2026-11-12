#!/usr/bin/env python
"""Capstone validation of the ZEB1-PPARG/CEBPB candidate.

Four questions, per domain consult:
  1. DIRECTION  -- zero ZEB1, measure PPARG/CEBPB, vs. the reverse. The one
                   directionality test we never actually ran.
  2. DONOR-LEVEL SIGNIFICANCE -- thousands of cells from ~60 donors makes almost
                   anything look significant. Correlations are computed WITHIN
                   each donor, so each donor is one observation, and condition
                   labels are permuted across donors for the null.
  3. EFFECT SIZE -- is the donor-level shift bigger than random gene pairs show?
  4. CONFOUNDERS -- does age or sex explain it? (BMI is NOT in this dataset;
                   that confounder remains unaddressed.)
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
MODEL_FILE = Path("diabetes_scGPT_mlm_objective/save/dev_diabetes_mlm_objective_claude_pearsonhvg_1-Oct26-22-11/model_e20.pt")
SCGPT_DIR = Path("scGPT_human")

FOCUS_PAIRS = [("ZEB1", "PPARG"), ("ZEB1", "CEBPB"), ("ZEB1", "CEBPA"), ("ZEB1", "CEBPD"),
               ("PPARG", "LPL")]                      # last one = reference positive control
PROBE_GENES = ["ZEB1", "PPARG", "CEBPA", "CEBPB", "CEBPD", "LPL", "GPAM", "PDE3B"]

MIN_CELLS_PER_DONOR = 30
N_PERM = 10000
N_RANDOM_PAIRS = 500
N_CELLS_PROBE = 128
N_BACKGROUND = 60
BATCH_SIZE = 16
SEED = 42

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_bins, mask_value, pad_value = 51, -1, -2

set_seed(SEED)
rng = np.random.default_rng(SEED)
save_dir = Path(f"diabetes_scGPT_mlm_objective/save/zeb1-validation-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# ---------------- data ----------------
adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
adata.obs = adata.obs.rename({"type": "condition", "celltype_atac": "cell_type"}, axis=1)
logger.info(f"obs columns available: {list(adata.obs.columns)}")

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

# donor id: prefer an obs column, else derive from the donor-prefixed barcode
donor_col = next((c for c in ["orig.ident", "orig_ident", "donor", "donor_id", "sample"]
                  if c in adata.obs.columns), None)
if donor_col:
    adata.obs["donor"] = adata.obs[donor_col].astype(str)
    logger.info(f"donor id taken from obs column '{donor_col}'")
else:
    adata.obs["donor"] = [str(b).split("_")[0] for b in adata.obs_names]
    logger.info("donor id derived from barcode prefix (no obs donor column found)")
logger.info(f"{adata.n_obs} cells, {adata.obs['donor'].nunique()} donors")

with open(GRN_DIR / "gene_panel.json") as f:
    panel = [g for g in json.load(f) if g in set(adata.var["gene_name"])]
adata = adata[:, panel].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
X = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
gidx = {g: i for i, g in enumerate(panel)}

# ---------------- per-donor correlations ----------------
donors = []
for d, sub in adata.obs.groupby("donor", observed=True):
    if len(sub) < MIN_CELLS_PER_DONOR:
        continue
    mask = (adata.obs["donor"] == d).values
    donors.append({
        "donor": d, "n_cells": int(mask.sum()),
        "condition": str(sub["condition"].iloc[0]),
        "age": float(pd.to_numeric(sub["age"], errors="coerce").iloc[0]) if "age" in sub else np.nan,
        "sex": str(sub["sex"].iloc[0]) if "sex" in sub else "NA",
        "_mask": mask,
    })
logger.info(f"Donors with >={MIN_CELLS_PER_DONOR} adipocytes: {len(donors)}")
logger.info(pd.Series([d["condition"] for d in donors]).value_counts().to_string())


def donor_corr(mask, ga, gb):
    a, b = X[mask, gidx[ga]], X[mask, gidx[gb]]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def perm_test(vals, is_diab, n_perm=N_PERM):
    """Difference in mean per-donor r, diabetic vs healthy, permuting donor labels."""
    ok = ~np.isnan(vals)
    v, g = vals[ok], is_diab[ok]
    if g.sum() < 3 or (~g).sum() < 3:
        return np.nan, np.nan
    obs = v[g].mean() - v[~g].mean()
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(g)
        null[i] = v[p].mean() - v[~p].mean()
    return obs, float((np.abs(null) >= abs(obs)).mean())


logger.info("\n" + "=" * 100)
logger.info("Q2/Q3 -- DONOR-LEVEL test (each donor = 1 observation; labels permuted across donors)")
logger.info("=" * 100)

is_diab = np.array([d["condition"] != "No_Diabetes" for d in donors])
focus_results = {}
for ga, gb in FOCUS_PAIRS:
    if ga not in gidx or gb not in gidx:
        logger.info(f"{ga}-{gb}: not in panel")
        continue
    vals = np.array([donor_corr(d["_mask"], ga, gb) for d in donors])
    obs, p = perm_test(vals, is_diab)
    healthy = np.nanmean(vals[~is_diab])
    diabetic = np.nanmean(vals[is_diab])
    focus_results[f"{ga}-{gb}"] = {"vals": vals, "obs": obs, "p": p}
    logger.info(f"{ga}-{gb}:  mean r healthy={healthy:+.3f}  diabetic={diabetic:+.3f}  "
                f"diff={obs:+.3f}  permutation p={p:.4f}")

# effect size: same donor-level test on random pairs
logger.info("\nBuilding random-pair null for effect size ...")
rand_diffs = []
for _ in range(N_RANDOM_PAIRS):
    ga, gb = rng.choice(panel, size=2, replace=False)
    vals = np.array([donor_corr(d["_mask"], ga, gb) for d in donors])
    ok = ~np.isnan(vals)
    if ok.sum() < 10:
        continue
    rand_diffs.append(vals[ok & is_diab].mean() - vals[ok & ~is_diab].mean())
rand_diffs = np.array(rand_diffs)
logger.info(f"Random-pair donor-level |diff|: median={np.median(np.abs(rand_diffs)):.3f}  "
            f"95th pct={np.percentile(np.abs(rand_diffs), 95):.3f}")
for name, r in focus_results.items():
    if not np.isnan(r["obs"]):
        pctile = float((np.abs(rand_diffs) < abs(r["obs"])).mean())
        logger.info(f"  {name}: diff={r['obs']:+.3f} -> bigger than {pctile:.1%} of random pairs")

# ---------------- confounders ----------------
logger.info("\n" + "=" * 100)
logger.info("Q4 -- CONFOUNDERS: per-donor r ~ diabetic + age + sex   (OLS, numpy lstsq)")
logger.info("NOTE: BMI is NOT in this dataset -- that confounder is unaddressed.")
logger.info("=" * 100)
age = np.array([d["age"] for d in donors])
male = np.array([1.0 if d["sex"] == "Male" else 0.0 for d in donors])
for name, r in focus_results.items():
    vals = r["vals"]
    ok = ~np.isnan(vals) & ~np.isnan(age)
    if ok.sum() < 8:
        continue
    A = np.column_stack([np.ones(ok.sum()), is_diab[ok].astype(float), age[ok], male[ok]])
    beta, *_ = np.linalg.lstsq(A, vals[ok], rcond=None)
    resid = vals[ok] - A @ beta
    dof = max(ok.sum() - A.shape[1], 1)
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * (resid @ resid) / dof)
    logger.info(f"{name}:  beta_diabetic={beta[1]:+.3f} (SE {se[1]:.3f}, t={beta[1]/se[1]:+.2f})  "
                f"beta_age={beta[2]:+.4f} (t={beta[2]/se[2]:+.2f})  "
                f"beta_male={beta[3]:+.3f} (t={beta[3]/se[3]:+.2f})")

# ---------------- Q1: directionality ----------------
logger.info("\n" + "=" * 100)
logger.info("Q1 -- DIRECTIONALITY probe on ZEB1 <-> PPARG/CEBPs (the test we hadn't run)")
logger.info("=" * 100)
Preprocessor(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False,
             normalize_total=False, result_normed_key=None, log1p=False,
             result_log1p_key=None, subset_hvg=False, binning=n_bins,
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
    use_batch_labels=False,
    num_batch_labels=len(set(adata.obs["age"].astype("category").cat.codes.values)),
    domain_spec_batchnorm=False, n_input_bins=n_bins, ecs_threshold=0,
    explicit_zero_prob=False, use_fast_transformer=True, pre_norm=False)
missing, unexpected = model.load_state_dict(torch.load(MODEL_FILE, map_location=device), strict=False)
logger.info(f"model loaded; missing={len(missing)} unexpected={len(unexpected)}")
model.to(device).eval()

genes = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)
probe = [g for g in PROBE_GENES if g in set(genes)]
background = list(rng.choice([g for g in genes if g not in set(probe)], size=N_BACKGROUND, replace=False))


def fwd(tg, vals):
    outs = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for i in range(0, tg.size(0), BATCH_SIZE):
            g = tg[i:i + BATCH_SIZE].to(device)
            v = vals[i:i + BATCH_SIZE].to(device)
            outs.append(model(g, v.float(), src_key_padding_mask=g.eq(vocab[pad_token]),
                              batch_labels=None, MVC=False, ECS=False)["mlm_output"].float().cpu())
    return torch.cat(outs, 0)


probe_out = {}
for cond in ["No_Diabetes", "T1D", "T2D"]:
    sub = adata[adata.obs["condition"] == cond]
    take = rng.choice(sub.n_obs, size=min(N_CELLS_PROBE, sub.n_obs), replace=False)
    sub = sub[take].copy()
    counts = sub.layers["X_binned"].toarray() if issparse(sub.layers["X_binned"]) else sub.layers["X_binned"]
    tok = tokenize_and_pad_batch(counts, gene_ids, max_len=len(genes) + 1, vocab=vocab,
                                 pad_token=pad_token, pad_value=pad_value,
                                 append_cls=True, include_zero_gene=True)
    tg, tv = tok["genes"], tok["values"]
    col = {g: c for c, g in enumerate(tg[0].tolist())}
    col_of = {g: col[vocab[g]] for g in probe + background if vocab[g] in col}
    deltas = {}
    for pg in probe + background:
        if pg not in col_of:
            continue
        measured = [m for m in probe if m != pg and m in col_of]
        vb = tv.clone()
        vb[:, [col_of[m] for m in measured]] = mask_value
        vp = vb.clone()
        vp[:, col_of[pg]] = 0
        pb, pp = fwd(tg, vb), fwd(tg, vp)
        deltas[pg] = {m: float((pp[:, col_of[m]] - pb[:, col_of[m]]).mean()) for m in measured}
    probe_out[cond] = deltas
    logger.info(f"[{cond}] probe done")

logger.info(f"\n{'pair':<18} {'zero A -> B':>12} {'zero B -> A':>12} {'ratio':>8}  (mean over conditions)")
for ga, gb in FOCUS_PAIRS:
    f = [probe_out[c][ga][gb] for c in probe_out if ga in probe_out[c] and gb in probe_out[c].get(ga, {})]
    r = [probe_out[c][gb][ga] for c in probe_out if gb in probe_out[c] and ga in probe_out[c].get(gb, {})]
    if not f or not r:
        continue
    fm, rm = float(np.mean(f)), float(np.mean(r))
    ratio = abs(fm) / abs(rm) if abs(rm) > 1e-9 else np.inf
    logger.info(f"{ga}->{gb:<12} {fm:>12.4f} {rm:>12.4f} {ratio:>8.1f}")

out = {k: {"obs_diff": v["obs"], "perm_p": v["p"],
           "per_donor_r": [None if np.isnan(x) else float(x) for x in v["vals"]]}
       for k, v in focus_results.items()}
with open(save_dir / "donor_level_results.json", "w") as f:
    json.dump({"pairs": out, "donors": [{k: v for k, v in d.items() if k != "_mask"} for d in donors]}, f, indent=2)
logger.info(f"\nSaved to {save_dir}")
logger.info("Done.")
