import copy
import json
import os
from pathlib import Path
import sys
import warnings
import time

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
# import gseapy as gp
from gears import PertData, GEARS

from scipy.sparse import issparse
import scipy as sp
from einops import rearrange
from torch.nn.functional import softmax, linear as F_linear
from tqdm import tqdm
import pandas as pd

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed 
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import Preprocessor

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')


save_dir = Path(f"diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")


set_seed(42)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 3000
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins


model_dir = Path("diabetes_scGPT_mlm_objective/save/dev_diabetes_mlm_objective_claude_pearsonhvg_1-Oct26-22-11")
model_config_file = "scGPT_human/args.json"
model_file = model_dir / "model_e20.pt"  # best/last epoch of the pearson_residuals-HVG finetune run; no best_model.pt alias was saved this run
vocab_file = "scGPT_human/vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# Retrieve model parameters from config files
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
print(
    f"Resume model from {model_file}, the model args will override the "
    f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

gene2idx = vocab.get_stoi()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    pad_value=pad_value,
    n_input_bins=n_input_bins,
    use_fast_transformer=True,
)

try:
    model.load_state_dict(torch.load(model_file))
    print(f"Loading all model params from {model_file}")
except:
    # only load params that are in the model and match the size
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)


# using all adipocytes (train and test)
adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
adata.obs = adata.obs.rename({'type' : 'condition', 'celltype_atac': 'cell_type'}, axis = 1)
adata = adata[((adata.obs['cell_type'].isin(['Adipocyte'])) & adata.obs['condition'].isin(['No_Diabetes', 'T1D', 'T2D']))] 

condition_id_labels = adata.obs["condition"].astype("category").cat.codes.values
conditions = adata.obs["condition"].unique()
num_types = len(np.unique(condition_id_labels))
id2type = dict(enumerate(adata.obs["condition"].astype("category").cat.categories))
adata.obs["condition_id"] = condition_id_labels
adata.var['gene_name'] = adata.var['gene_id']
adata.var_names = adata.var.gene_id
adata.obs['batch_id'] = adata.obs['age'].astype("category").cat.codes.values


# qc cell-level filtering
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
n_obs_before = adata.n_obs
adata = adata[
    (adata.obs["pct_counts_mt"] < 10) &
    (adata.obs["total_counts"] > 1500) &
    (adata.obs["n_genes_by_counts"] > 800)
].copy()
logger.info(f"After QC: {adata.n_obs}  / {n_obs_before} cells")
logger.info(adata.obs['condition'].value_counts().to_string())
logger.info(adata.obs.groupby('condition')['age'].nunique().to_string())


# NEW: PTPRC purity gate — exclude cells with any detectable immune-marker expression
ptprc_idx = adata.var_names.get_loc('PTPRC')
ptprc_counts = adata.X[:, ptprc_idx]
if issparse(adata.X):
    ptprc_counts = ptprc_counts.toarray().flatten()
adata = adata[ptprc_counts == 0].copy()
logger.info(f"After PTPRC purity gate: {adata.n_obs} cells")
logger.info(adata.obs['condition'].value_counts().to_string())


# only keep genes that are present in vocab 
adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"]]
gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
adata = adata[:, adata.var["id_in_vocab"] >= 0]

# drop genes that are not expressed in more than 5% of all adipocyte cells (gene-level prevalence-based filter)
n_genes_before_prevalence = adata.n_vars
min_cells = int(0.05 * adata.n_obs)  # expressed in at least 5% of cells
genes_before = set(adata.var_names)
sc.pp.filter_genes(adata, min_cells=min_cells)
genes_dropped = genes_before - set(adata.var_names)
logger.info(
    f"Prevalence filter (min_cells={min_cells}, {100*min_cells/adata.n_obs:.1f}% of cells): "
    f"{adata.n_vars} / {n_genes_before_prevalence} genes retained, "
    f"{len(genes_dropped)} dropped"
)
# logger.info(f"Dropped genes: {sorted(genes_dropped)}")


# preprocessing
# data confirmed RAW counts (see src/check_gene_panel_flavor.py diagnostic: non-negative,
# integer-valued, max 4980). Previously set to False here, which silently skipped log1p
# before binning -- inconsistent with mlm_finetune_adipocyte*.py, which always log1p's.
# Fixed to True so attention is extracted from data prepared the same way the model was trained on.
data_is_raw = True
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1, gene level count-based filter (total count > 3)
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg= False,  # 5. whether to subset the raw data to highly variable genes (irrelevant -- always False here, so hvg_flavor below is unused)
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata)


# filter to include only hvgs selected in mlm finetuning
with open(model_dir / "hvg_selection.json", "r") as f:
    hvg_genes = json.load(f)

hvg_genes_present = [g for g in hvg_genes if g in adata.var_names] # only use original 3k hvgs that are still present post gene filtering
logger.info(f"Using {len(hvg_genes_present)} / {len(hvg_genes)} genes (dropped {len(hvg_genes) - len(hvg_genes_present)})")

# extend to include known TF/target marker genes (all 10 already included via the
# pearson_residuals panel + forced markers at finetune time, so this is a no-op safety net)
canonical_markers = ["PPARG", "CEBPA", "CEBPB", "CEBPD", "SREBF1", "FABP4", "CD36", "PLIN1", "LPL", "ADIPOQ"]
extended_gene_list = hvg_genes_present + [g for g in canonical_markers if g not in hvg_genes_present]
logger.info(
    f"Extended gene panel: {len(extended_gene_list)} genes "
    f"({len(hvg_genes_present)} original + {len(extended_gene_list) - len(hvg_genes_present)} markers added)"
)

adata = adata[:, extended_gene_list].copy()
# adata = adata[:, hvg_genes_present].copy()


# dataset preparation
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_all = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=len(genes)+1,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)

all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
condition_ids = np.array(adata.obs["condition"].tolist())


# --------extract attention matrices---------
torch.cuda.empty_cache()
dict_sum_condition = {}
# Use this argument to specify which layer to extract the attention weights from
# Default to 11, extraction from the last (12th) layer. Note that index starts from 0
num_attn_layers = 11 
batch_size = 8
model.eval()
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    M = all_gene_ids.size(1)
    N = all_gene_ids.size(0)
    device = next(model.parameters()).device

    total_batches = (N + batch_size - 1) // batch_size
    for i in tqdm(range(0, N, batch_size)):

        # progress logging every 100 batches
        batch_num = i // batch_size + 1
        if batch_num % 100 == 0:
            logger.info(f"Attention extraction: {batch_num} / {total_batches} batches done ({batch_num/total_batches*100:.1f}%)")

        batch_size = all_gene_ids[i : i + batch_size].size(0)
        outputs = np.zeros((batch_size, M, M), dtype=np.float32)
        # Replicate the operations in model forward pass
        src_embs = model.encoder(torch.tensor(all_gene_ids[i : i + batch_size], dtype=torch.long).to(device))
        val_embs = model.value_encoder(torch.tensor(all_values[i : i + batch_size], dtype=torch.float).to(device))
        total_embs = src_embs + val_embs
        # total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        # Send total_embs to attention layers for attention operations
        # Retrieve the output from second to last layer
        for layer in model.transformer_encoder.layers[:num_attn_layers]:
            total_embs = layer(total_embs, src_key_padding_mask=src_key_padding_mask[i : i + batch_size].to(device))
        # Send total_embs to the last layer's attention module.
        # Original code assumed a flash-attn FlashMHA module exposing a fused `.Wqkv`
        # linear layer, but flash_attn is not installed in this environment, so scgpt's
        # TransformerModel falls back to standard torch.nn.MultiheadAttention, which
        # exposes the same fused QKV projection as `in_proj_weight`/`in_proj_bias`
        # instead. F.linear with those weights reproduces what `.Wqkv(total_embs)` would.
        last_attn = model.transformer_encoder.layers[num_attn_layers].self_attn
        qkv = F_linear(total_embs, last_attn.in_proj_weight, last_attn.in_proj_bias)
        # Retrieve q, k, and v from the fused projection
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=8)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]
        # https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
        # q = [batch, gene, n_heads, n_hid]
        # k = [batch, gene, n_heads, n_hid]
        # attn_scores = [batch, n_heads, gene, gene]
        attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
        # Rank normalization by row
        attn_scores = attn_scores.reshape((-1, M))
        order = torch.argsort(attn_scores, dim=1)
        rank = torch.argsort(order, dim=1)
        attn_scores = rank.reshape((-1, 8, M, M))/M
        # Rank normalization by column
        attn_scores = attn_scores.permute(0, 1, 3, 2).reshape((-1, M))
        order = torch.argsort(attn_scores, dim=1)
        rank = torch.argsort(order, dim=1)
        attn_scores = (rank.reshape((-1, 8, M, M))/M).permute(0, 1, 3, 2)
        # Average 8 attention heads
        attn_scores = attn_scores.mean(1)
        
        outputs = attn_scores.detach().cpu().numpy()
        
        for index in range(batch_size):
            # Keep track of sum per condition
            c = condition_ids[i : i + batch_size][index]
            if c not in dict_sum_condition:
                dict_sum_condition[c] = np.zeros((M, M), dtype=np.float32)
            dict_sum_condition[c] += outputs[index, :, :]




# split by condition
groups = adata.obs.groupby('condition').groups
dict_sum_condition_mean = dict_sum_condition.copy()
for i in groups.keys():
    dict_sum_condition_mean[i] = dict_sum_condition_mean[i]/len(groups[i])
gene_vocab_idx = all_gene_ids[0].clone().detach().cpu().numpy()
logger.info(dict_sum_condition_mean)



# extract difference matrices: condition vs healthy baseline
diff_T1D = dict_sum_condition_mean['T1D'] - dict_sum_condition_mean['No_Diabetes']
diff_T2D = dict_sum_condition_mean['T2D'] - dict_sum_condition_mean['No_Diabetes']

def get_topk_influenced_genes_global(diff_matrix, gene_vocab_idx, vocab, topk=20, mode="both"):
    """
    mode:
      'received' - genes whose attention RECEIVED from others changed most (column sums) (in-degree)
      'sent'     - genes whose attention TO others changed most (row sums) (out-degree)
      'both'     - combined magnitude of both
    """
    received_change = diff_matrix.sum(axis=0)  # sum over rows -> per-column (attention received)
    sent_change = diff_matrix.sum(axis=1)       # sum over columns -> per-row (attention sent)

    if mode == "received":
        score = received_change
    elif mode == "sent":
        score = sent_change
    else:  # both -- combined absolute magnitude
        score = np.abs(received_change) + np.abs(sent_change)

    top_idx = np.argpartition(score, -topk)[-topk:]
    top_idx = top_idx[np.argsort(-score[top_idx])]  # sort descending by score

    top_genes = vocab.lookup_tokens(gene_vocab_idx[top_idx].tolist())
    top_scores = score[top_idx]

    return list(top_genes), list(top_scores)

genes_T1D, scores_T1D = get_topk_influenced_genes_global(diff_T1D, gene_vocab_idx, vocab, topk=20, mode="both")
genes_T2D, scores_T2D = get_topk_influenced_genes_global(diff_T2D, gene_vocab_idx, vocab, topk=20, mode="both")

logger.info(f"Top T1D-driving genes: {genes_T1D}")
logger.info(f"Scores T1D: {scores_T1D}")
logger.info(f"Top T2D-driving genes: {genes_T2D}")
logger.info(f"Scores T2D: {scores_T2D}")


# --- SAVE EVERYTHING ---

# 1. Raw per-condition mean attention matrices (the expensive-to-recompute artifact)
np.savez(
    save_dir / "attention_matrices.npz",
    **{cond: mat for cond, mat in dict_sum_condition_mean.items()}
)

# 2. Difference matrices
np.savez(
    save_dir / "difference_matrices.npz",
    diff_T1D=diff_T1D,
    diff_T2D=diff_T2D,
)

# 3. Gene vocab index (needed to map matrix positions back to gene names later)
np.save(save_dir / "gene_vocab_idx.npy", gene_vocab_idx)
with open(save_dir / "gene_panel.json", "w") as f:
    json.dump(vocab.lookup_tokens(gene_vocab_idx.tolist()), f)

# 4. Condition / donor counts -- the sanity-check numbers that matter most
condition_counts = adata.obs['condition'].value_counts().to_dict()
donor_counts = adata.obs.groupby('condition')['age'].nunique().to_dict()  # swap 'age' for your real donor/batch column if different
with open(save_dir / "sample_counts.json", "w") as f:
    json.dump({
        "cells_per_condition": condition_counts,
        "donors_per_condition": donor_counts,
    }, f, indent=2)

# 5. Ranked gene lists + scores, for every topk/mode combination you've run so far
ranked_results = {
    "T1D": {"genes": genes_T1D, "scores": [float(s) for s in scores_T1D]},
    "T2D": {"genes": genes_T2D, "scores": [float(s) for s in scores_T2D]},
}
with open(save_dir / "ranked_genes_top20.json", "w") as f:
    json.dump(ranked_results, f, indent=2)

# 6. Also save as a flat CSV for quick viewing / spreadsheet use
rows = []
for cond, d in ranked_results.items():
    for rank, (gene, score) in enumerate(zip(d["genes"], d["scores"]), start=1):
        rows.append({"condition": cond, "rank": rank, "gene": gene, "score": score})
pd.DataFrame(rows).to_csv(save_dir / "ranked_genes_top20.csv", index=False)

logger.info(f"Done. Files saved:")
for f in sorted(save_dir.glob("*")):
    logger.info(f"  {f.name}")




# - canonical marker gene - specific tests
def get_score_for_specific_genes(diff_matrix, gene_vocab_idx, vocab, gene_names):
    """Look up score + rank-percentile for specific genes, whether or not they're in the global top-k."""
    received_change = diff_matrix.sum(axis=0)
    sent_change = diff_matrix.sum(axis=1)
    score = np.abs(received_change) + np.abs(sent_change)

    all_gene_names = vocab.lookup_tokens(gene_vocab_idx.tolist())
    gene_to_idx = {g: i for i, g in enumerate(all_gene_names)}

    # rank all genes by score, descending, to get percentile rank
    order = np.argsort(-score)
    rank_of = {idx: r for r, idx in enumerate(order)}
    n_genes = len(score)

    results = {}
    for g in gene_names:
        if g not in gene_to_idx:
            results[g] = None
            continue
        idx = gene_to_idx[g]
        results[g] = {
            "score": float(score[idx]),
            "rank": int(rank_of[idx]) + 1,  # 1-indexed
            "percentile": float(1 - rank_of[idx] / n_genes),  # 1.0 = top gene
        }
    return results

marker_scores_T1D = get_score_for_specific_genes(diff_T1D, gene_vocab_idx, vocab, canonical_markers)
marker_scores_T2D = get_score_for_specific_genes(diff_T2D, gene_vocab_idx, vocab, canonical_markers)
logger.info(f"Canonical marker scores (T1D): {json.dumps(marker_scores_T1D, indent=2)}")
logger.info(f"Canonical marker scores (T2D): {json.dumps(marker_scores_T2D, indent=2)}")



def get_raw_attention_stats(attn_matrix, gene_vocab_idx, vocab, gene_names):
    all_gene_names = vocab.lookup_tokens(gene_vocab_idx.tolist())
    gene_to_idx = {g: i for i, g in enumerate(all_gene_names)}
    row_sums = attn_matrix.sum(axis=1)
    panel_mean, panel_std = row_sums.mean(), row_sums.std()

    stats = {}
    for g in gene_names:
        if g not in gene_to_idx:
            continue
        idx = gene_to_idx[g]
        stats[g] = {
            "row_sum": float(row_sums[idx]),
            "z_score_vs_panel": float((row_sums[idx] - panel_mean) / panel_std),
        }
    return stats

for cond in dict_sum_condition_mean:
    stats = get_raw_attention_stats(dict_sum_condition_mean[cond], gene_vocab_idx, vocab, canonical_markers)
    logger.info(f"Raw attention stats for markers, condition={cond}: {json.dumps(stats, indent=2)}")





