#!/usr/bin/env python
# coding: utf-8

# ## TODOs: 
# 1. create the online sampling pipeline via the data collator (which also means to do dataset and loader creation manually)

# In[1]:

import subprocess

gpu_info = subprocess.check_output(["nvidia-smi"]).decode('utf-8')
print(gpu_info) 


import torch

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Error: {e}")


import numpy as np
import pandas as pd

import copy
import gc
import json
import os
import sys
import time
import warnings
import traceback
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Tuple, Dict, Union, Optional

import matplotlib.pyplot as plt
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

import scanpy as sc
from anndata import AnnData

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from sklearn.metrics import silhouette_score, pairwise_distances

import scgpt as scg
from scgpt import prepare_data, prepare_dataloader, define_wandb_metrcis
from scgpt.model import TransformerModel
from scgpt.scbank import DataBank
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer import GeneVocab, get_default_gene_vocab, tokenize_and_pad_batch
from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained
from scgpt.preprocess import Preprocessor



# print(f"Current working directory: {Path.cwd()}")


class config:
    # sanity
    sanity = False
    
    # static configs
    model_path = 'scGPT_human'
    pad_token = "<pad>"
    mask_value = -1
    pad_value = -2
    seed = 42
    task_name = "diabetes_mlm_objective_claude_pearsonhvg"
    task = "integration"
    trial_version = 1

    # sampling/preprocessing
    n_hvg = 3000 # don't use this once we have the sliding gene window selector
    n_hvp = 4000 # NOTE: still do hvp since there's too many peaks to attend to (too many additoinal vocabs), but slightly higher count should be allowed
    n_bins = 51
    input_layer_key = "X_binned"

    # architecture (IGNORED, we load the model)
    layer_size = 512
    nhead = 8 # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    nlayers = 4 # NOTE: for developing
    freeze = False 

    # objectives
    GEP=True # Gene expression modelling
    GEPC=True # Gene expression modelling for cell objective
    CLS=False
    ESC=False
    DAB = False # DAB objective for batch correction
    DSBN = False  # Domain-spec batchnorm

    # objective params
    mask_ratio=0.4 # Default mask ratio
    explicit_zero_prob = False  # whether explicit bernoulli for zeros
    ecs_thres=0  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0 # DAB weight 

    # dataset settings
    use_batch_labels = False # batch labels?
    use_mod = False # multi-modal?
    pre_norm=False # Default setting
    include_zero_gene = True # have compute?

    # sequence handling
    per_seq_batch_sample = False
    max_seq_len = 3001 # Default n_hvg+1 for pad_token

    # training optimization
    epochs=20
    warmup_epochs = 3
    lr=1e-5
    wd=1e-2 
    batch_size=64
    eval_batch_size=64
    dropout=0.2 
    fast_transformer= not sanity
    amp=True  

    # logging
    save_eval_interval=5 # Default model evaluation interval
    log_interval=100 # Default log interval

# dynamic bs selection
if torch.cuda.is_available():
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_vram < 8:
        config.batch_size = 16  # safety for crappy GPUs
        config.eval_batch_size = 16
    elif total_vram < 30:
        config.batch_size = 32 # safety for mid-tier
        config.eval_batch_size = 32
    else:
        config.batch_size = 64
        config.eval_batch_size = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# settings for input and preprocessing
special_tokens = [config.pad_token, "<cls>", "<eoc>"]
set_seed(config.seed)

if config.sanity:
    logger = scg.logger
    save_dir = Path("./")
else:
    save_dir = Path(f"diabetes_scGPT_mlm_objective/save/dev_{config.task_name}_{config.trial_version}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

logger.info(f"Running with batch size of: {config.batch_size}")


# ## Load and preprocess data
adata = sc.read_h5ad("diabetes_scGPT_mlm_objective/data_files/rna_scanpy_object_260516.h5ad")
# meta_data = pd.read_csv('diabetes_scGPT_mlm_objective/data_files/meta_data_final_annotations_260515.tsv', sep='\t')
adata.obs = adata.obs.rename({'type' : 'condition', 'celltype_atac': 'cell_type'}, axis = 1)

# # add a dummy_batch cuz we have no batch info
# adata_t.obs["str_batch"] = "batch_0"

# filter out the other types of Diabetes (only 1 or 2 donor, trivial), and restricts to adipocytes only
adata = adata[((adata.obs['cell_type'].isin(['Adipocyte'])) & adata.obs['condition'].isin(['No_Diabetes', 'T1D', 'T2D']))] 

# load only the dev indices
with open("diabetes_scGPT_mlm_objective/data_files/dev_idx_adipocyte.json", "r") as f:
    dev_idx = json.load(f)
adata = adata[dev_idx].copy()

# rename required cols
adata.var['gene_name'] = adata.var['gene_id']
adata.var_names = adata.var.gene_id
adata.obs['batch_id'] = adata.obs['age'].astype("category").cat.codes.values

# ## load model info
if config.model_path is not None:
    model_dir = Path(config.model_path)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    
    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size 
    nhead = config.nhead
    nlayers = config.nlayers  
    d_hid = config.layer_size


# filtering for sanity run (only include 100 cells for each type)
if config.sanity:
    target_types = ["T2D", "T1D", "No_Diabetes"]
    
    filtered_obs = adata.obs[adata.obs['condition'].isin(target_types)]
    
    # randomly sample 100 from each group
    sampled_obs = filtered_obs.groupby('condition', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 100), random_state=42, replace=False)
    )
    
    keep_cells = sampled_obs.index
    adata_filtered = adata[keep_cells].copy()
    
    print(adata_filtered.obs['condition'].value_counts())
    adata = adata_filtered



# cell-level qc 
n_cells_orig = adata.n_obs
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata = adata[
    (adata.obs["pct_counts_mt"] < 10) &       # Exclude dying/burst cells
    (adata.obs["total_counts"] > 1500) &      # Exclude empty droplets
    (adata.obs["n_genes_by_counts"] > 800)    # Exclude ultra-low complexity background RNA
].copy()

logger.info(f"Kept {adata.n_obs} / {n_cells_orig} high-quality cells")


# ## Preprocess (change to the online gene window sampling later by discarding this whole preprocess pipeline)
# Gene panel selection: pearson_residuals (Lause/Kobak/Berens 2021) instead of
# seurat_v3. Diagnostic check (src/check_gene_panel_flavor.py) confirmed adata.X
# is raw counts (required by both flavors) and that pearson_residuals recovers
# 7/10 known adipocyte TF/target genes vs. seurat_v3's 3/10 on this dataset.
sc.pp.filter_genes(adata, min_counts=3)  # same as Preprocessor's filter_gene_by_counts=3, done here since HVG selection now happens before Preprocessor

sc.experimental.pp.highly_variable_genes(
    adata,
    n_top_genes=config.n_hvg,
    flavor="pearson_residuals",
    check_values=True,
)

# Force-include known adipogenic TFs not picked up by pearson_residuals on this
# data (CEBPA, SREBF1) -- same "extend panel with canonical markers" pattern
# already used in grn_mlm_adipocyte.py.
forced_markers = ["CEBPA", "SREBF1"]
for g in forced_markers:
    if g in adata.var["gene_name"].values:
        adata.var.loc[adata.var["gene_name"] == g, "highly_variable"] = True
    else:
        logger.warning(f"Forced marker {g} not found in adata, skipping")

adata = adata[:, adata.var["highly_variable"]].copy()

preprocessor_rna = Preprocessor(
    use_key="X",
    filter_gene_by_counts=False,  # already done above, before HVG selection
    filter_cell_by_counts=False,
    normalize_total=1e4,
    result_normed_key="X_normed",
    log1p=True,
    result_log1p_key="X_log1p",
    subset_hvg=False,  # already subset above
    binning=config.n_bins,
    result_binned_key="X_binned",
)
preprocessor_rna(adata, batch_key = None)

hvg_genes = adata.var["gene_name"].tolist()
hvg_path = save_dir / "hvg_selection.json"
with open(hvg_path, "w") as f:
    json.dump(hvg_genes, f)
logger.info(f"Saved {len(hvg_genes)} genes (pearson_residuals HVG + forced markers) to {hvg_path}")

# panel size is no longer guaranteed to be exactly config.n_hvg (forced_markers
# can add genes on top of the n_top_genes pearson_residuals selection), so
# max_seq_len must be recomputed instead of using the static n_hvg+1 value
config.max_seq_len = len(hvg_genes) + 1  # +1 for <cls> token
logger.info(f"Set config.max_seq_len = {config.max_seq_len} (panel size {len(hvg_genes)} + 1 cls token)")


# ## Tokenize
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].toarray()
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

# change cell type labels to condition later if doing cls multitask
celltypes_labels = adata.obs["cell_type"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

if config.model_path is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=config.include_zero_gene,
    mod_type=mod_type if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=config.max_seq_len,
    vocab=vocab,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    append_cls=True,
    include_zero_gene=config.include_zero_gene,
    mod_type=mod_type if config.use_mod else None,
    vocab_mod=vocab_mod if config.use_mod else None,
)
logger.info(
    f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
)
logger.info(
    f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
    f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
)


# ## Load model weights
model_dict = torch.load(model_file)
ntokens = len(vocab)
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid, 
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=config.DAB,
    use_batch_labels=config.use_batch_labels,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)

if config.model_path is not None:
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # for k, v in pretrained_dict.items():
        #     logger.info(f"Loading params {k} with shape {v.shape}")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)

# ## Training Functions
criterion = masked_mse_loss
# criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, weight_decay=getattr(config, "wd", 1e-4), eps=1e-4 if config.amp else 1e-8
)
T_max = config.epochs # switch to config.epochs * len(train_loader) if scheduled by iteration 
eta_min = getattr(config, "eta_min", config.lr * 0.1)
base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=T_max, 
    eta_min=eta_min
)

# optional warmup step
if getattr(config, "warmup_epochs", 0) > 0:
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    
    warmup_steps = config.warmup_epochs # switch to config.warmup_epochs * len(train_loader) if scheduled by iteration 
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, base_scheduler], 
        milestones=[warmup_steps]
    )
else:
    scheduler = base_scheduler

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# eval metrics per epoch for condition-level info based on generated embeddings
def eval_custom_metrics(adata: sc.AnnData, embedding_key: str = "X_scGPT", condition_key: str = "condition"):
    """
    Lightweight, fast evaluation of condition separation metrics.
    Calculates Silhouette Score, PERMANOVA, and distance between cluster centers.
    """
    # 1. Grab embeddings
    embeddings = adata.obsm[embedding_key]
    conditions = adata.obs[condition_key].values
    
    results = {}
    
    # print('computing silhouette')
    sil = silhouette_score(embeddings, conditions, metric="cosine")
    results["Condition_Silhouette"] = sil

    # print('compute l2 distances between healthy vs diabetic clusters')
    centers = {}
    for cond in ['No_Diabetes', 'T1D', 'T2D']:
        centers[cond] = embeddings[conditions == cond].mean(axis=0)
            
    results["L2_No_Diabetes_vs_T1D"] = np.linalg.norm(centers['No_Diabetes'] - centers['T1D'])
    results["L2_No_Diabetes_vs_T2D"] = np.linalg.norm(centers['No_Diabetes'] - centers['T2D'])

    return results


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train one epoch
    """
    model.train()
    total_error = 0.0
    total_loss, total_gepc, total_mse = 0.0, 0.0, 0.0
    
    start_time = time.time()
    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        # celltype_labels = batch_data["celltype_labels"].to(device)
        
        src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if config.DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
            )

            masked_positions = input_values.eq(config.mask_value)  # the postions to predict
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )
            metrics_to_log = {"train/mse": loss_mse.item()}
            if config.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            # if config.GEPC and config.explicit_zero_prob:
            #     loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
            #         output_dict["mvc_zero_probs"], target_values, masked_positions
            #     )
            #     loss = loss + loss_gepc_zero_log_prob
            #     metrics_to_log.update(
            #         {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
            #     )
            # if config.ecs_thres > 0:
            #     loss_ecs = 10 * output_dict["loss_ecs"]
            #     loss = loss + loss_ecs
            #     metrics_to_log.update({"train/ecs": loss_ecs.item()})

            # if config.DAB:
            #     loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            #     loss = loss + config.dab_weight * loss_dab
            #     metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )
    
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if config.GEPC else 0.0
            total_error += mre.item() # error rate
            if batch % config.log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / config.log_interval # execution speed per batch
                cur_loss = total_loss / config.log_interval
                cur_mse = total_mse / config.log_interval
                cur_gepc = total_gepc / config.log_interval if config.GEPC else 0.0
                cur_error = total_error / config.log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                start_time = time.time()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    # total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if config.DSBN else None,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(config.mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                # loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            # total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    return total_loss / total_num, total_error / total_num

def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].toarray()
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    celltypes_labels = adata_t.obs["cell_type"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[config.pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
                batch_labels=torch.from_numpy(batch_ids).long() if config.DSBN else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_custom_metrics(adata_t,)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        
        from matplotlib import rc_context
        with rc_context({'figure.figsize': (4, 4)}):
            fig = sc.pl.umap(
                adata_t,
                color=["condition"],
                title=[
                    f"Diabetes state, silhoutte = {results.get('Condition_Silhouette', 0.0):.4f}",
                ],
                frameon=False,
                return_fig=True,
                show=False,
            )
        results["condition_plot"] = fig

        # with rc_context({'figure.figsize': (4, 4)}):
        #     fig_ct = sc.pl.umap(
        #         adata_t,
        #         color=["cell_type"],
        #         title=[
        #             f"Cell Type, silhoutte = {results.get('CellType_Silhouette', 0.0):.4f}",
        #         ],
        #         frameon=False,
        #         return_fig=True,
        #         show=False,
        #     )
        # results["cell_type_plot"] = fig_ct

    if len(include_types) == 1:
        return results


# ## Train it

best_val_loss = float("inf")
best_model = None

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()

    # get dataset and loader
    train_data_pt, valid_data_pt = prepare_data(
        tokenized_train=tokenized_train, 
        tokenized_valid=tokenized_valid, 
        train_batch_labels=np.zeros(len(tokenized_train['values']), dtype=int),
        valid_batch_labels=np.zeros(len(tokenized_valid['values']), dtype=int),
        config=config,
        epoch=epoch,
        sort_seq_batch=False
    )
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=True,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.eval_batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )

    # run
    train(
        model,
        loader=train_loader,
    )
    val_loss, val_err = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_err:5.4f}"
    )
    logger.info("-" * 89)

    # update
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        if not config.sanity:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        # eval on testdata
        results = eval_testdata(
            best_model,
            adata_t= adata,
            include_types=["cls"],
        )
        results["condition_plot"].savefig(
            save_dir / f"embeddings_condition_umap[cls]_e{best_model_epoch}.png", dpi=100, bbox_inches='tight'
        )

        # results["cell_type_plot"].savefig(
        #     save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=100, bbox_inches='tight'
        # )

        metrics_to_log = {}
        metrics_to_log['best_model_epoch'] = best_model_epoch
        metrics_to_log.update({k: v for k, v in results.items() if k != "condition"})
        
        logger.info(metrics_to_log)
        
    scheduler.step()
