# diabetes_scGPT_mlm_objective

Fine-tuning [scGPT](https://github.com/bowang-lab/scGPT) on adipocyte single-cell RNA data from diabetic (T1D/T2D) and healthy donors, then using the fine-tuned model's attention weights to propose gene regulatory network (GRN) edges that shift between disease states.

## Pipeline

1. **Fine-tune** scGPT (masked gene expression modeling) on adipocyte cells only, using a gene panel selected to retain known adipogenic transcription factors (`src/mlm_finetune_adipocyte_claude.py`).
2. **Extract attention-based GRN signal** from the fine-tuned model: per-condition attention matrices, condition-vs-healthy difference matrices, and gene/edge-level rankings by how much attention shifts with disease state (`src/grn_mlm_adipocyte_claude.py`).
3. **Validate** the attention-based findings against independent evidence before trusting them:
   - `src/check_de_overlap.py` — do the flagged genes show up in existing pseudobulk DE results? (Checks whether the finding is just restating differential expression.)
   - `src/check_coexpression_shift.py` — does real gene-gene correlation (no model involved) show the same shift as attention?
   - `src/check_de_adipocyte_only_fdr.py` — sanity-checks the DE multiple-testing correction.
   - `src/check_tf_edges.py` — looks up specific literature-sourced TF-target edges in the attention/difference matrices.

## Key finding so far

Switching gene panel selection from `seurat_v3` to `pearson_residuals` HVG flavor raised known-marker survival (10 canonical adipocyte TFs/targets) from 3/10 to 10/10 in the fine-tuning gene panel. On the refit model, `PPARG`'s attention to its canonical lipid/adipokine targets (`LPL`, `PLIN1`, `ADIPOQ`) are among the most disease-shifted edges in the entire attention network — a finding that holds up independently in real gene-gene correlation, and isn't explained by those genes simply being differentially expressed. The `PPARG`-`CEBPA` TF-TF cross-regulation loop, by contrast, shows no comparable signal. See `save/grn-claude-pearsonhvg-*/` for the underlying run outputs.

In-silico perturbation (a natural next step toward causal validation) was scoped and found **not feasible** with the current data — it requires a real experimental perturbation dataset (control vs. CRISPR-perturbed cells), which this project doesn't have. The sibling `obesity_scGPT_perturbation_prediction` project does have one, if that infrastructure is ever reused here.

## Directory layout

- `src/` — training and analysis scripts. `*_claude.py` variants are the gene-panel-fixed versions; the originals are kept for reference.
- `scripts/` — container launch wrappers (`*_launch.sh`), one per script in `src/`. See "Running things" below.
- `data_files/` — raw data (h5ad, metadata, dev/test splits). **Not tracked in git** (too large) — see `.gitignore`.
- `save/` — run outputs. Model weights (`.pt`) and large attention/correlation matrices (`.npz`) are gitignored; logs, HVG selections, ranked gene lists, and plots are tracked.
- `notebooks/` — exploratory notebooks (QC, SCENIC comparison, etc).

## Running things

This runs on a shared DSMLP research cluster login node — **nothing gets executed directly on the login node**. Every script has a matching launcher under `scripts/` that submits it as a Kubernetes job in a container:

```bash
./scripts/mlm_obj_launch_claude.sh -b -v l40s -l gpu-class=large   # large/GPU jobs (fine-tuning, GRN extraction)
./scripts/check_de_overlap_launch.sh -b                            # small/CPU-only jobs (no -v/-l needed)
```

Check job status with `kubectl get pods`, logs with `kubectl logs <pod-name>`, and clean up finished pods with `kubectl delete pod <pod-name>` when done.

## Known limitations

- Validated edges are literature-picked (10 markers, 8 edges), not from a systematic scan — a genuinely novel candidate GRN edge hasn't been proposed yet.
- T1D and T2D results are noticeably asymmetric in several checks; not yet explained.
- The SCENIC comparison in `notebooks/attention_vs_scenic_comparison.ipynb` uses an all-celltype network and isn't a fair comparison to this adipocyte-only model; the co-expression check above is used in its place.
- Attention and real correlation both support the `PPARG`-target finding, but neither is causal evidence — perturbation data would be needed for that, and isn't available here (see above).
