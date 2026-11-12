# diabetes_scGPT_mlm_objective

Fine-tuning [scGPT](https://github.com/bowang-lab/scGPT) on adipocyte single-cell RNA data from diabetic (T1D/T2D) and healthy donors, then using the fine-tuned model's attention weights to propose gene regulatory network (GRN) edges — with a donor-level statistical framework for deciding which proposed edges are real.

## Pipeline

**1. Gene panel + fine-tuning.** Masked gene expression modeling on adipocyte cells only, using an HVG panel selected to retain known adipogenic transcription factors (`src/mlm_finetune_adipocyte_claude.py`, panel diagnostics in `src/check_gene_panel_flavor.py`).

**2. Attention-based GRN extraction.** Per-condition attention matrices from the fine-tuned model, condition-vs-healthy difference matrices, and gene/edge rankings (`src/grn_mlm_adipocyte_claude.py`).

**3. Validation.** Every proposed edge is put through independent checks before being believed:

| script | question it answers |
| --- | --- |
| `src/check_tf_edges.py` | Where do literature-sourced TF–target edges land in the attention matrices? |
| `src/check_de_overlap.py` | Are the flagged genes just differentially expressed? (pseudobulk edgeR comparison) |
| `src/check_de_adipocyte_only_fdr.py` | Was the DE multiple-testing correction computed within-celltype? |
| `src/check_coexpression_shift.py` | Does real gene–gene correlation — no model involved — show the same pattern as attention? |
| `src/check_positive_controls.py` | Do known targets of each TF score as expected, separating "real but stable" from "no signal at all"? |
| `src/check_insilico_perturbation.py` | Per-cell masking probe: zero a gene's input and measure the effect on its partners. |
| `src/profile_candidate_tfs.py`, `src/build_candidate_list.py` | Rank candidate TF–partner edges by agreement across all three methods. |
| `src/validate_zeb1_pparg.py` | Donor-level test, effect size vs. a random-pair null, and age/sex confounder regression on specific candidate edges. |
| `src/donor_aware_rescan.py` | Correlation recomputed *within* each donor, Welch t-test across donors, BH-FDR over all 2,403,528 gene pairs. |

## Results

**The gene panel fix.** The original `seurat_v3` HVG selection was dropping 7 of 10 canonical adipocyte TFs/targets from the 3,000-gene panel, which meant no downstream GRN analysis could test for them at all. Switching to `pearson_residuals` ([Lause/Kobak/Berens 2021](https://doi.org/10.1186/s13059-021-02451-7)) recovers 7/10; force-including `CEBPA` and `SREBF1` reaches 10/10. The refit model (`save/dev_diabetes_mlm_objective_claude_pearsonhvg_1-Oct26-22-11`) trained 20 epochs with validation loss falling monotonically to the final epoch.

**Donor-level statistics are required, and they change the answer.** Attention and co-expression were originally computed on cells *pooled within each condition*, and both appeared to show `PPARG`'s relationships with `LPL`/`PLIN1`/`ADIPOQ` strengthening in diabetes. Pooling lets between-donor variation in mean expression manufacture correlation structure, so the two methods "independently agreeing" was one confound counted twice. Recomputed within each of 59 donors (22 No_Diabetes / 18 T1D / 19 T2D, ≥30 adipocytes each) and compared across donors:

| contrast | edges surviving FDR<0.05 | smallest q |
| --- | --- | --- |
| T1D vs healthy | 0 / 2,403,528 | 0.165 |
| T2D vs healthy | 0 / 2,403,528 | 0.161 |
| diabetic (pooled) vs healthy | 0 / 2,403,528 | 0.095 |

`PPARG`–`LPL`, which the pooled analysis put at 0.42 → 0.57, does not move at donor level (0.405 healthy vs 0.400 T1D, p = 0.93). The per-cell masking probe — the one method that never pooled cells — had disagreed with the pooled result all along, which is now understood as the probe being correct.

**What the data does support.** The baseline relationships are real and strong within individual donors (`PPARG`–`LPL` r ≈ 0.40 in both groups, 98th–100th percentile of the network). The masking probe gives a consistent direction for `PPARG`→targets (5.4× forward vs. reverse). The markers are not differentially expressed (0/10 in T1D, 1/10 in T2D), so none of this is a restatement of ordinary DE. `PPARG`↔`CEBPA` — the central mechanism in the standard adipogenesis model — shows no comparable signal in this data.

**Takeaway.** Pooled-cell scGPT attention analysis produces disease-associated network shifts that do not survive donor-level testing. The validation stack here (per-cell probe + within-donor correlation + FDR across the full edge set) is what separates the two, and it is reusable on any scGPT attention network built from multi-donor data.

Run outputs: `save/grn-claude-pearsonhvg-Oct27-01-05/`, `save/perturb-probe-Nov11-23-16/`, `save/tf-profile-Nov12-00-03/`, `save/zeb1-validation-Nov12-00-21/`.

## Directory layout

- `src/` — training and analysis scripts. `*_claude.py` variants are the gene-panel-fixed versions; the originals are kept for reference.
- `scripts/` — container launch wrappers (`*_launch.sh`), one per script in `src/`. See "Running things" below.
- `data_files/` — raw data (h5ad, metadata, dev/test splits). Not tracked in git — see `.gitignore`.
- `save/` — run outputs. Model weights (`.pt`) and large attention/correlation matrices (`.npz`) are gitignored; logs, HVG selections, ranked gene/edge lists, and plots are tracked.
- `notebooks/` — exploratory notebooks (QC thresholds, SCENIC+ comparison, helpers).

## Running things

This runs on a shared DSMLP research cluster login node — **nothing is executed directly on the login node**. Every script has a matching launcher under `scripts/` that submits it as a Kubernetes job in a container:

```bash
./scripts/mlm_obj_launch_claude.sh -b -v l40s -l gpu-class=large
```

```bash
./scripts/check_de_overlap_launch.sh -b
```

GPU jobs (fine-tuning, GRN extraction, the masking probe) need `-v l40s -l gpu-class=large`; CPU-only checks don't. Job status with `kubectl get pods`, logs with `kubectl logs <pod-name>`, and clean up finished pods with `kubectl delete pod <pod-name>` when done.
