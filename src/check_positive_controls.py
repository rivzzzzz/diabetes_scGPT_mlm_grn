#!/usr/bin/env python
"""Positive-control test: known targets of MLXIPL/ZEB1/RORA did NOT appear in the
strict partner profiles. Two possible reasons, and they mean opposite things:

  (a) the metric ranks by SHIFT, and known targets are stably co-expressed
      without changing between conditions -> method is fine, metric just
      answers a different question
  (b) known targets show neither strong baseline correlation nor shift
      -> the method is failing on these TFs and the partners it DID return
      should not be trusted

This prints, for each known target, its baseline correlation AND its shift
percentile, so (a) and (b) can be told apart.
"""
import json
from pathlib import Path

import numpy as np

GRN_DIR = Path("diabetes_scGPT_mlm_objective/save/grn-claude-pearsonhvg-Oct27-01-05")
COEXP_DIR = GRN_DIR / "coexpression"

# literature-established targets, per the domain consult
KNOWN = {
    "MLXIPL": ["TXNIP", "FASN", "ACACA", "ACLY", "ELOVL6", "SCD", "PKLR"],
    "ZEB1": ["PPARG", "CEBPA", "CEBPB", "CEBPD"],
    "RORA": ["ARNTL", "NR1D1", "NR1D2", "PER2", "CRY1"],      # circadian, RORA's best-known targets
    "PPARG": ["FABP4", "CD36", "PLIN1", "LPL", "ADIPOQ"],      # reference: known to work
}

with open(COEXP_DIR / "gene_panel.json") as f:
    corr_panel = json.load(f)
with open(GRN_DIR / "gene_panel.json") as f:
    attn_panel = json.load(f)
ci = {g: i for i, g in enumerate(corr_panel)}
ai = {g: i for i, g in enumerate(attn_panel)}

corr_raw = np.load(COEXP_DIR / "correlation_matrices.npz")
corr_diff = np.load(COEXP_DIR / "correlation_difference_matrices.npz")
attn_diff = np.load(GRN_DIR / "difference_matrices.npz")


def offdiag(m):
    return m[~np.eye(m.shape[0], dtype=bool)]


corr_bg = {k: np.sort(np.abs(offdiag(corr_diff[k]))) for k in corr_diff.files}
attn_bg = {k: np.sort(np.abs(offdiag(attn_diff[k]))) for k in attn_diff.files}
# baseline |r| background, to say whether a baseline correlation is itself notable
base_bg = np.sort(np.abs(offdiag(corr_raw["No_Diabetes"])))


def pctile(v, bg):
    return float(np.searchsorted(bg, abs(v)) / len(bg))


for tf, targets in KNOWN.items():
    print(f"\n{'=' * 95}")
    print(f"{tf}  -- known targets as positive controls")
    print(f"{'=' * 95}")
    if tf not in ci:
        print(f"  {tf} not in panel")
        continue
    print(f"{'target':<10} {'in panel':<9} {'r_hlth':>7} {'r_T1D':>7} {'r_T2D':>7} "
          f"{'|r| pct':>8} {'corrShift pct':>14} {'attnShift pct':>14}")
    for t in targets:
        if t not in ci or t not in ai:
            print(f"{t:<10} {'NO':<9}  -- not in gene panel, cannot assess --")
            continue
        a, b = ci[tf], ci[t]
        r_h = corr_raw["No_Diabetes"][a, b]
        r_1 = corr_raw["T1D"][a, b]
        r_2 = corr_raw["T2D"][a, b]
        base_pct = pctile(r_h, base_bg)
        cshift = min(pctile(corr_diff["diff_T1D"][a, b], corr_bg["diff_T1D"]),
                     pctile(corr_diff["diff_T2D"][a, b], corr_bg["diff_T2D"]))
        x, y = ai[tf], ai[t]
        ashift = min(
            pctile(max(abs(attn_diff["diff_T1D"][x, y]), abs(attn_diff["diff_T1D"][y, x])), attn_bg["diff_T1D"]),
            pctile(max(abs(attn_diff["diff_T2D"][x, y]), abs(attn_diff["diff_T2D"][y, x])), attn_bg["diff_T2D"]),
        )
        print(f"{t:<10} {'yes':<9} {r_h:>7.2f} {r_1:>7.2f} {r_2:>7.2f} "
              f"{base_pct:>8.0%} {cshift:>14.0%} {ashift:>14.0%}")

print("\nReading guide:")
print("  high |r| pct + low shift pct  -> relationship is real and STABLE; metric (which ranks")
print("                                   change) correctly ignores it. Method is fine.")
print("  low  |r| pct + low shift pct  -> no signal at all for a known edge; method is failing")
print("                                   for that TF and its reported partners are suspect.")
