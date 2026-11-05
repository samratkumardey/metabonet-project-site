#!/usr/bin/env python3
"""Render every static figure on the project page into static/images/.

    python tools/make_figures.py

Numbers come from two places only:
  * static/tables/*.csv  (results CSV and the patient-atlas tables), read here;
  * the measured constants in the MEASURED block below, each with the analysis
    it came from. They are copied from the project's run logs, not re-derived.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "static" / "tables"
OUT = ROOT / "static" / "images"
OUT.mkdir(parents=True, exist_ok=True)
H = [30, 60, 90, 120]

INK, MUTED, RULE = "#1f2937", "#6b7280", "#e5e7eb"
C_LSTM, C_GBM, C_BLEND, C_PERS, C_CAUS = "#0e7490", "#b45309", "#6d28d9", "#9ca3af", "#be123c"
C_POS, C_NEG = "#15803d", "#b91c1c"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": RULE,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#f1f5f9", "grid.linewidth": 1, "legend.frameon": False,
    "savefig.dpi": 200, "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", name)


res = pd.read_csv(TAB / "metabonet_results.csv")
live = res[res.split == "public test"]
loso = res[res.split == "held-out study"]


def series(df, model, metric):
    g = df[df.model == model].set_index("horizon")[metric]
    return [float(g[h]) for h in H]


# --------------------------------------------------------------- MEASURED ---
# Persistence on the live anchors (modelling-strategy analysis, 18 Sep 2026)
PERSIST = {"dtsa": [79.92, 61.34, 51.81, 46.31], "mard": [13.12, 21.31, 26.62, 30.26]}
# Template share by study (submission-contract analysis, 17 Sep 2026)
SHARE = [("Loop", 76.84), ("ReplaceBG", 13.47), ("IOBP2", 5.53), ("PEDAP", 2.39),
         ("BrisT1D", 0.74), ("CTR3", 0.40), ("HUPA-UCM", 0.34), ("AZT1D", 0.29),
         ("ShanghaiT1DM", 0.01)]
# LSTM - LightGBM, average DTS-A, TEST_COHORT, paired subject bootstrap (37_xcohort_table.py)
GAP = [("PEDAP", 2.09, 1.62, 2.64), ("IOBP2", 1.83, 1.68, 1.97),
       ("ReplaceBG", 1.33, 1.17, 1.49), ("CTR3", 1.12, 0.46, 1.83),
       ("Loop", -0.74, -0.97, -0.51)]
# blend50 vs each parent (38_blend.py)
BLEND_VS_GBM = [("CTR3", 1.59, 1.23, 1.98), ("IOBP2", 2.37, 2.28, 2.46),
                ("Loop", 1.39, 1.27, 1.51), ("PEDAP", 2.29, 2.00, 2.63),
                ("ReplaceBG", 1.45, 1.37, 1.54)]
BLEND_VS_LSTM = [("CTR3", 0.47, 0.11, 0.83), ("IOBP2", 0.54, 0.47, 0.61),
                 ("Loop", 2.13, 1.99, 2.27), ("PEDAP", 0.20, -0.04, 0.40),
                 ("ReplaceBG", 0.12, 0.04, 0.21)]
# Tabular feature-family ledger (LightGBM pipeline, DTS-A gain)
LEDGER = [("Known future insulin & carbs (what-if)", 4.52),
          ("Time-to-next-event + 10-min future bins", 0.53),
          ("Per-subject bias correction", 0.32),
          ("Capacity: 511 leaves, lr 0.02", 0.14),
          ("Two-compartment PK horizon block", 0.08),
          ("Longer training 6k -> 12k rounds", 0.06),
          ("Anchor-level PK state", 0.005),
          ("Equal-weight blend of 17 tabular models", -0.12),
          ("Band-centred point selection", -0.24)]
# Mean glucose change anchor -> t+120 (modelling-strategy analysis)
INFO = [("Carbs in next 2 h", 10.89), ("No carbs", -5.50),
        ("Bolus in next 2 h", 10.76), ("No bolus", -8.83)]
# Persistence at t+60 by reference band (modelling-strategy analysis)
BANDS = ["<54", "54-70", "70-180", "180-250", ">250"]
BAND_ROWS = [0.45, 2.68, 75.44, 16.23, 5.20]
BAND_MARD = [2.15, 6.62, 72.29, 14.55, 4.38]
BAND_DTSA = [16.86, 29.01, 62.39, 61.84, 65.01]
# Calibration on live_lstmseed6, binned by prediction (54_calibration_check.py)
CAL = {"mean": [0.38, 0.89, 1.34, 1.80], "min": [0.07, 0.34, 0.33, 0.33],
       "max": [0.53, 1.06, 1.59, 2.21]}
CAL_DTSA = [(94.07, 93.99), (86.43, 86.21), (81.37, 81.01), (76.91, 76.37)]
# GBM baseline ladder, subject-disjoint 5-fold CV on train (run baseline_20260918)
LADDER = [("Persistence", 23.56, 58.70), ("Damped trend", 23.33, 59.29),
          ("Ridge", 21.72, 61.05), ("LightGBM", 19.59, 66.26)]
# Seed spread across 3 seeds on the annual anchors, mg/dL
SPREAD = [2.34, 3.49, 4.29, 4.98]
# Regime: LSTM - blend50 on live patients seen / unseen in training (42_live_by_overlap.py)
REGIME = [("Seen in training\n(128 patients)", 0.280, 0.149, 0.409),
          ("Never seen\n(151 patients)", 0.448, 0.359, 0.539)]


# 1 ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
names = [s for s, _ in SHARE][::-1]
vals = [v for _, v in SHARE][::-1]
cols = [C_LSTM if n == "Loop" else "#94a3b8" for n in names]
ax.barh(names, vals, color=cols, height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.8, i, f"{v:.2f}%", va="center", fontsize=9, color=INK)
ax.set_xlim(0, 88)
ax.set_xlabel("share of the 2,648,987 scored anchors (%)")
ax.set_title("The live score is three-quarters Loop", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_study_share.png")

# 2 ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
spec = [("Persistence", None, C_PERS, "--"),
        ("LightGBM, no future inputs", "live_causal", C_CAUS, ":"),
        ("LightGBM what-if", "live_whatif", C_GBM, "-"),
        ("Blend (6-seed LSTM + LightGBM)", "live_blendseed6", C_BLEND, "-"),
        ("LSTM, 6-seed mean", "live_lstmseed6", C_LSTM, "-")]
for ax, metric, lab in [(axes[0], "dtsa", "DTS zone A (%)  - higher is better"),
                        (axes[1], "mard", "MARD (%)  - lower is better")]:
    for name, model, col, ls in spec:
        y = PERSIST[metric] if model is None else series(live, model, metric)
        ax.plot(H, y, ls, color=col, marker="o", ms=4.5, lw=2.2 if model == "live_lstmseed6" else 1.6, label=name)
    ax.set_xticks(H)
    ax.set_xlabel("prediction horizon (min)")
    ax.set_ylabel(lab)
axes[0].set_title("Public test set, per horizon", loc="left")
axes[1].legend(loc="upper left", fontsize=8.6)
save(fig, "fig_live_horizons.png")

# 3 ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=False)


def forest(ax, rows, title, col):
    for i, (n, g, lo, hi) in enumerate(rows):
        c = C_POS if lo > 0 else (C_NEG if hi < 0 else MUTED)
        ax.plot([lo, hi], [i, i], color=c, lw=2.4, solid_capstyle="round")
        ax.plot(g, i, "o", color=c, ms=7)
        ax.text(hi + 0.08, i, f"{g:+.2f}", va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)


forest(axes[0], GAP, "LSTM minus LightGBM (DTS-A points)", C_LSTM)
axes[0].set_xlabel("paired 95% CI over subjects")
axes[0].set_xlim(-1.5, 3.2)
forest(axes[1], [(f"{a[0]}", a[1], a[2], a[3]) for a in BLEND_VS_LSTM],
       "Blend50 minus LSTM (DTS-A points)", C_BLEND)
axes[1].set_xlabel("paired 95% CI over subjects")
axes[1].set_xlim(-0.5, 2.8)
save(fig, "fig_loso_forest.png")

# 4 ------------------------------------------------------------------------
variants = [("dual_s0", "Two-stream encoder"), ("tcn_s0", "TCN encoder"),
            ("transformer_s0", "Transformer encoder"),
            ("lstm_e50_s0", "50-epoch cosine schedule"),
            ("lstm_dz25_s0", "DTS-A checkpoint selection, 25 ep"),
            ("lstm_dz50_s0", "DTS-A checkpoint selection, 50 ep"),
            ("lstm_futseq_s0", "GRU decoder over known future")]
folds = ["CTR3", "IOBP2", "Loop", "PEDAP", "ReplaceBG"]
mean_dtsa = loso.groupby(["fold", "model"]).dtsa.mean()
M = np.array([[mean_dtsa.get((f, v), np.nan) - mean_dtsa[(f, "lstm_s0")] for f in folds]
              for v, _ in variants])
# lstm_futseq_s0 finished on IOBP2 and PEDAP after the results CSV was written.
# Their four-horizon means come from 37_xcohort_table.py (79.16 and 70.83).
FUTSEQ_LATE = {"IOBP2": 79.16, "PEDAP": 70.83}
r = [v for v, _ in variants].index("lstm_futseq_s0")
for f, val in FUTSEQ_LATE.items():
    j = folds.index(f)
    if np.isnan(M[r, j]):
        M[r, j] = val - mean_dtsa[(f, "lstm_s0")]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
im = ax.imshow(M, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(M[i, j]) > 1.8 else INK)
ax.set_xticks(range(len(folds)))
ax.set_xticklabels(folds)
ax.set_yticks(range(len(variants)))
ax.set_yticklabels([f"{lab}  ({np.nanmean(M[i]):+.2f})" for i, (_, lab) in enumerate(variants)])
ax.grid(False)
ax.set_title("Single-change screens vs plain LSTM (held-out study, mean DTS-A)", loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("DTS-A difference")
save(fig, "fig_screens_heatmap.png")

# 5 ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 3.9))
labs = [a for a, _ in LEDGER][::-1]
vals = [b for _, b in LEDGER][::-1]
ax.barh(labs, vals, color=[C_POS if v > 0 else C_NEG for v in vals], height=0.6)
for i, v in enumerate(vals):
    ax.text(v + (0.06 if v >= 0 else -0.06), i, f"{v:+.2f}" if abs(v) >= 0.01 else f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9, color=INK)
ax.axvline(0, color=INK, lw=1)
ax.set_xlim(-0.8, 5.3)
ax.set_xlabel("measured DTS-A gain (points)")
ax.set_title("One idea in nine paid off: the known-future block", loc="left")
ax.grid(axis="y", visible=False)
save(fig, "fig_feature_ledger.png")

# 6 ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labs = [a for a, _ in INFO]
vals = [b for _, b in INFO]
ax.bar(labs, vals, color=[C_GBM, "#d6d3d1", C_LSTM, "#d6d3d1"], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + (0.6 if v > 0 else -1.6), f"{v:+.2f}", ha="center", fontsize=9.5, color=INK)
ax.axhline(0, color=INK, lw=1)
ax.set_ylabel("mean glucose change\nanchor -> t+120 (mg/dL)")
ax.set_ylim(-12, 14)
ax.set_title("What is already written down about the next two hours", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "fig_known_future.png")

# 7 ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 3.6))
x = np.arange(len(BANDS))
w = 0.38
ax.bar(x - w / 2, BAND_ROWS, w, color="#cbd5e1", label="share of readings")
ax.bar(x + w / 2, BAND_MARD, w, color=C_LSTM, label="share of total MARD")
ax2 = ax.twinx()
ax2.plot(x, BAND_DTSA, "o-", color=C_NEG, label="DTS zone A of persistence")
ax2.set_ylim(0, 100)
ax2.set_ylabel("zone A (%)", color=C_NEG)
ax2.spines["right"].set_visible(True)
ax2.grid(False)
ax.set_xticks(x)
ax.set_xticklabels([b + " mg/dL" for b in BANDS])
ax.set_ylabel("% of all")
ax.set_title("In-range error carries the metric; lows carry the clinical risk", loc="left")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.6)
ax.grid(axis="x", visible=False)
save(fig, "fig_error_bands.png")

# 8 ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
ax = axes[0]
ax.vlines(H, CAL["min"], CAL["max"], color=C_LSTM, lw=6, alpha=0.35, label="range across 20 bins")
ax.plot(H, CAL["mean"], "o-", color=C_LSTM, label="mean residual")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(H)
ax.set_xlabel("horizon (min)")
ax.set_ylabel("truth - prediction (mg/dL)")
ax.set_ylim(-1, 3)
ax.set_title("Residual by prediction level: a flat offset", loc="left")
ax.legend(fontsize=8.6, loc="upper left")
ax = axes[1]
d = [b - a for a, b in CAL_DTSA]
ax.bar([str(h) for h in H], d, color=C_NEG, width=0.55)
for i, v in enumerate(d):
    ax.text(i, v - 0.04, f"{v:+.2f}", ha="center", va="top", fontsize=9, color=INK)
ax.axhline(0, color=INK, lw=1)
ax.set_ylim(-0.7, 0.1)
ax.set_xlabel("horizon (min)")
ax.set_ylabel("DTS-A change")
ax.set_title("Correcting it loses, even in-sample", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "fig_calibration.png")

# 9 ------------------------------------------------------------------------
cb = pd.read_csv(TAB / "60_consensus_bands.csv").set_index("split")
fig, ax = plt.subplots(figsize=(8.2, 2.5))
segs = [("<55", "pct_below_55", "#7f1d1d"), ("55-70", None, "#ef4444"),
        ("70-180", "pct_in_range_70_180", "#16a34a"), ("180-250", None, "#f59e0b"),
        (">250", "pct_above_250", "#b45309")]
for row, split in enumerate(["train", "test"]):
    r = cb.loc[split]
    vals = [r.pct_below_55, r.pct_below_70 - r.pct_below_55, r.pct_in_range_70_180,
            r.pct_above_180 - r.pct_above_250, r.pct_above_250]
    left = 0
    for (lab, _, col), v in zip(segs, vals):
        ax.barh(row, v, left=left, color=col, height=0.55, label=lab if row == 0 else None)
        if v > 4:
            ax.text(left + v / 2, row, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=9)
        left += v
ax.set_yticks([0, 1])
ax.set_yticklabels([f"train  ({int(cb.loc['train'].cgm_readings):,} readings)",
                    f"test  ({int(cb.loc['test'].cgm_readings):,} readings)"])
ax.set_xlim(0, 100)
ax.set_xlabel("% of CGM readings by consensus glucose band (mg/dL)")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.38), fontsize=8.6)
ax.grid(False)
ax.set_title("Corpus glycemic profile", loc="left")
save(fig, "fig_glycemic_bands.png")

# 10 -----------------------------------------------------------------------
pc = pd.read_csv(TAB / "66_pump_population_coverage.csv")
pc = pc[(pc.split == "test") & (pc.subjects >= 5)].sort_values("subjects", ascending=True)
fig, ax = plt.subplots(figsize=(8.2, 4.6))
y = np.arange(len(pc))
ax.barh(y - 0.2, pc.carbs, 0.38, color=C_GBM, label="carbs recorded")
ax.barh(y + 0.2, pc.meal_label, 0.38, color=C_LSTM, label="meal_label recorded")
ax.set_yticks(y)
ax.set_yticklabels([f"{p}  (n={s})" for p, s in zip(pc.pump, pc.subjects)])
ax.set_xlim(0, 105)
ax.set_xlabel("% of test subjects on that pump with any value")
ax.set_title("The pump decides whether food data exists", loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.6)
ax.set_title("The pump decides whether food data exists", loc="left", pad=26)
ax.grid(axis="y", visible=False)
save(fig, "fig_food_by_pump.png")

# 11 -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
ax = axes[0]
seeds = ["live_lstm", "live_lstm_s1", "live_lstm_s2", "live_lstm_s3", "live_lstm_s4", "live_lstm_s5"]
sv = [np.mean(series(live, s, "dtsa")) for s in seeds]
avg = np.mean(series(live, "live_lstmseed6", "dtsa"))
ax.bar([f"s{i}" for i in range(6)], sv, color="#94a3b8", width=0.6)
ax.bar(["mean of 6"], [avg], color=C_LSTM, width=0.6)
for i, v in enumerate(sv + [avg]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.6, color=INK)
ax.set_ylim(84.0, 84.85)
ax.set_xlabel("single seeds  (y-axis starts at 84.0)")
ax.set_ylabel("mean DTS-A, public test (%)")
ax.set_title("Averaging seeds beats any single seed", loc="left")
ax.grid(axis="x", visible=False)
ax = axes[1]
ax.bar([str(h) for h in H], SPREAD, color=C_LSTM, width=0.55)
for i, v in enumerate(SPREAD):
    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9, color=INK)
ax.set_xlabel("horizon (min)")
ax.set_ylabel("mean spread across 3 seeds (mg/dL)")
ax.set_title("What one seed carries as noise (annual anchors)", loc="left")
ax.grid(axis="x", visible=False)
save(fig, "fig_seeds.png")

# 12 -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2))
names = [a for a, _, _ in LADDER]
axes[0].bar(names, [b for _, b, _ in LADDER], color=["#cbd5e1"] * 3 + [C_GBM], width=0.6)
axes[0].set_ylim(18, 24.5)
axes[0].set_ylabel("MARD (%)")
axes[0].set_title("Model ladder, subject-disjoint CV", loc="left")
axes[1].bar(names, [c for _, _, c in LADDER], color=["#cbd5e1"] * 3 + [C_GBM], width=0.6)
axes[1].set_ylim(55, 68)
axes[1].set_ylabel("DTS zone A (%)")
for ax, k in [(axes[0], 1), (axes[1], 2)]:
    for i, r in enumerate(LADDER):
        ax.text(i, r[k] + (0.08 if k == 1 else 0.2), f"{r[k]:.2f}", ha="center", fontsize=9, color=INK)
    ax.grid(axis="x", visible=False)
save(fig, "fig_ladder.png")

# 13 -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
forest(axes[0], REGIME, "LSTM minus blend, live patients", C_LSTM)
axes[0].set_xlim(-0.2, 0.75)
axes[0].set_ylim(1.7, -0.7)
axes[0].set_xlabel("DTS-A points, paired 95% CI")
gap = [a - b for a, b in zip(series(live, "live_lstmseed6", "dtsa"), series(live, "live_blendseed6", "dtsa"))]
axes[1].bar([str(h) for h in H], gap, color=C_LSTM, width=0.55)
for i, v in enumerate(gap):
    axes[1].text(i, v + 0.02, f"{v:+.2f}", ha="center", fontsize=9, color=INK)
axes[1].set_xlabel("horizon (min)")
axes[1].set_ylabel("DTS-A points")
axes[1].set_title("Pure LSTM minus blend, public test (6 seeds)", loc="left")
axes[1].set_ylim(0, 1.4)
axes[1].grid(axis="x", visible=False)
save(fig, "fig_regime.png")

print("done")
