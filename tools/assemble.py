#!/usr/bin/env python3
"""Assemble index.html from tools/parts/*.html, filling {{TABLE}} slots from CSVs.

    python tools/build_data.py && python tools/make_figures.py && python tools/assemble.py
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "static" / "tables"
PARTS = sorted((ROOT / "tools" / "parts").glob("*.html"))
H = [30, 60, 90, 120]


def table(head, rows, num_from=1, hl=None, cls=None):
    hl = hl or set()
    th = "".join(f'<th class="num">{h}</th>' if i >= num_from else f"<th>{h}</th>"
                 for i, h in enumerate(head))
    body = []
    for k, r in enumerate(rows):
        tds = "".join(f'<td class="num">{c}</td>' if i >= num_from else f"<td>{c}</td>"
                      for i, c in enumerate(r))
        c = ' class="hl"' if k in hl else ""
        body.append(f"<tr{c}>{tds}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# Live-template composition, measured from template.parquet (submission-contract analysis)
TEMPLATE = {"Loop": (2035382, 76.84, 159), "ReplaceBG": (356829, 13.47, 35),
            "IOBP2": (146525, 5.53, 62), "PEDAP": (63213, 2.39, 6),
            "BrisT1D": (19530, 0.74, 2), "CTR3": (10520, 0.40, 5),
            "HUPA-UCM": (8957, 0.34, 4), "AZT1D": (7750, 0.29, 4),
            "ShanghaiT1DM": (281, 0.01, 2)}


def study_table():
    inv = pd.read_csv(TAB / "63_stratum_inventory.csv")
    s = inv[inv.family == "study"]
    rows = []
    for _, r in s.iterrows():
        a, sh, sub = TEMPLATE.get(r.level, (0, 0.0, 0))
        rows.append([f"<b>{r.level}</b>" if r.level == "Loop" else r.level,
                     f"{int(r.subjects_train):,}", f"{int(r.subjects_test):,}",
                     f"{sub:,}" if a else "—", f"{a:,}" if a else "—",
                     f"{sh:.2f}%" if a else "0%"])
    tr, te = int(s.subjects_train.sum()), int(s.subjects_test.sum())
    rows.append(["<b>Total (subject–study pairs)</b>", f"<b>{tr:,}</b>", f"<b>{te:,}</b>",
                 "<b>279</b>", "<b>2,648,987</b>", "<b>100%</b>"])
    return table(["Study", "Subjects, train", "Subjects, test", "Scored subjects",
                  "Scored anchors", "Share of score"], rows)


def device_table(family):
    inv = pd.read_csv(TAB / "63_stratum_inventory.csv")
    s = inv[inv.family == family]
    rows = [[html.escape(r.level), f"{int(r.subjects_train):,}", f"{int(r.subjects_test):,}",
             "train only" if r.presence == "train only" else "both"] for _, r in s.iterrows()]
    return table(["Sensor" if family == "sensor" else "Insulin delivery device",
                  "Train", "Test", "Present in"], rows)


def coverage_table():
    c = pd.read_csv(TAB / "61_column_coverage.csv").sort_values("pct_rows_filled_test", ascending=False)
    rows = [[f"<code>{r.column}</code>", f"{r.pct_rows_filled_train:.2f}", f"{r.pct_rows_filled_test:.2f}",
             f"{r.pct_subjects_train:.1f}", f"{r.pct_subjects_test:.1f}", html.escape(r.verdict)]
            for _, r in c.iterrows()]
    return table(["Column", "% rows, train", "% rows, test", "% subjects, train",
                  "% subjects, test", "Verdict"], rows).replace(
        '<td class="num">EMPTY', '<td class="num neg">EMPTY')


def best_table():
    d = pd.read_csv(TAB / "metabonet_results.csv")
    m = d.groupby(["split", "fold", "model"]).dtsa.mean()
    order = [("public test", "live", "Public test set"), ("held-out study", "CTR3", "CTR3"),
             ("held-out study", "IOBP2", "IOBP2"), ("held-out study", "Loop", "Loop"),
             ("held-out study", "PEDAP", "PEDAP"), ("held-out study", "ReplaceBG", "ReplaceBG")]
    rows = []
    for split, fold, lab in order:
        best = m.loc[(split, fold)].idxmax()
        g = d[(d.split == split) & (d.fold == fold) & (d.model == best)].set_index("horizon")
        dt = [g.loc[h, "dtsa"] for h in H]
        rows.append([f"<b>{lab}</b>" if split == "public test" else lab,
                     f'<span class="mname">{best}</span>',
                     *[f"{v:.2f}" for v in dt], f"<b>{sum(dt) / 4:.2f}</b>",
                     f"{g.mard.mean():.2f}", f"{g.rmse.mean():.2f}", f"{g.mse.mean():.1f}",
                     f"{int(g.n.iloc[0]):,}"])
    return table(["Test set", "Best model", "DTS-A 30", "60", "90", "120", "Mean DTS-A",
                  "Mean MARD", "Mean RMSE", "Mean MSE", "Anchors"], rows, num_from=2, hl={0})


def live_sub_table():
    d = pd.read_csv(TAB / "metabonet_results.csv")
    d = d[d.split == "public test"]
    recipe = {
        "live_lstmseed6": "Mean of 6 LSTM seeds", "live_blendseed6": "0.5 × 6-seed LSTM + 0.5 × LightGBM",
        "live_blendseed": "0.5 × 3-seed LSTM + 0.5 × LightGBM", "live_blend50": "0.5 × LSTM seed 0 + 0.5 × LightGBM",
        "live_whatif": "LightGBM what-if (240 features)", "live_causal": "LightGBM, past only",
        "live_lstm": "LSTM seed 0",
    }
    confirmed = {"live_lstmseed6", "live_blendseed6", "live_blendseed", "live_whatif"}
    rows = []
    for mname in recipe:
        g = d[d.model == mname].set_index("horizon")
        dt = [g.loc[h, "dtsa"] for h in H]
        rows.append([f'<span class="mname">{mname}</span>', recipe[mname],
                     *[f"{v:.2f}" for v in dt], f"<b>{sum(dt) / 4:.2f}</b>", f"{g.mard.mean():.2f}",
                     "✓" if mname in confirmed else ""])
    rows.insert(3, ['<span class="mname">live_lstmseed</span>', "Mean of 3 LSTM seeds",
                    "94.03", "86.31", "81.22", "76.77", "<b>84.58</b>", "10.95", "✓"])
    return table(["Entry", "Recipe", "30", "60", "90", "120", "Mean DTS-A", "Mean MARD", "Board"],
                 rows, num_from=2, hl={0})


def main():
    out = "".join(p.read_text() for p in PARTS)
    slots = {
        "{{STUDY_TABLE}}": study_table(),
        "{{SENSOR_TABLE}}": device_table("sensor"),
        "{{PUMP_TABLE}}": device_table("pump"),
        "{{COVERAGE_TABLE}}": coverage_table(),
        "{{BEST_TABLE}}": best_table(),
        "{{LIVE_SUB_TABLE}}": live_sub_table(),
    }
    for k, v in slots.items():
        assert k in out, k
        out = out.replace(k, v)
    assert "{{" not in out
    (ROOT / "index.html").write_text(out)
    print("wrote index.html", len(out), "bytes")


if __name__ == "__main__":
    main()
