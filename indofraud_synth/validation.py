"""Aggregate calibration checks: generated data vs documented targets."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .config import SynthConfig


def calibration_report(tx: pd.DataFrame, cfg: SynthConfig) -> dict:
    rail_names = tx.attrs.get("rail_names", list(cfg.rails.keys()))
    checks = []

    def add(name, actual, target, tol, kind="relative"):
        if kind == "relative":
            err = abs(actual - target) / max(abs(target), 1e-9)
            ok = err <= tol
        else:  # range
            ok = target[0] <= actual <= target[1]
            err = None
        checks.append({"check": name, "actual": float(actual),
                       "target": target if kind == "relative" else list(target),
                       "tolerance": tol, "rel_error": err, "ok": bool(ok)})

    n = len(tx)
    for i, r in enumerate(rail_names):
        sub = tx[tx.rail == i]
        if len(sub) == 0:
            continue
        add(f"rail_share[{r}]", len(sub) / n, cfg.rails[r]["share"], 0.25)
        add(f"mean_ticket[{r}]", sub.amount.mean(), cfg.rails[r]["mean"], 0.15)
        add(f"fraud_rate[{r}]", sub.is_fraud.mean(), cfg.fraud_rates.get(r, 0.0), 0.40)

    # seasonality presence checks
    daily = tx.groupby((tx.ts - tx.ts.min()) // 86400).size()
    days = pd.date_range(pd.Timestamp(cfg.start_date), periods=cfg.horizon_days, freq="D")
    dom = days.day.to_numpy()[: len(daily)]
    payday_mask = np.isin(dom, [25, 26, 27, 28, 29, 30, 31, 1])
    if payday_mask.any() and (~payday_mask).any():
        uplift = daily.to_numpy()[payday_mask].mean() / daily.to_numpy()[~payday_mask].mean()
        add("payday_uplift", uplift, (1.15, 2.6), None, kind="range")
    promo_days = pd.to_datetime(cfg.promo_days)
    p_idx = [i for i, d in enumerate(days[: len(daily)]) if d in promo_days]
    if p_idx:
        base = np.delete(daily.to_numpy(), p_idx).mean()
        add("promo_uplift", daily.to_numpy()[p_idx].mean() / base, (1.4, 6.0), None,
            kind="range")

    hour_counts = tx.groupby("hour").size()
    peak = int(hour_counts.idxmax())
    checks.append({"check": "diurnal_peak_hour", "actual": peak,
                   "target": [10, 21], "ok": bool(10 <= peak <= 21)})

    silo_sizes = tx.groupby("silo_id").size().sort_values(ascending=False)
    skew = float(silo_sizes.iloc[0] / max(silo_sizes.iloc[-1], 1))
    checks.append({"check": "silo_max_min_ratio", "actual": skew,
                   "target": [3.0, 1e9], "ok": bool(skew >= 3.0)})

    # Behavioral realism: regional affinity makes legit region_mismatch lower than fraud
    legit_rm = float(tx.loc[tx.is_fraud == 0, "region_mismatch"].mean()) if (tx.is_fraud == 0).any() else 0.0
    fraud_rm = float(tx.loc[tx.is_fraud == 1, "region_mismatch"].mean()) if (tx.is_fraud == 1).any() else 0.0
    rm_gap = fraud_rm - legit_rm
    checks.append({"check": "region_mismatch_gap", "actual": rm_gap,
                   "target": [0.15, 1.0], "ok": bool(rm_gap >= 0.15)})

    # 70 / 15 / 15 chronological split typology audit
    t1 = tx.ts.quantile(0.70)
    t2 = tx.ts.quantile(0.85)
    sp = np.where(tx.ts <= t1, "train", np.where(tx.ts <= t2, "val", "test"))
    split_typ = {}
    for sp_name in ("train", "val", "test"):
        sub_sp = tx[sp == sp_name]
        split_typ[sp_name] = {k: int(v) for k, v in sub_sp.typology.value_counts().items()}

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks,
            "silo_sizes": {str(k): int(v) for k, v in silo_sizes.items()},
            "typology_counts": {k: int(v) for k, v in tx.typology.value_counts().items()},
            "split_typology_counts": split_typ}


def report_markdown(report: dict) -> str:
    lines = ["# IndoFraud-Synth calibration report", "",
             f"**Overall: {'PASS' if report['ok'] else 'FAIL'}**", "",
             "| Check | Actual | Target | OK |", "|---|---|---|---|"]
    for c in report["checks"]:
        lines.append(f"| {c['check']} | {c['actual']:.4g} | {c['target']} | "
                     f"{'✅' if c['ok'] else '❌'} |")
    lines += ["", "## Silo sizes", "```json",
              json.dumps(report["silo_sizes"], indent=2), "```",
              "", "## Typology counts", "```json",
              json.dumps(report["typology_counts"], indent=2), "```",
              "", "## Temporal 70/15/15 split typology counts", "```json",
              json.dumps(report.get("split_typology_counts", {}), indent=2), "```"]
    return "\n".join(lines)


def save_report(report: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "calibration_report.md"), "w") as f:
        f.write(report_markdown(report))
