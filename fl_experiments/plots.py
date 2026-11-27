"""Publication-grade figure generator for multi-seed FL fraud-detection suites.

Generates camera-ready figures (both 300-DPI PNG and vector PDF):

Single-partition suite (`<suite_dir>/<condition>/seed_<N>/results.json`):
  fig1_round_convergence.{png,pdf}  — Round-by-round macro val AUC-PR (XGBoost,
                                      CatBoost, MLP) highlighting baseline Round-2
                                      Hessian collapse vs. normalized stability.
  fig2_main_comparison.{png,pdf}    — (a) Pooled & Macro Test AUC-PR (mean ± 95% CI)
                                      and (b) IDR Savings vs. Alert-Rate frontier.
  fig3_per_silo_fairness.{png,pdf}  — Per-institution Test AUC-PR ordered across
                                      the 15x KBMI size skew + delta over local-only.
  fig4_efficiency_pareto.{png,pdf}  — Communication volume (total & MB/round),
                                      global tree footprint, and p95 latency.
  fig5_ablation_2model.{png,pdf}    — Individual vs. 2-model & 3-model ensemble
                                      synergy + learned model weight composition.

Multi-partition suite root (`<suite_root>/<partition>/<condition>/seed_<N>/results.json`):
  fig6_skew_degradation.{png,pdf}   — RQ1 skew degradation across IID -> Dirichlet
                                      (alpha = 5.0, 0.5, 0.1) and realistic 15x KBMI
                                      profile: (a) paired Delta Macro AP vs. Centralized
                                      and (b) Pooled Test AUC-PR highlighting the
                                      Local-Only Simpson's paradox under label skew.

CLI:
  python -m fl_experiments.plots --suite-dir results/suite_indofraud_v2/profile
  python -m fl_experiments.plots --suite-dir results/suite_indofraud_v2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stats import friedman_test, holm_wilcoxon, mean_ci95

# Colorblind-safe Okabe-Ito inspired palette
PALETTE = {
    "centralized": "#222222",
    "local_only": "#7F7F7F",
    "baseline": "#D55E00",          # vermilion
    "baseline_cal": "#E69F00",      # orange
    "norm_fedbn": "#0072B2",        # blue
    "norm_size_fedbn": "#009E73",   # bluish green
    "accent": "#CC79A7",            # reddish purple
    "sky": "#56B4E9",               # sky blue
}


def _apply_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.3,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def load_suite_by_condition(suite_dir: str) -> dict[str, dict[int, dict]]:
    """Return `{condition: {seed: results_dict}}`."""
    data: dict[str, dict[int, dict]] = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(suite_dir, "*", "seed_*", "results.json"))):
        parts = os.path.normpath(path).split(os.sep)
        cond = parts[-3]
        seed = int(parts[-2].replace("seed_", ""))
        with open(path) as f:
            data[cond][seed] = json.load(f)
    return dict(data)


def _baseline_anchor_runs(data: dict[str, dict[int, dict]]) -> dict[int, dict]:
    """Find one results dict per seed containing `baselines`."""
    out: dict[int, dict] = {}
    for cond in sorted(data):
        for seed, res in data[cond].items():
            if seed not in out and "baselines" in res:
                out[seed] = res
    return out


# ---------------- Figure 1: Round-by-Round Convergence ----------------

def plot_round_convergence(data: dict[str, dict[int, dict]], out_dir: str):
    conds = [
        ("baseline", "Baseline FL (Unnorm. + FedAvg)", PALETTE["baseline"], "--", "o"),
        ("normalized_fedbn", "Proposed Uniform + FedBN", PALETTE["norm_fedbn"], "-", "s"),
        ("normalized_size_fedbn", "Proposed Size-Weighted + FedBN", PALETTE["norm_size_fedbn"], "-", "^"),
    ]
    conds = [c for c in conds if c[0] in data]
    models = [("xgboost", "XGBoost"), ("catboost", "CatBoost"), ("mlp", "MLP")]

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.6), sharey=False)
    for ax, (mkey, mtitle) in zip(axes, models):
        max_r = 1
        for ckey, label, color, ls, marker in conds:
            by_r: dict[int, list[float]] = defaultdict(list)
            n_seeds_cond = len(data[ckey])
            for res in data[ckey].values():
                for entry in res.get("fl", {}).get(mkey, {}).get("rounds", []):
                    r = int(entry["round"])
                    v = entry.get("macro_val_ap")
                    if v is not None and np.isfinite(v):
                        by_r[r].append(float(v))
            # Filter out trailing survivorship rounds where < 60% of seeds remained active
            min_active = max(1, int(np.ceil(0.6 * n_seeds_cond)))
            rs = sorted(r for r, vals in by_r.items() if len(vals) >= min_active)
            if not rs:
                continue
            max_r = max(max_r, rs[-1])
            means = np.array([mean_ci95(by_r[r])["mean"] for r in rs])
            cis = np.array([mean_ci95(by_r[r])["ci95"] for r in rs])
            ax.plot(rs, means, label=label, color=color, linestyle=ls,
                    marker=marker, markersize=4.5, linewidth=1.8)
            ax.fill_between(rs, means - cis, means + cis, color=color, alpha=0.14, linewidth=0)

        ax.set_title(mtitle)
        ax.set_xlabel("Federation Round")
        ax.set_xticks(range(1, max_r + 1))
        if mkey == "xgboost":
            ax.set_ylabel("Macro Validation AUC-PR")
            ax.set_ylim(-0.02, 0.52)
            ax.annotate(
                "Round-2 Hessian\ncollapse (~0.001)",
                xy=(2.05, 0.015), xytext=(3.6, 0.14),
                arrowprops=dict(arrowstyle="->", color=PALETTE["baseline"], lw=1.2),
                fontsize=8.0, color=PALETTE["baseline"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["baseline"], alpha=0.9),
            )
        elif mkey == "catboost":
            ax.set_ylim(-0.02, 0.52)
            ax.annotate(
                "Round-2 collapse\n(unnormalized)",
                xy=(2.05, 0.015), xytext=(3.2, 0.14),
                arrowprops=dict(arrowstyle="->", color=PALETTE["baseline"], lw=1.2),
                fontsize=8.0, color=PALETTE["baseline"], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["baseline"], alpha=0.9),
            )
        else:
            ax.set_ylim(0.41, 0.495)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(conds),
               bbox_to_anchor=(0.5, 1.06), frameon=True)
    fig.suptitle("Round-by-Round Federated Validation Convergence (5 Seeds, Mean ± 95% CI)",
                 y=1.13, fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_round_convergence")


# ---------------- Figure 2: Main AP Comparison + Cost/Alert Frontier ----------------

def plot_main_comparison(data: dict[str, dict[int, dict]], out_dir: str):
    anchors = _baseline_anchor_runs(data)
    seeds = sorted(anchors.keys())
    if not seeds:
        return

    def _collect(extractor):
        vals = [extractor(s) for s in seeds]
        return mean_ci95(vals)

    methods_bar = [
        ("Local-Only\nEnsemble",
         lambda s: anchors[s]["baselines"]["local_only"]["ensemble"]["pooled_ap"],
         lambda s: anchors[s]["baselines"]["local_only"]["ensemble"]["macro_ap"],
         PALETTE["local_only"]),
        ("Baseline FL\n(WV)",
         lambda s: data["baseline"][s]["ensemble"]["variants"]["weighted_voting"]["pooled_ap"],
         lambda s: data["baseline"][s]["ensemble"]["variants"]["weighted_voting"]["macro_ap"],
         PALETTE["baseline"]),
        ("Baseline FL\n(+Cal/Bayes)",
         lambda s: data["baseline"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["pooled_ap"],
         lambda s: data["baseline"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["macro_ap"],
         PALETTE["baseline_cal"]),
        ("Proposed\nUniform+FedBN",
         lambda s: data["normalized_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["pooled_ap"],
         lambda s: data["normalized_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["macro_ap"],
         PALETTE["norm_fedbn"]),
        ("Proposed\nSize+FedBN",
         lambda s: data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["pooled_ap"],
         lambda s: data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["macro_ap"],
         PALETTE["norm_size_fedbn"]),
    ]

    cent_pooled = _collect(lambda s: anchors[s]["baselines"]["centralized"]["ensemble"]["pooled_ap"])
    cent_macro = _collect(lambda s: anchors[s]["baselines"]["centralized"]["ensemble"]["macro_ap"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # Panel (a): Grouped bar chart (Pooled AP vs Macro AP)
    x = np.arange(len(methods_bar))
    w = 0.35
    for idx, (label, fn_p, fn_m, col) in enumerate(methods_bar):
        sp = _collect(fn_p)
        sm = _collect(fn_m)
        ax1.bar(idx - w / 2, sp["mean"], width=w, yerr=sp["ci95"], capsize=3.5,
                color=col, edgecolor="#222222", linewidth=0.6,
                label="Pooled AP" if idx == 0 else None, error_kw={"lw": 1.1})
        ax1.bar(idx + w / 2, sm["mean"], width=w, yerr=sm["ci95"], capsize=3.5,
                color=col, alpha=0.55, edgecolor="#222222", linewidth=0.6, hatch="//",
                label="Macro AP" if idx == 0 else None, error_kw={"lw": 1.1})
        pct = 100.0 * sp["mean"] / cent_pooled["mean"]
        ax1.text(idx - w / 2, sp["mean"] + sp["ci95"] + 0.006, f"{sp['mean']:.3f}",
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold")
        ax1.text(idx + w / 2, sm["mean"] + sm["ci95"] + 0.006, f"{sm['mean']:.3f}",
                 ha="center", va="bottom", fontsize=7.4, color="#333333")
        ax1.text(idx, 0.432, f"{pct:.1f}%", ha="center", va="bottom",
                 fontsize=7.3, color="#222222",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    ax1.axhline(cent_pooled["mean"], color=PALETTE["centralized"], ls="--", lw=1.4,
                label=f"Centralized Pooled ({cent_pooled['mean']:.3f})")
    ax1.axhline(cent_macro["mean"], color=PALETTE["centralized"], ls=":", lw=1.2,
                label=f"Centralized Macro ({cent_macro['mean']:.3f})")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m[0] for m in methods_bar])
    ax1.set_ylim(0.42, 0.688)
    ax1.set_ylabel("Out-of-Time Test AUC-PR")
    ax1.set_title("(a) Test AUC-PR vs. Centralized Upper Bound (% Recovery)")
    # Custom legend placed cleanly in upper empty band above bars
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    leg_elems = [
        Patch(facecolor="#888888", edgecolor="#222222", label="Pooled AP (solid)"),
        Patch(facecolor="#888888", alpha=0.55, edgecolor="#222222", hatch="//", label="Macro AP (hatched)"),
        Line2D([0], [0], color=PALETTE["centralized"], ls="--", lw=1.4,
               label=f"Centralized Pooled ({cent_pooled['mean']:.3f})"),
        Line2D([0], [0], color=PALETTE["centralized"], ls=":", lw=1.2,
               label=f"Centralized Macro ({cent_macro['mean']:.3f})"),
    ]
    ax1.legend(handles=leg_elems, loc="upper center", ncol=2, fontsize=7.5)

    # Panel (b): IDR Savings vs Alert Rate (Operational Efficiency Frontier)
    frontier_pts = [
        ("Local-Only (Flat Thr)",
         lambda s: 100 * anchors[s]["baselines"]["local_only"]["ensemble"]["alert_rate"],
         lambda s: 100 * anchors[s]["baselines"]["local_only"]["ensemble"]["savings"],
         PALETTE["local_only"], "D", (1.62, 93.45), "center"),
        ("Centralized (Flat Thr)",
         lambda s: 100 * anchors[s]["baselines"]["centralized"]["ensemble"]["alert_rate"],
         lambda s: 100 * anchors[s]["baselines"]["centralized"]["ensemble"]["savings"],
         PALETTE["centralized"], "*", (1.76, 95.95), "center"),
        ("Baseline FL (WV)",
         lambda s: 100 * data["baseline"][s]["ensemble"]["variants"]["weighted_voting"]["alert_rate"],
         lambda s: 100 * data["baseline"][s]["ensemble"]["variants"]["weighted_voting"]["savings"],
         PALETTE["baseline"], "o", (1.28, 94.38), "center"),
        ("Baseline FL\n(Bayes@1%)",
         lambda s: 100 * data["baseline"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["alert_rate"],
         lambda s: 100 * data["baseline"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["savings"],
         PALETTE["baseline_cal"], "o", (0.78, 93.72), "center"),
        ("Proposed Size+FedBN\n(WV Flat Thr)",
         lambda s: 100 * data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_voting"]["alert_rate"],
         lambda s: 100 * data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_voting"]["savings"],
         PALETTE["sky"], "^", (2.02, 94.55), "center"),
        ("Proposed Uniform+FedBN\n(Bayes@1%)",
         lambda s: 100 * data["normalized_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["alert_rate"],
         lambda s: 100 * data["normalized_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["savings"],
         PALETTE["norm_fedbn"], "s", (1.06, 95.72), "left"),
        ("Proposed Size+FedBN\n(Bayes@1%)",
         lambda s: 100 * data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["alert_rate"],
         lambda s: 100 * data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["savings"],
         PALETTE["norm_size_fedbn"], "^", (1.06, 96.45), "left"),
        ("Proposed Size+FedBN\n(Bayes Uncapped)",
         lambda s: 100 * data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes"]["alert_rate"],
         lambda s: 100 * data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes"]["savings"],
         PALETTE["norm_size_fedbn"], "P", (1.62, 97.26), "center"),
    ]

    for lbl, fn_x, fn_y, col, mk, text_xy, ha in frontier_pts:
        sx = _collect(fn_x)
        sy = _collect(fn_y)
        ms = 9.5 if mk == "*" else 6.8
        ax2.errorbar(
            sx["mean"], sy["mean"], xerr=sx["ci95"], yerr=sy["ci95"],
            fmt=mk, color=col, ecolor=col, elinewidth=1.0, capsize=2.8,
            markersize=ms, markeredgecolor="#222222", markeredgewidth=0.6, zorder=4,
        )
        ax2.annotate(
            lbl, xy=(sx["mean"], sy["mean"]), xytext=text_xy,
            textcoords="data", ha=ha, va="center", fontsize=7.5, color="#222222",
            fontweight="bold" if "Proposed" in lbl and "Bayes@1%" in lbl else "normal",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.75, shrinkA=2, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        )

    ax2.axvline(1.0, color="#999999", ls=":", lw=1.1)
    ax2.text(1.015, 93.30, "1% Alert Budget", fontsize=7.6, color="#666666", rotation=90, va="bottom")
    ax2.set_xlim(0.55, 2.35)
    ax2.set_ylim(93.15, 97.65)
    ax2.set_xlabel("Alert Rate (% of Test Transactions, Lower = Less Friction)")
    ax2.set_ylabel("IDR Fraud-Loss Savings (% Avoided, Higher = Better)")
    ax2.set_title("(b) Operational Cost-Efficiency Frontier (IDR, $C_{FP}$ = Rp 7,500)")

    fig.tight_layout()
    _save(fig, out_dir, "fig2_main_comparison")


# ---------------- Figure 3: Per-Silo Fairness Across 15x KBMI Skew ----------------

def plot_per_silo_fairness(data: dict[str, dict[int, dict]], out_dir: str):
    anchors = _baseline_anchor_runs(data)
    seeds = sorted(anchors.keys())
    if not seeds:
        return

    # Order silos strictly descending by training volume
    silo_stats = anchors[seeds[0]]["silo_stats"]
    silos = sorted(silo_stats.keys(), key=lambda k: silo_stats[k]["n_train"], reverse=True)
    n_trains = [silo_stats[s]["n_train"] for s in silos]
    short_labels = [f"{s}\n({n // 1000}k)" for s, n in zip(silos, n_trains)]

    series_spec = [
        ("Centralized Ensemble", lambda s: anchors[s]["baselines"]["centralized"]["ensemble"]["per_silo_ap"],
         PALETTE["centralized"], "--", "o"),
        ("Local-Only Ensemble", lambda s: anchors[s]["baselines"]["local_only"]["ensemble"]["per_silo_ap"],
         PALETTE["local_only"], ":", "D"),
        ("Baseline FL (WV)", lambda s: data["baseline"][s]["ensemble"]["variants"]["weighted_voting"]["per_silo_ap"],
         PALETTE["baseline"], "-.", "x"),
        ("Proposed Uniform+FedBN", lambda s: data["normalized_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["per_silo_ap"],
         PALETTE["norm_fedbn"], "-", "s"),
        ("Proposed Size+FedBN", lambda s: data["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["per_silo_ap"],
         PALETTE["norm_size_fedbn"], "-", "^"),
    ]

    means_map: dict[str, np.ndarray] = {}
    cis_map: dict[str, np.ndarray] = {}
    for lbl, fn, *_ in series_spec:
        ms, cs = [], []
        for silo in silos:
            vals = [fn(s)[silo] for s in seeds if silo in fn(s) and fn(s)[silo] is not None]
            st = mean_ci95(vals)
            ms.append(st["mean"])
            cs.append(st["ci95"])
        means_map[lbl] = np.array(ms)
        cis_map[lbl] = np.array(cs)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.2, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.55, 1.0]}
    )
    x = np.arange(len(silos))

    for lbl, _, col, ls, mk in series_spec:
        m = means_map[lbl]
        c = cis_map[lbl]
        lw = 2.0 if "Proposed" in lbl else 1.4
        ax1.plot(x, m, label=lbl, color=col, linestyle=ls, marker=mk,
                 markersize=5.5, linewidth=lw)
        if "Proposed" in lbl or "Centralized" in lbl:
            ax1.fill_between(x, m - c, m + c, color=col, alpha=0.10, linewidth=0)

    ax1.set_ylabel("Per-Silo Test AUC-PR")
    ax1.set_ylim(0.28, 0.80)
    ax1.set_title("Per-Institution Performance Across the 15× KBMI Size Skew (5 Seeds, Mean ± 95% CI)")
    ax1.legend(loc="lower left", ncol=3, fontsize=8.0)

    # Bottom panel: Delta AP over Local-Only Ensemble
    delta_base = means_map["Baseline FL (WV)"] - means_map["Local-Only Ensemble"]
    delta_unif = means_map["Proposed Uniform+FedBN"] - means_map["Local-Only Ensemble"]
    delta_size = means_map["Proposed Size+FedBN"] - means_map["Local-Only Ensemble"]
    w = 0.26
    ax2.axhline(0.0, color="#222222", lw=0.9)
    ax2.bar(x - w, delta_base, width=w, color=PALETTE["baseline"], alpha=0.85,
            edgecolor="#222222", lw=0.5, label="Baseline FL − Local-Only")
    ax2.bar(x, delta_unif, width=w, color=PALETTE["norm_fedbn"], alpha=0.9,
            edgecolor="#222222", lw=0.5, label="Proposed Uniform+FedBN − Local-Only")
    ax2.bar(x + w, delta_size, width=w, color=PALETTE["norm_size_fedbn"], alpha=0.9,
            edgecolor="#222222", lw=0.5, label="Proposed Size+FedBN − Local-Only")

    # Annotate biggest inclusion win on bank_small_1 and bpd_2
    for target_idx in (8, 9):
        val = max(delta_unif[target_idx], delta_size[target_idx])
        ax2.text(target_idx + 0.05, val + 0.008, f"+{val:.3f}", ha="center", va="bottom",
                 fontsize=7.8, fontweight="bold", color=PALETTE["norm_fedbn"])

    ax2.set_ylabel("Δ Test AUC-PR\nvs. Local-Only")
    ax2.set_ylim(-0.11, 0.23)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=18, ha="right")
    ax2.set_xlabel("Institution Silo (Ordered Largest → Smallest Training Volume)")
    ax2.legend(loc="upper left", ncol=3, fontsize=7.8)

    fig.tight_layout()
    _save(fig, out_dir, "fig3_per_silo_fairness")


# ---------------- Figure 4: Efficiency & Deployment Footprint ----------------

def plot_efficiency_footprint(data: dict[str, dict[int, dict]], out_dir: str):
    cond_spec = [
        ("baseline", "Baseline FL\n(Unnorm.)", PALETTE["baseline"]),
        ("normalized_fedbn", "Proposed\nUniform+FedBN", PALETTE["norm_fedbn"]),
        ("normalized_size_fedbn", "Proposed\nSize+FedBN", PALETTE["norm_size_fedbn"]),
    ]
    cond_spec = [c for c in cond_spec if c[0] in data]
    labels = [c[1] for c in cond_spec]
    x = np.arange(len(cond_spec))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.8, 3.8))

    # (a) Total Communication (Upload vs Broadcast) + MB/round annotation
    up_m, dn_m, tot_ci, mb_per_r = [], [], [], []
    for ckey, _, _ in cond_spec:
        ups, dns, tots, per_rs = [], [], [], []
        for res in data[ckey].values():
            fl = res["fl"]
            u = sum(fl[m]["upload_bytes"] for m in fl) / (1024 * 1024)
            d = sum(fl[m]["broadcast_bytes"] for m in fl) / (1024 * 1024)
            n_r = max(len(fl[m]["rounds"]) for m in fl)
            ups.append(u); dns.append(d); tots.append(u + d); per_rs.append((u + d) / max(n_r, 1))
        up_m.append(mean_ci95(ups)["mean"])
        dn_m.append(mean_ci95(dns)["mean"])
        tot_ci.append(mean_ci95(tots)["ci95"])
        mb_per_r.append(mean_ci95(per_rs)["mean"])

    ax1.bar(x, dn_m, width=0.52, color=[c[2] for c in cond_spec], alpha=0.88,
            edgecolor="#222222", lw=0.6, label="Server Broadcast")
    ax1.bar(x, up_m, bottom=dn_m, width=0.52, yerr=tot_ci, capsize=3.5,
            color="#F0E442", edgecolor="#222222", lw=0.6, label="Client Upload (Delta)")
    for i in range(len(cond_spec)):
        tot = up_m[i] + dn_m[i]
        ax1.text(i, tot + tot_ci[i] + 16, f"{tot:.0f} MB\n({mb_per_r[i]:.0f}/rd)",
                 ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(0, max(u + d + ci for u, d, ci in zip(up_m, dn_m, tot_ci)) * 1.24)
    ax1.set_ylabel("Cumulative Transfer (MB)")
    ax1.set_title("(a) Communication Volume")
    ax1.legend(loc="upper right", fontsize=7.6)

    # (b) Global Tree Count (XGBoost + CatBoost stacked)
    xgb_m, cat_m, tree_ci = [], [], []
    for ckey, _, _ in cond_spec:
        xs, cs, ts = [], [], []
        for res in data[ckey].values():
            sz = res["global_model_sizes"]
            xt = sz.get("xgboost", {}).get("n_trees", 0)
            ct = sz.get("catboost", {}).get("n_trees", 0)
            xs.append(xt); cs.append(ct); ts.append(xt + ct)
        xgb_m.append(mean_ci95(xs)["mean"])
        cat_m.append(mean_ci95(cs)["mean"])
        tree_ci.append(mean_ci95(ts)["ci95"])

    ax2.bar(x, xgb_m, width=0.52, color="#0072B2", alpha=0.88,
            edgecolor="#222222", lw=0.6, label="XGBoost Trees")
    ax2.bar(x, cat_m, bottom=xgb_m, width=0.52, yerr=tree_ci, capsize=3.5,
            color="#56B4E9", alpha=0.88, edgecolor="#222222", lw=0.6, label="CatBoost Trees")
    for i in range(len(cond_spec)):
        tot = xgb_m[i] + cat_m[i]
        red = 100.0 * (1.0 - tot / (xgb_m[0] + cat_m[0])) if i > 0 else 0.0
        txt = f"{tot:,.0f}" if i == 0 else f"{tot:,.0f}\n(−{red:.0f}%)"
        ax2.text(i, tot + tree_ci[i] + 150, txt, ha="center", va="bottom",
                 fontsize=7.8, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_ylim(0, (xgb_m[0] + cat_m[0]) * 1.22)
    ax2.set_ylabel("Total Trees in Global Ensemble")
    ax2.set_title("(b) Model Size (Bounded Tree Budget)")
    ax2.legend(loc="upper right", fontsize=7.6)

    # (c) Single-Transaction p95 Latency (ms)
    lat_ens_m, lat_ens_ci = [], []
    lat_xgb_m, lat_cat_m, lat_mlp_m = [], [], []
    for ckey, _, _ in cond_spec:
        le, lx, lc, lm = [], [], [], []
        for res in data[ckey].values():
            lt = res.get("latency_ms_p95", {})
            if "ensemble" in lt: le.append(lt["ensemble"])
            if "xgboost" in lt: lx.append(lt["xgboost"])
            if "catboost" in lt: lc.append(lt["catboost"])
            if "mlp" in lt: lm.append(lt["mlp"])
        lat_ens_m.append(mean_ci95(le)["mean"])
        lat_ens_ci.append(mean_ci95(le)["ci95"])
        lat_xgb_m.append(mean_ci95(lx)["mean"])
        lat_cat_m.append(mean_ci95(lc)["mean"])
        lat_mlp_m.append(mean_ci95(lm)["mean"])

    bw = 0.20
    ax3.bar(x - 1.5 * bw, lat_xgb_m, width=bw, color="#0072B2", edgecolor="#222222", lw=0.4, label="XGBoost")
    ax3.bar(x - 0.5 * bw, lat_cat_m, width=bw, color="#56B4E9", edgecolor="#222222", lw=0.4, label="CatBoost")
    ax3.bar(x + 0.5 * bw, lat_mlp_m, width=bw, color="#CC79A7", edgecolor="#222222", lw=0.4, label="MLP")
    ax3.bar(x + 1.5 * bw, lat_ens_m, width=bw, yerr=lat_ens_ci, capsize=2.8,
            color="#222222", edgecolor="#222222", lw=0.4, label="Ensemble (p95)")
    for i in range(len(cond_spec)):
        ax3.text(i + 1.5 * bw, lat_ens_m[i] + lat_ens_ci[i] + 0.25, f"{lat_ens_m[i]:.1f}ms",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.set_ylim(0, max(m + c for m, c in zip(lat_ens_m, lat_ens_ci)) * 1.28)
    ax3.set_ylabel("p95 Inference Latency (ms / txn)")
    ax3.set_title("(c) Single-Transaction Latency")
    ax3.legend(loc="upper right", ncol=2, fontsize=7.2)

    fig.tight_layout()
    _save(fig, out_dir, "fig4_efficiency_pareto")


# ---------------- Figure 5: 2-Model Ablations & Weight Composition ----------------

def plot_ablation_and_weights(data: dict[str, dict[int, dict]], out_dir: str):
    target_cond = "normalized_size_fedbn" if "normalized_size_fedbn" in data else "normalized_fedbn"
    seeds = sorted(data[target_cond].keys())
    anchors = _baseline_anchor_runs(data)

    # Panel (a): Individual FL models vs 2-model & 3-model calibrated ensembles
    items = [
        ("FL CatBoost", lambda s: data[target_cond][s]["ensemble"]["variants"]["catboost"]["pooled_ap"], "#56B4E9"),
        ("FL MLP\n(FedBN)", lambda s: data[target_cond][s]["ensemble"]["variants"]["mlp"]["pooled_ap"], "#CC79A7"),
        ("FL XGBoost", lambda s: data[target_cond][s]["ensemble"]["variants"]["xgboost"]["pooled_ap"], "#0072B2"),
        ("Cat + MLP", lambda s: data[target_cond][s]["ensemble"]["ablations_2model"]["catboost+mlp"]["weighted_calibrated_bayes_1pct"]["pooled_ap"], "#E69F00"),
        ("XGB + Cat", lambda s: data[target_cond][s]["ensemble"]["ablations_2model"]["xgboost+catboost"]["weighted_calibrated_bayes_1pct"]["pooled_ap"], "#E69F00"),
        ("XGB + MLP", lambda s: data[target_cond][s]["ensemble"]["ablations_2model"]["xgboost+mlp"]["weighted_calibrated_bayes_1pct"]["pooled_ap"], "#009E73"),
        ("3-Model\nTrio (Cal.)", lambda s: data[target_cond][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"]["pooled_ap"], "#009E73"),
        ("3-Model\nTrio (WV)", lambda s: data[target_cond][s]["ensemble"]["variants"]["weighted_voting"]["pooled_ap"], "#222222"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    x = np.arange(len(items))
    for i, (lbl, fn, col) in enumerate(items):
        st = mean_ci95([fn(s) for s in seeds])
        ax1.bar(i, st["mean"], width=0.62, yerr=st["ci95"], capsize=3.2,
                color=col, alpha=0.88, edgecolor="#222222", lw=0.6)
        ax1.text(i, st["mean"] + st["ci95"] + 0.004, f"{st['mean']:.3f}",
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([it[0] for it in items], fontsize=8.0)
    ax1.set_ylim(0.46, 0.625)
    ax1.set_ylabel("Out-of-Time Pooled Test AUC-PR")
    ax1.set_title("(a) Base-Model vs. 2-Model & 3-Model Ensemble Ablation (Size+FedBN)")

    # Panel (b): Learned Calibrated Ensemble Weights across settings
    w_groups = [
        ("Centralized\nEnsemble", lambda s: anchors[s]["baselines"]["centralized"]["ensemble"]["weights"]),
        ("Baseline FL\n(Calibrated)", lambda s: data["baseline"][s]["ensemble"]["calibrated_weights"]),
        ("Uniform+FedBN\n(Calibrated)", lambda s: data["normalized_fedbn"][s]["ensemble"]["calibrated_weights"]),
        ("Size+FedBN\n(Calibrated)", lambda s: data["normalized_size_fedbn"][s]["ensemble"]["calibrated_weights"]),
    ]
    gx = np.arange(len(w_groups))
    w_xgb, w_cat, w_mlp = [], [], []
    for _, fn in w_groups:
        w_xgb.append(100 * mean_ci95([fn(s).get("xgboost", 0.0) for s in seeds])["mean"])
        w_cat.append(100 * mean_ci95([fn(s).get("catboost", 0.0) for s in seeds])["mean"])
        w_mlp.append(100 * mean_ci95([fn(s).get("mlp", 0.0) for s in seeds])["mean"])

    ax2.bar(gx, w_xgb, width=0.52, color="#0072B2", edgecolor="#222222", lw=0.6, label="XGBoost")
    ax2.bar(gx, w_cat, bottom=w_xgb, width=0.52, color="#56B4E9", edgecolor="#222222", lw=0.6, label="CatBoost")
    bottoms = [a + b for a, b in zip(w_xgb, w_cat)]
    ax2.bar(gx, w_mlp, bottom=bottoms, width=0.52, color="#CC79A7", edgecolor="#222222", lw=0.6, label="MLP")

    for i in range(len(w_groups)):
        if w_xgb[i] >= 8:
            ax2.text(i, w_xgb[i] / 2, f"{w_xgb[i]:.0f}%", ha="center", va="center",
                     color="white", fontsize=8.0, fontweight="bold")
        if w_cat[i] >= 8:
            ax2.text(i, w_xgb[i] + w_cat[i] / 2, f"{w_cat[i]:.0f}%", ha="center", va="center",
                     color="#111111", fontsize=8.0, fontweight="bold")
        if w_mlp[i] >= 8:
            ax2.text(i, bottoms[i] + w_mlp[i] / 2, f"{w_mlp[i]:.0f}%", ha="center", va="center",
                     color="white", fontsize=8.0, fontweight="bold")

    ax2.set_xticks(gx)
    ax2.set_xticklabels([g[0] for g in w_groups], fontsize=8.0)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("Learned Ensemble Weight (%)")
    ax2.set_title("(b) Ensemble Weight Composition (Mean Across 5 Seeds)")
    ax2.legend(loc="upper center", ncol=3, fontsize=7.8)

    fig.tight_layout()
    _save(fig, out_dir, "fig5_ablation_2model")


# ---------------- Figure 6: RQ1 Multi-Partition Skew Degradation ----------------

_SKEW_PARTITION_ORDER = [
    ("equal", "IID\n(Equal)"),
    ("dirichlet_a5p0", "Dirichlet\n(α = 5.0)"),
    ("dirichlet_a0p5", "Dirichlet\n(α = 0.5)"),
    ("dirichlet_a0p1", "Dirichlet\n(α = 0.1)"),
]
_PROFILE_PARTITION = ("profile", "Profile\n(15× KBMI)")


def _discover_partitions(suite_root: str) -> list[tuple[str, str, dict[str, dict[int, dict]]]]:
    """Return ordered list of `(part_key, display_label, condition_seed_data)`."""
    found = []
    for key, label in _SKEW_PARTITION_ORDER + [_PROFILE_PARTITION]:
        pdir = os.path.join(suite_root, key)
        if os.path.isdir(pdir):
            pdata = load_suite_by_condition(pdir)
            if pdata:
                found.append((key, label, pdata))
    return found


def plot_skew_degradation(suite_root: str, out_dir: str | None = None) -> list[str]:
    """Generate Fig. 6 (RQ1): Cross-partition skew degradation & Simpson's paradox."""
    _apply_pub_style()
    parts = _discover_partitions(suite_root)
    if not parts:
        raise SystemExit(f"No partition subdirectories with results.json found under {suite_root}")

    fig_dir = out_dir or os.path.join(suite_root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Canonical primary methods matching stats.py `select_primary_methods`
    method_specs = [
        (
            "centralized:ensemble",
            "Centralized Ensemble",
            lambda d, a, s, m: a[s]["baselines"]["centralized"]["ensemble"][m],
            PALETTE["centralized"], "--", "o", 1.7,
        ),
        (
            "local_only:ensemble",
            "Local-Only Ensemble",
            lambda d, a, s, m: a[s]["baselines"]["local_only"]["ensemble"][m],
            PALETTE["local_only"], ":", "D", 1.7,
        ),
        (
            "baseline:weighted_voting",
            "Baseline FL (WV)",
            lambda d, a, s, m: d["baseline"][s]["ensemble"]["variants"]["weighted_voting"][m],
            PALETTE["baseline"], "-.", "v", 1.75,
        ),
        (
            "normalized_fedbn:weighted_calibrated_bayes_1pct",
            "Proposed Uniform+FedBN",
            lambda d, a, s, m: d["normalized_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"][m],
            PALETTE["norm_fedbn"], "-", "s", 2.1,
        ),
        (
            "normalized_size_fedbn:weighted_calibrated_bayes_1pct",
            "Proposed Size+FedBN",
            lambda d, a, s, m: d["normalized_size_fedbn"][s]["ensemble"]["variants"]["weighted_calibrated_bayes_1pct"][m],
            PALETTE["norm_size_fedbn"], "-", "^", 2.0,
        ),
    ]

    # Collect per-partition summary stats + full N-block matrix for Friedman/Holm badge
    skew_keys = {k for k, _ in _SKEW_PARTITION_ORDER}
    part_stats: dict[str, dict[str, dict]] = {}
    block_rows_macro: list[list[float]] = []

    for pkey, _, pdata in parts:
        anchors = _baseline_anchor_runs(pdata)
        seeds = sorted(anchors.keys())
        # Ensure all conditions are present for each seed
        valid_seeds = [
            s for s in seeds
            if all(c in pdata and s in pdata[c] for c in ("baseline", "normalized_fedbn", "normalized_size_fedbn"))
        ]
        part_stats[pkey] = {}
        cent_macro_by_seed = {
            s: anchors[s]["baselines"]["centralized"]["ensemble"]["macro_ap"]
            for s in valid_seeds
        }
        for mkey, _, fn, *_ in method_specs:
            pooled_vals = [fn(pdata, anchors, s, "pooled_ap") for s in valid_seeds]
            macro_vals = [fn(pdata, anchors, s, "macro_ap") for s in valid_seeds]
            delta_vals = [mv - cent_macro_by_seed[s] for mv, s in zip(macro_vals, valid_seeds)]
            part_stats[pkey][mkey] = {
                "pooled": mean_ci95(pooled_vals),
                "macro": mean_ci95(macro_vals),
                "delta_macro": mean_ci95(delta_vals),
            }
        for s in valid_seeds:
            block_rows_macro.append([
                fn(pdata, anchors, s, "macro_ap") for _, _, fn, *_ in method_specs
            ])

    # Compute N-block Friedman & Holm-Wilcoxon dynamically for the statistical badge
    mnames = [ms[0] for ms in method_specs]
    mat_macro = np.asarray(block_rows_macro, dtype=np.float64)
    fr_macro = friedman_test(mat_macro, mnames, higher_is_better=True)
    hw_macro = holm_wilcoxon(mat_macro, mnames, higher_is_better=True)
    p_unif_vs_base = float("nan")
    for pair in hw_macro:
        pair_set = {pair["method_a"], pair["method_b"]}
        if pair_set == {"baseline:weighted_voting", "normalized_fedbn:weighted_calibrated_bayes_1pct"}:
            p_unif_vs_base = pair["p_holm"]
            break

    # X-axis coordinates: continuous Dirichlet progression [0..k-1] + separated Profile column
    skew_parts = [(k, lbl) for k, lbl, _ in parts if k in skew_keys]
    has_profile = any(k == _PROFILE_PARTITION[0] for k, _, _ in parts)

    x_skew = np.arange(len(skew_parts), dtype=float)
    sep_x = len(skew_parts) - 0.42
    prof_x = len(skew_parts) + 0.30

    tick_pos = list(x_skew) + ([prof_x] if has_profile else [])
    tick_lbls = [lbl for _, lbl in skew_parts] + ([_PROFILE_PARTITION[1]] if has_profile else [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.65))

    # Subtle shaded region for realistic Profile regime on both panels
    if has_profile:
        for ax in (ax1, ax2):
            ax.axvline(sep_x, color="#666666", ls="--", lw=1.1, alpha=0.75, zorder=1)
            ax.axvspan(sep_x, prof_x + 0.62, color="#EEF2F7", alpha=0.65, zorder=0)
            ax.text(
                prof_x, 0.975, "Realistic\nInstitutional", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.6, color="#4A5568", style="italic",
            )

    # ---- Panel (a): Paired Delta Macro Test AUC-PR vs. Centralized ----
    ax1.axhline(0.0, color=PALETTE["centralized"], ls="--", lw=1.5, zorder=2)

    non_cent_specs = [ms for ms in method_specs if ms[0] != "centralized:ensemble"]
    dodges_skew_a = [-0.075, -0.025, 0.025, 0.075]
    dodges_prof_a = [-0.18, -0.06, 0.06, 0.18]

    for idx, (mkey, lbl, _, col, ls, mk, lw) in enumerate(non_cent_specs):
        dx = dodges_skew_a[idx]
        ys = np.array([part_stats[k][mkey]["delta_macro"]["mean"] for k, _ in skew_parts])
        cis = np.array([part_stats[k][mkey]["delta_macro"]["ci95"] for k, _ in skew_parts])
        ax1.errorbar(
            x_skew + dx, ys, yerr=cis, color=col, linestyle=ls, marker=mk,
            markersize=5.8, linewidth=lw, capsize=3.0, elinewidth=1.05,
            markeredgecolor="#222222", markeredgewidth=0.5, zorder=4 if "Proposed" in lbl else 3,
        )
        if has_profile:
            pk = _PROFILE_PARTITION[0]
            py = part_stats[pk][mkey]["delta_macro"]["mean"]
            pci = part_stats[pk][mkey]["delta_macro"]["ci95"]
            ax1.errorbar(
                prof_x + dodges_prof_a[idx], py, yerr=pci, color=col, linestyle="none",
                marker=mk, markersize=6.2, capsize=3.2, elinewidth=1.15,
                markeredgecolor="#222222", markeredgewidth=0.55, zorder=4 if "Proposed" in lbl else 3,
            )

    # Annotate Centralized raw Macro AP anchors softly right above y=0
    anchor_xs = list(x_skew) + ([prof_x] if has_profile else [])
    anchor_keys = [k for k, _ in skew_parts] + ([_PROFILE_PARTITION[0]] if has_profile else [])
    for xi, pk in zip(anchor_xs, anchor_keys):
        cm = part_stats[pk]["centralized:ensemble"]["macro"]["mean"]
        tag = f"Cent: {cm:.3f}" if xi == 0 else f"{cm:.3f}"
        ax1.text(
            xi, 0.014, tag, ha="center", va="bottom",
            fontsize=7.3, color="#333333", fontweight="bold" if xi == 0 else "normal",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.94),
            zorder=7,
        )

    # Callout 1: Local-Only within-silo collapse at Dir(0.1)
    if "dirichlet_a0p1" in part_stats:
        lo_d01 = part_stats["dirichlet_a0p1"]["local_only:ensemble"]["delta_macro"]["mean"]
        ax1.annotate(
            f"Local-Only within-silo\ncollapse ({lo_d01:+.3f} Macro AP)",
            xy=(3.0 + dodges_skew_a[0] - 0.03, lo_d01),
            xytext=(1.55, -0.31),
            ha="center", va="center", fontsize=7.7, color="#333333", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["local_only"], lw=1.2, connectionstyle="arc3,rad=0.12"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["local_only"], alpha=0.94),
            zorder=6,
        )

    # Callout 2: Profile improvement (+0.044 over Baseline)
    if has_profile:
        unif_p = part_stats["profile"]["normalized_fedbn:weighted_calibrated_bayes_1pct"]["delta_macro"]["mean"]
        base_p = part_stats["profile"]["baseline:weighted_voting"]["delta_macro"]["mean"]
        gain_p = unif_p - base_p
        ax1.annotate(
            f"Proposed vs. Baseline:\n{unif_p:+.3f} vs. {base_p:+.3f}\n(+{gain_p:.3f} Macro AP)",
            xy=(prof_x + dodges_prof_a[2], unif_p - 0.025),
            xytext=(prof_x - 0.04, -0.23),
            ha="center", va="center", fontsize=7.4, color=PALETTE["norm_fedbn"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["norm_fedbn"], lw=1.15, connectionstyle="arc3,rad=0.18"),
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=PALETTE["norm_fedbn"], alpha=0.95),
            zorder=6,
        )

    # Dynamic Friedman + Holm-Wilcoxon badge (bottom-left) with publication math formatting
    def _fmt_p_tex(p: float) -> str:
        if not np.isfinite(p):
            return r"\mathrm{n/a}"
        if p >= 0.001:
            return f"{p:.3f}"
        mant, exp = f"{p:.1e}".split("e")
        return rf"{mant}\times 10^{{{int(exp)}}}"

    n_blocks = mat_macro.shape[0]
    stat_txt = (
        f"{n_blocks}-Block Friedman (Macro AP): "
        rf"$\chi^2 = {fr_macro['chi2']:.2f},\; p = {_fmt_p_tex(fr_macro['p_value'])}$"
        "\n"
        rf"Uniform+FedBN vs. Baseline: $p_{{\mathrm{{Holm}}}} = {_fmt_p_tex(p_unif_vs_base)}$"
    )
    ax1.text(
        0.02, 0.03, stat_txt, transform=ax1.transAxes, ha="left", va="bottom",
        fontsize=7.3, color="#222222",
        bbox=dict(boxstyle="round,pad=0.28", fc="#FAFAFA", ec="#BBBBBB", alpha=0.95),
        zorder=6,
    )

    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(tick_lbls)
    ax1.set_xlim(-0.44, (prof_x + 0.62) if has_profile else (len(skew_parts) - 0.5))
    ax1.set_ylim(-0.44, 0.12)
    ax1.set_ylabel("Δ Per-Silo Macro Test AUC-PR\n(Method − Centralized Upper Bound)")
    ax1.set_title("(a) Unconfounded Within-Silo Degradation vs. Centralized")

    # ---- Panel (b): Pooled Test AUC-PR Across Regimes (Exposing Simpson's Paradox) ----
    dodges_skew_b = [-0.08, -0.04, 0.0, 0.04, 0.08]
    dodges_prof_b = [-0.20, -0.10, 0.0, 0.10, 0.20]

    for idx, (mkey, lbl, _, col, ls, mk, lw) in enumerate(method_specs):
        dx = dodges_skew_b[idx]
        ys = np.array([part_stats[k][mkey]["pooled"]["mean"] for k, _ in skew_parts])
        cis = np.array([part_stats[k][mkey]["pooled"]["ci95"] for k, _ in skew_parts])
        # Clip error bars to valid [0, 1] AUC-PR domain
        yerr_lo = np.minimum(cis, ys)
        yerr_hi = np.minimum(cis, 1.0 - ys)
        ax2.errorbar(
            x_skew + dx, ys, yerr=[yerr_lo, yerr_hi], label=lbl, color=col,
            linestyle=ls, marker=mk, markersize=5.8, linewidth=lw,
            capsize=3.0, elinewidth=1.05, markeredgecolor="#222222", markeredgewidth=0.5,
            zorder=4 if "Proposed" in lbl else 3,
        )
        if has_profile:
            pk = _PROFILE_PARTITION[0]
            py = part_stats[pk][mkey]["pooled"]["mean"]
            pci = part_stats[pk][mkey]["pooled"]["ci95"]
            ax2.errorbar(
                prof_x + dodges_prof_b[idx], py,
                yerr=[[min(pci, py)], [min(pci, 1.0 - py)]],
                color=col, linestyle="none", marker=mk, markersize=6.2,
                capsize=3.2, elinewidth=1.15, markeredgecolor="#222222", markeredgewidth=0.55,
                zorder=4 if "Proposed" in lbl else 3,
            )

    # Callout 1: Simpson's Paradox on Local-Only at Dir(0.5) & Dir(0.1)
    if "dirichlet_a0p1" in part_stats:
        lo_p01 = part_stats["dirichlet_a0p1"]["local_only:ensemble"]["pooled"]["mean"]
        lo_m01 = part_stats["dirichlet_a0p1"]["local_only:ensemble"]["macro"]["mean"]
        ax2.annotate(
            f"Simpson's Paradox (Local-Only):\n{lo_p01:.3f} Pooled AP via silo base-rate\nmemorization (vs. {lo_m01:.3f} Macro AP)",
            xy=(3.0 + dodges_skew_b[1] - 0.04, lo_p01),
            xytext=(0.95, 0.905),
            ha="center", va="center", fontsize=7.6, color="#333333", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["local_only"], lw=1.2, connectionstyle="arc3,rad=-0.10"),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["local_only"], alpha=0.95),
            zorder=6,
        )

    # Callout 2: Baseline FL degradation vs Proposed resilience
    if "equal" in part_stats and "dirichlet_a0p1" in part_stats:
        b_eq = part_stats["equal"]["baseline:weighted_voting"]["pooled"]["mean"]
        b_01 = part_stats["dirichlet_a0p1"]["baseline:weighted_voting"]["pooled"]["mean"]
        pct_drop = 100.0 * (b_eq - b_01) / b_eq
        ax2.annotate(
            f"Baseline FL degrades\n{b_eq:.3f} → {b_01:.3f} (−{pct_drop:.0f}%)",
            xy=(3.0 + dodges_skew_b[2] - 0.02, b_01 - 0.015),
            xytext=(1.35, 0.27),
            ha="center", va="center", fontsize=7.6, color=PALETTE["baseline"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["baseline"], lw=1.2, connectionstyle="arc3,rad=0.14"),
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=PALETTE["baseline"], alpha=0.95),
            zorder=6,
        )

    # Callout 3: Proposed beats Baseline across all 5 partitions (Profile anchor)
    if has_profile:
        unif_pp = part_stats["profile"]["normalized_fedbn:weighted_calibrated_bayes_1pct"]["pooled"]["mean"]
        base_pp = part_stats["profile"]["baseline:weighted_voting"]["pooled"]["mean"]
        ax2.annotate(
            f"Proposed: {unif_pp:.3f}\nvs. Baseline: {base_pp:.3f}",
            xy=(prof_x + dodges_prof_b[3], unif_pp - 0.02),
            xytext=(prof_x - 0.02, 0.365),
            ha="center", va="center", fontsize=7.4, color=PALETTE["norm_fedbn"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["norm_fedbn"], lw=1.15, connectionstyle="arc3,rad=0.18"),
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=PALETTE["norm_fedbn"], alpha=0.95),
            zorder=6,
        )

    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbls)
    ax2.set_xlim(-0.44, (prof_x + 0.62) if has_profile else (len(skew_parts) - 0.5))
    ax2.set_ylim(0.11, 1.07)
    ax2.set_ylabel("Out-of-Time Pooled Test AUC-PR")
    ax2.set_title("(b) Pooled Test AUC-PR (Exposing Label-Skew Confounding)")

    # Shared legend across the top
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(method_specs),
        bbox_to_anchor=(0.5, 1.03), frameon=True, fontsize=8.1,
    )
    fig.suptitle(
        "RQ1: Non-IID Skew Degradation Across Five Partition Regimes (K = 10 Silos, 5 Seeds, Mean ± 95% CI)",
        y=1.10, fontsize=11.0, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, fig_dir, "fig6_skew_degradation")
    return [
        os.path.join(fig_dir, "fig6_skew_degradation.png"),
        os.path.join(fig_dir, "fig6_skew_degradation.pdf"),
    ]


def generate_all_figures(suite_dir: str, out_dir: str | None = None) -> list[str]:
    _apply_pub_style()
    data = load_suite_by_condition(suite_dir)
    if not data:
        # Auto-fallback to multi-partition suite root if partition subdirs exist
        parts = _discover_partitions(suite_dir)
        if parts:
            return plot_skew_degradation(suite_dir, out_dir)
        raise SystemExit(f"No <condition>/seed_*/results.json found under {suite_dir}")
    fig_dir = out_dir or os.path.join(suite_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_round_convergence(data, fig_dir)
    plot_main_comparison(data, fig_dir)
    plot_per_silo_fairness(data, fig_dir)
    plot_efficiency_footprint(data, fig_dir)
    plot_ablation_and_weights(data, fig_dir)
    return sorted(
        os.path.join(fig_dir, f) for f in os.listdir(fig_dir)
        if f.endswith((".png", ".pdf"))
    )


def main():
    ap = argparse.ArgumentParser(description="Generate publication figures from a multi-seed FL suite")
    ap.add_argument(
        "--suite-dir", required=True,
        help="Single-partition dir (<cond>/seed_<N>/results.json) or multi-partition suite root",
    )
    ap.add_argument("--out", default=None, help="Output directory (defaults to <suite-dir>/figures)")
    args = ap.parse_args()
    files = generate_all_figures(args.suite_dir, args.out)
    print(f"Generated {len(files)} figure files -> {args.out or os.path.join(args.suite_dir, 'figures')}")
    for f in files:
        print(" ", f)


if __name__ == "__main__":
    main()
