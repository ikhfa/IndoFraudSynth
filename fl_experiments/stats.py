"""Statistical analysis for multi-seed FL experiments:
- Mean +/- 95% Student-t CI
- Friedman test (chi2, p-value, average ranks, Kendall's W effect size)
- Nemenyi post-hoc Critical Difference (exact via SciPy studentized_range)
- Pairwise Holm-Bonferroni-adjusted Wilcoxon signed-rank tests + rank-biserial effect size
- Critical Difference (CD) diagram, Markdown summary, and LaTeX table export.

CLI:
  python -m fl_experiments.stats --suite-dir results/suite_indofraud --metric pooled_ap
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats


DEFAULT_METRICS = [
    "pooled_ap",
    "macro_ap",
    "recall_at_fpr_001",
    "f1",
    "alert_rate",
    "fpr",
    "savings",
    "bayes_savings",
    "bayes_savings_at_1pct",
    "ece",
    "brier",
]

LOWER_IS_BETTER = {"ece", "brier", "latency_ms", "comm_mb", "fpr"}

_BASELINE_VARIANT_PREF = (
    "weighted_voting",
    "soft_voting",
    "weighted_calibrated_bayes_1pct",
    "weighted_calibrated_bayes",
    "weighted_calibrated_per_silo",
)

_PROPOSED_VARIANT_PREF = (
    "weighted_calibrated_bayes_1pct",
    "weighted_calibrated_bayes",
    "weighted_calibrated_per_silo",
    "weighted_voting",
    "soft_voting",
)


_BASELINE_ANCHOR_PREF = (
    "ensemble",
    "xgboost",
    "catboost",
    "mlp",
)


def select_primary_methods(complete_methods: list[str]) -> list[str]:
    """Select 1 canonical representative per FL condition and 1 per baseline anchor
    (`centralized` / `local_only`, preferring `ensemble`) so Friedman-Nemenyi and
    Holm-Wilcoxon tests are not diluted by 12–20 strongly correlated variants."""
    by_cond: dict[str, dict[str, str]] = {}
    selected: list[str] = []
    for m in complete_methods:
        if ":" not in m:
            selected.append(m)
            continue
        cond, var = m.split(":", 1)
        by_cond.setdefault(cond, {})[var] = m

    for cond in sorted(by_cond):
        variants = by_cond[cond]
        if cond in ("centralized", "local_only"):
            pref = _BASELINE_ANCHOR_PREF
        elif cond == "baseline":
            pref = _BASELINE_VARIANT_PREF
        else:
            pref = _PROPOSED_VARIANT_PREF
        chosen = None
        for v in pref:
            if v in variants:
                chosen = variants[v]
                break
        if chosen is None:
            chosen = variants[sorted(variants)[0]]
        selected.append(chosen)
    return sorted(selected)


def mean_ci95(values) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan"), "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95": 0.0, "n": 1}
    std = float(arr.std(ddof=1))
    t_crit = float(sp_stats.t.ppf(0.975, df=n - 1))
    ci95 = float(t_crit * std / np.sqrt(n))
    return {"mean": mean, "std": std, "ci95": ci95, "n": n}


def compute_ranks(matrix: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Rank methods per block (row): rank 1 = best."""
    M = np.asarray(matrix, dtype=np.float64)
    ranks = np.empty_like(M)
    for i in range(M.shape[0]):
        row = -M[i] if higher_is_better else M[i]
        ranks[i] = sp_stats.rankdata(row, method="average")
    return ranks


def friedman_test(matrix: np.ndarray, method_names: list[str],
                  higher_is_better: bool = True) -> dict:
    """Friedman test across N blocks (rows) x k methods (columns)."""
    M = np.asarray(matrix, dtype=np.float64)
    n_blocks, k = M.shape
    ranks = compute_ranks(M, higher_is_better=higher_is_better)
    avg_ranks = {m: float(r) for m, r in zip(method_names, ranks.mean(axis=0))}
    if k < 3 or n_blocks < 2:
        return {
            "chi2": float("nan"), "p_value": float("nan"), "kendalls_w": float("nan"),
            "n_blocks": n_blocks, "k_methods": k, "avg_ranks": avg_ranks,
            "note": "Friedman test requires k >= 3 methods and n >= 2 blocks.",
        }
    # Check for completely identical columns across all rows
    if np.allclose(M, M[:, :1]):
        return {
            "chi2": 0.0, "p_value": 1.0, "kendalls_w": 0.0,
            "n_blocks": n_blocks, "k_methods": k, "avg_ranks": avg_ranks,
        }
    stat, pval = sp_stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    w = float(stat / (n_blocks * (k - 1))) if n_blocks * (k - 1) > 0 else float("nan")
    return {
        "chi2": float(stat),
        "p_value": float(pval),
        "kendalls_w": float(np.clip(w, 0.0, 1.0)) if np.isfinite(w) else float("nan"),
        "n_blocks": n_blocks,
        "k_methods": k,
        "avg_ranks": avg_ranks,
    }


def nemenyi_posthoc(avg_ranks: dict[str, float], n_blocks: int, alpha: float = 0.05) -> dict:
    """Nemenyi post-hoc critical difference and pairwise p-values (Demsar 2006)."""
    names = list(avg_ranks.keys())
    k = len(names)
    if k < 2 or n_blocks < 1:
        return {"alpha": alpha, "q_alpha": float("nan"), "cd": float("nan"), "pairs": []}
    # Demsar (2006) q_alpha = studentized_range(k, inf) / sqrt(2)
    q_sr = float(sp_stats.studentized_range.ppf(1.0 - alpha, k, np.inf))
    q_alpha = q_sr / np.sqrt(2.0)
    se = np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    cd = float(q_alpha * se)

    pairs = []
    for a, b in combinations(names, 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        q_obs = (diff / se) * np.sqrt(2.0) if se > 0 else 0.0
        pval = float(sp_stats.studentized_range.sf(q_obs, k, np.inf))
        pairs.append({
            "method_a": a,
            "method_b": b,
            "rank_diff": float(diff),
            "p_value": pval,
            "significant": bool(diff > cd),
        })
    return {"alpha": alpha, "q_alpha": float(q_alpha), "cd": cd, "pairs": pairs}


def holm_wilcoxon(matrix: np.ndarray, method_names: list[str],
                  higher_is_better: bool = True, alpha: float = 0.05) -> list[dict]:
    """Pairwise two-sided Wilcoxon signed-rank tests with Holm-Bonferroni correction
    and matched-pairs rank-biserial effect size r_rb in [-1, 1]."""
    M = np.asarray(matrix, dtype=np.float64)
    k = M.shape[1]
    raw = []
    for i, j in combinations(range(k), 2):
        diff = M[:, i] - M[:, j]
        if not higher_is_better:
            diff = -diff
        nz = diff[diff != 0]
        if len(nz) == 0:
            stat, pval, r_rb = 0.0, 1.0, 0.0
        else:
            res = sp_stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
            stat, pval = float(res.statistic), float(res.pvalue)
            ranks = sp_stats.rankdata(np.abs(nz))
            r_plus = float(ranks[nz > 0].sum())
            r_minus = float(ranks[nz < 0].sum())
            denom = r_plus + r_minus
            r_rb = (r_plus - r_minus) / denom if denom > 0 else 0.0
        raw.append({
            "method_a": method_names[i],
            "method_b": method_names[j],
            "mean_diff": float(np.mean(diff)),
            "stat": stat,
            "p_raw": pval,
            "rank_biserial": float(r_rb),
        })

    # Holm-Bonferroni step-down adjustment
    m = len(raw)
    order = np.argsort([r["p_raw"] for r in raw])
    running_max = 0.0
    for rank_idx, idx in enumerate(order):
        adj = min(1.0, raw[idx]["p_raw"] * (m - rank_idx))
        running_max = max(running_max, adj)
        raw[idx]["p_holm"] = float(running_max)
        raw[idx]["significant"] = bool(running_max < alpha)
    return raw


def plot_cd_diagram(avg_ranks: dict[str, float], cd: float, out_path: str,
                    title: str = "Critical Difference Diagram (Nemenyi, alpha=0.05)"):
    """Render a Demsar-style Critical Difference diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    k = len(items)
    if k < 2:
        return
    names = [x[0] for x in items]
    ranks = [x[1] for x in items]

    fig_h = max(2.8, 0.45 * k + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Top rank axis (1 = best on the right, standard Demsar convention)
    axis_y = 0.78
    ax.plot([1, k], [axis_y, axis_y], color="black", lw=1.2)
    for tick in range(1, k + 1):
        x = k + 1 - tick
        ax.plot([x, x], [axis_y, axis_y + 0.03], color="black", lw=1.0)
        ax.text(x, axis_y + 0.05, str(tick), ha="center", va="bottom", fontsize=9)

    # CD bar at top-left
    if np.isfinite(cd) and cd > 0:
        cd_x0 = 1.0
        cd_x1 = min(1.0 + cd, float(k))
        ax.plot([cd_x0, cd_x1], [0.93, 0.93], color="crimson", lw=2.0)
        ax.plot([cd_x0, cd_x0], [0.91, 0.95], color="crimson", lw=1.5)
        ax.plot([cd_x1, cd_x1], [0.91, 0.95], color="crimson", lw=1.5)
        ax.text((cd_x0 + cd_x1) / 2, 0.955, f"CD = {cd:.2f}", ha="center", va="bottom",
                fontsize=9, color="crimson", fontweight="bold")

    # Method elbows: best half right, worst half left
    mid = (k + 1) // 2
    right_items = items[:mid]
    left_items = list(reversed(items[mid:]))

    for idx, (name, r) in enumerate(right_items):
        x = k + 1 - r
        y = 0.60 - idx * (0.48 / max(mid, 1))
        ax.plot([x, x, k + 0.25], [axis_y, y, y], color="#333333", lw=0.9)
        ax.text(k + 0.28, y, f"{name} ({r:.2f})", ha="left", va="center", fontsize=8.5)

    for idx, (name, r) in enumerate(left_items):
        x = k + 1 - r
        y = 0.60 - idx * (0.48 / max(len(left_items), 1))
        ax.plot([x, x, 0.75], [axis_y, y, y], color="#333333", lw=0.9)
        ax.text(0.72, y, f"({r:.2f}) {name}", ha="right", va="center", fontsize=8.5)

    # Maximal cliques of methods NOT significantly different (diff <= CD)
    if np.isfinite(cd) and cd > 0:
        cliques = []
        for i in range(k):
            j = i
            while j + 1 < k and ranks[j + 1] - ranks[i] <= cd + 1e-12:
                j += 1
            if j > i:
                if not any(a <= i and j <= b for a, b in cliques):
                    cliques.append((i, j))
        for c_idx, (i, j) in enumerate(cliques):
            y_bar = axis_y - 0.055 - c_idx * 0.032
            x_hi = k + 1 - ranks[i]
            x_lo = k + 1 - ranks[j]
            ax.plot([x_lo, x_hi], [y_bar, y_bar], color="black", lw=3.0, solid_capstyle="butt")

    ax.set_title(title, fontsize=10.5, pad=6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def extract_methods_from_run(res: dict, condition: str) -> dict[str, dict]:
    """Extract comparable method-level metric dicts from a single `results.json`."""
    out = {}
    ens = res.get("ensemble", {}).get("variants", {})
    fl_comm_bytes = sum(
        (res.get("fl", {}).get(m, {}).get("upload_bytes", 0)
         + res.get("fl", {}).get(m, {}).get("broadcast_bytes", 0))
        for m in res.get("fl", {})
    )
    comm_mb = fl_comm_bytes / (1024.0 * 1024.0)
    lat = res.get("latency_ms_p95", {})
    if "ensemble" in lat and lat["ensemble"] is not None and np.isfinite(lat["ensemble"]):
        ens_lat = float(lat["ensemble"])
    elif lat:
        ens_lat = float(sum(v for k, v in lat.items()
                            if k != "ensemble" and v is not None and np.isfinite(v)))
    else:
        ens_lat = float("nan")

    # Key ensemble variants tagged with the FL condition name
    for vname in ("soft_voting", "weighted_voting", "weighted_calibrated_per_silo",
                  "weighted_calibrated_bayes", "weighted_calibrated_bayes_1pct"):
        if vname in ens:
            entry = dict(ens[vname])
            entry["comm_mb"] = comm_mb
            entry["latency_ms"] = ens_lat
            out[f"{condition}:{vname}"] = entry

    # Centralized baselines (condition-independent; deduplicated per seed)
    cent = res.get("baselines", {}).get("centralized", {})
    for m, vals in cent.items():
        entry = dict(vals)
        entry.setdefault("macro_ap", float("nan"))
        entry["comm_mb"] = 0.0
        entry["latency_ms"] = float("nan")
        out[f"centralized:{m}"] = entry

    # Local-only baselines (preserve full metric parity when available)
    loc = res.get("baselines", {}).get("local_only", {})
    for m, vals in loc.items():
        entry = dict(vals)
        entry.setdefault("pooled_ap", float("nan"))
        entry.setdefault("macro_ap", float("nan"))
        entry["comm_mb"] = 0.0
        entry["latency_ms"] = float("nan")
        out[f"local_only:{m}"] = entry
    return out


def _sort_blocks(blocks) -> list:
    if all(isinstance(b, (int, np.integer)) for b in blocks):
        return sorted(int(b) for b in blocks)
    return sorted((str(b) for b in blocks))


def analyze_runs(runs: list[tuple[str, int | str, dict]], out_dir: str,
                 primary_metric: str = "pooled_ap", alpha: float = 0.05,
                 scope: str = "primary") -> dict:
    """Aggregate `(condition, block_id, results_dict)` tuples and run statistical tests.

    `scope`:
      - `"primary"` (default): compare 1 canonical representative per FL condition
        (+ centralized/local baselines) so Friedman-Nemenyi / Holm-Wilcoxon power
        is not diluted by testing 4 correlated ensemble variants per condition.
      - `"all"`: include every complete method/variant in the statistical tests.
    """
    if scope not in ("primary", "all"):
        raise ValueError(f"Unknown scope '{scope}'. Choose 'primary' or 'all'.")
    os.makedirs(out_dir, exist_ok=True)

    # Organize by method -> block_id -> metric dict
    by_method: dict[str, dict[int | str, dict]] = {}
    blocks = _sort_blocks({blk for _, blk, _ in runs})
    for cond, blk, res in runs:
        extracted = extract_methods_from_run(res, cond)
        for mname, metrics in extracted.items():
            by_method.setdefault(mname, {})
            # Preserve first centralized/local_only observation per block if repeated
            if blk not in by_method[mname]:
                by_method[mname][blk] = metrics

    # Per-method mean +/- 95% CI across blocks
    summary_table = {}
    for mname in sorted(by_method):
        summary_table[mname] = {}
        for metric in DEFAULT_METRICS + ["comm_mb", "latency_ms"]:
            vals = [by_method[mname][s].get(metric) for s in blocks if s in by_method[mname]]
            summary_table[mname][metric] = mean_ci95(vals)

    # Build complete block matrix (blocks x methods) for statistical testing on `primary_metric`
    higher_is_better = primary_metric not in LOWER_IS_BETTER
    complete_all = [
        m for m in sorted(by_method)
        if all(
            s in by_method[m]
            and by_method[m][s].get(primary_metric) is not None
            and np.isfinite(by_method[m][s].get(primary_metric))
            for s in blocks
        )
    ]
    tested_methods = (select_primary_methods(complete_all) if scope == "primary"
                      else complete_all)

    stat_block = {}
    if len(tested_methods) >= 2 and len(blocks) >= 2:
        M = np.array(
            [[by_method[m][s][primary_metric] for m in tested_methods] for s in blocks],
            dtype=np.float64,
        )
        fried = friedman_test(M, tested_methods, higher_is_better=higher_is_better)
        nem = nemenyi_posthoc(fried["avg_ranks"], len(blocks), alpha=alpha)
        wilc = holm_wilcoxon(M, tested_methods, higher_is_better=higher_is_better, alpha=alpha)
        n_blocks = len(blocks)
        n_pairs = len(wilc)
        min_wilcoxon_p = float(2.0 / (2 ** n_blocks))
        min_holm_p = float(min(1.0, min_wilcoxon_p * max(n_pairs, 1)))
        power_warning = None
        if min_wilcoxon_p >= alpha:
            power_warning = (
                f"Underpowered Wilcoxon test: with N={n_blocks} blocks, the exact "
                f"two-sided Wilcoxon floor is 2/2^{n_blocks} = {min_wilcoxon_p:.4f} >= "
                f"alpha={alpha}, so no pairwise Wilcoxon test can reach significance. "
                "Use N >= 10 blocks (e.g., >=10 seeds or multi-partition/dataset blocks)."
            )
        elif min_holm_p >= alpha:
            power_warning = (
                f"Low Holm-Wilcoxon power: with N={n_blocks} blocks and {n_pairs} pairs, "
                f"the minimum Holm-adjusted p-value is {min_holm_p:.4f} >= alpha={alpha}. "
                "Increase N or reduce tested methods."
            )
        stat_block = {
            "primary_metric": primary_metric,
            "higher_is_better": higher_is_better,
            "scope": scope,
            "methods": tested_methods,
            "all_complete_methods": complete_all,
            "seeds": blocks,
            "n_blocks": n_blocks,
            "min_wilcoxon_p": min_wilcoxon_p,
            "min_holm_p": min_holm_p,
            "power_warning": power_warning,
            "friedman": fried,
            "nemenyi": nem,
            "holm_wilcoxon": wilc,
        }
        try:
            plot_cd_diagram(
                fried["avg_ranks"], nem["cd"],
                os.path.join(out_dir, "cd_diagram.png"),
                title=f"Critical Difference ({primary_metric}, N={n_blocks} blocks, alpha={alpha})",
            )
        except Exception as e:
            stat_block["cd_plot_error"] = str(e)

    report = {
        "seeds": blocks,
        "primary_metric": primary_metric,
        "scope": scope,
        "summary": summary_table,
        "statistics": stat_block,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(render_markdown(report))
    with open(os.path.join(out_dir, "table.tex"), "w") as f:
        f.write(render_latex(report))
    return report


def _fmt_ci(stat: dict, digits: int = 4) -> str:
    if not stat or stat.get("n", 0) == 0 or not np.isfinite(stat["mean"]):
        return "—"
    return f"{stat['mean']:.{digits}f} ± {stat['ci95']:.{digits}f}"


def render_markdown(report: dict) -> str:
    lines = [
        "# Multi-Seed Experiment Summary",
        "",
        f"- **Blocks / Seeds:** {report['seeds']} (N = {len(report['seeds'])})",
        f"- **Primary metric:** `{report['primary_metric']}` (scope: `{report.get('scope', 'primary')}`)",
        "",
        "## Performance (Mean ± 95% CI)",
        "",
        "| Method | Pooled AP | Macro AP | Rec@1%FPR | Alert% | Savings | Bayes@1% | ECE | Comm (MB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, row in report["summary"].items():
        al = row.get("alert_rate")
        al_str = (f"{100 * al['mean']:.2f}%" if al and al.get("n", 0) > 0 and np.isfinite(al["mean"])
                  else "—")
        lines.append(
            f"| `{m}` | {_fmt_ci(row['pooled_ap'])} | {_fmt_ci(row['macro_ap'])} | "
            f"{_fmt_ci(row['recall_at_fpr_001'], 3)} | {al_str} | "
            f"{_fmt_ci(row['savings'], 3)} | "
            f"{_fmt_ci(row.get('bayes_savings_at_1pct'), 3)} | {_fmt_ci(row['ece'], 4)} | "
            f"{_fmt_ci(row['comm_mb'], 2)} |"
        )
    lines += [
        "",
        "> _Note: `Savings` evaluates each variant's deployed decision rule (per-silo 1% alert cap for `*_bayes_1pct`), "
        "whereas `Bayes@1%` evaluates a pooled 1% global alert cap across all test transactions._",
    ]

    st = report.get("statistics", {})
    if st:
        fr = st["friedman"]
        nem = st["nemenyi"]
        lines += [
            "",
            f"## Friedman–Nemenyi Analysis (`{st['primary_metric']}`, scope=`{st.get('scope', 'primary')}`)",
            "",
        ]
        if st.get("power_warning"):
            lines += [f"> **Warning:** {st['power_warning']}", ""]
        lines += [
            f"- **Friedman $\\chi^2$:** `{fr['chi2']:.4f}` (p = `{fr['p_value']:.4g}`, Kendall's W = `{fr['kendalls_w']:.3f}`)",
            f"- **Nemenyi CD ($\\alpha={nem['alpha']}$):** `{nem['cd']:.3f}` (q = `{nem['q_alpha']:.3f}`)",
            f"- **Wilcoxon floor ($N={st.get('n_blocks', len(report['seeds']))}$):** `p_raw >= {st.get('min_wilcoxon_p', float('nan')):.4g}`, `p_holm >= {st.get('min_holm_p', float('nan')):.4g}`",
            "",
            "### Average Ranks (1 = Best)",
            "",
            "| Method | Avg Rank |",
            "|---|---|",
        ]
        for m, r in sorted(fr["avg_ranks"].items(), key=lambda kv: kv[1]):
            lines.append(f"| `{m}` | {r:.3f} |")

        sig_nem = [p for p in nem["pairs"] if p["significant"]]
        lines += ["", f"### Significant Nemenyi Pairs ({len(sig_nem)} / {len(nem['pairs'])})", ""]
        if sig_nem:
            lines += ["| Method A | Method B | |Rank Diff| | p-value |", "|---|---|---|---|"]
            for p in sig_nem:
                lines.append(f"| `{p['method_a']}` | `{p['method_b']}` | {p['rank_diff']:.3f} | {p['p_value']:.4g} |")
        else:
            lines.append("_No pairs exceeded CD (increase number of seeds/blocks for higher power)._")

        lines += [
            "",
            "### Pairwise Holm-Wilcoxon Tests (Top 10 by p_holm)",
            "",
            "| Method A | Method B | Mean Diff | r_rb | p_raw | p_holm | Sig |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in sorted(st["holm_wilcoxon"], key=lambda x: x["p_holm"])[:10]:
            lines.append(
                f"| `{row['method_a']}` | `{row['method_b']}` | {row['mean_diff']:+.4f} | "
                f"{row['rank_biserial']:+.3f} | {row['p_raw']:.4g} | {row['p_holm']:.4g} | "
                f"{'yes' if row['significant'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def render_latex(report: dict) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-seed evaluation results (mean $\pm$ 95\% CI across "
        + f"{len(report['seeds'])}" + r" blocks).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Pooled AP & Macro AP & Bayes@1\% & ECE & Comm (MB) \\",
        r"\midrule",
    ]
    for m, row in report["summary"].items():
        esc = m.replace("_", r"\_")
        def _tex(st, d=4):
            if not st or st.get("n", 0) == 0 or not np.isfinite(st["mean"]):
                return "--"
            return f"${st['mean']:.{d}f} \\pm {st['ci95']:.{d}f}$"
        lines.append(
            f"{esc} & {_tex(row['pooled_ap'])} & {_tex(row['macro_ap'])} & "
            f"{_tex(row.get('bayes_savings_at_1pct') or row.get('bayes_savings'), 3)} & "
            f"{_tex(row['ece'], 4)} & {_tex(row['comm_mb'], 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def load_suite_dir(suite_dir: str) -> list[tuple[str, int | str, dict]]:
    """Discover `<suite_dir>/<condition>/seed_<N>/results.json` runs, or
    multi-partition `<suite_dir>/<partition>/<condition>/seed_<N>/results.json`
    blocks (pooled as block_id='<partition>:seed_<N>')."""
    runs: list[tuple[str, int | str, dict]] = []
    direct = sorted(glob.glob(os.path.join(suite_dir, "*", "seed_*", "results.json")))
    if direct:
        for path in direct:
            parts = os.path.normpath(path).split(os.sep)
            cond = parts[-3]
            seed = int(parts[-2].replace("seed_", ""))
            with open(path) as f:
                runs.append((cond, seed, json.load(f)))
        return runs

    nested = sorted(glob.glob(os.path.join(suite_dir, "*", "*", "seed_*", "results.json")))
    for path in nested:
        parts = os.path.normpath(path).split(os.sep)
        part = parts[-4]
        cond = parts[-3]
        seed = parts[-2]
        with open(path) as f:
            runs.append((cond, f"{part}:{seed}", json.load(f)))
    return runs


def main():
    ap = argparse.ArgumentParser(description="Statistical analysis for multi-seed FL suite")
    ap.add_argument("--suite-dir", required=True,
                    help="Directory with <condition>/seed_<N>/results.json or <partition>/<condition>/seed_<N>/results.json")
    ap.add_argument("--metric", default="pooled_ap")
    ap.add_argument("--scope", default="primary", choices=["primary", "all"],
                    help="Test 1 canonical representative per condition (primary) or all variants (all)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="Output directory (defaults to --suite-dir)")
    args = ap.parse_args()

    runs = load_suite_dir(args.suite_dir)
    if not runs:
        raise SystemExit(f"No results.json files found under {args.suite_dir}")
    out_dir = args.out or args.suite_dir
    rep = analyze_runs(runs, out_dir, primary_metric=args.metric, alpha=args.alpha,
                       scope=args.scope)
    print(render_markdown(rep))
    print(f"Saved summary.json, summary.md, table.tex, cd_diagram.png -> {out_dir}")


if __name__ == "__main__":
    main()
