import json
import os
import sys

import numpy as np
import pytest

from fl_experiments.stats import (
    mean_ci95, friedman_test, nemenyi_posthoc, holm_wilcoxon,
    plot_cd_diagram, analyze_runs, select_primary_methods, load_suite_dir,
)
from fl_experiments.run_suite import parse_partition_spec, build_run_cmd


def test_mean_ci95():
    st = mean_ci95([0.80, 0.82, 0.84, 0.86, 0.88])
    assert st["n"] == 5
    assert np.isclose(st["mean"], 0.84)
    assert st["ci95"] > 0.0
    assert mean_ci95([0.5])["ci95"] == 0.0
    assert mean_ci95([])["n"] == 0


def test_friedman_nemenyi_and_holm_wilcoxon(tmp_path):
    # 10 blocks x 4 clearly ordered methods -> highly significant Friedman + Nemenyi
    rng = np.random.default_rng(0)
    n = 10
    M = np.column_stack([
        0.50 + rng.normal(0, 0.01, n),
        0.65 + rng.normal(0, 0.01, n),
        0.78 + rng.normal(0, 0.01, n),
        0.92 + rng.normal(0, 0.01, n),
    ])
    names = ["m1", "m2", "m3", "m4"]
    fr = friedman_test(M, names, higher_is_better=True)
    assert fr["p_value"] < 0.001
    assert 0.9 <= fr["kendalls_w"] <= 1.0
    assert fr["avg_ranks"]["m4"] < fr["avg_ranks"]["m3"] < fr["avg_ranks"]["m2"] < fr["avg_ranks"]["m1"]

    # Check exact Nemenyi q_alpha for k=5, alpha=0.05 against Demsar (2006) table (~2.728)
    nem5 = nemenyi_posthoc({f"m{i}": float(i) for i in range(1, 6)}, n_blocks=10, alpha=0.05)
    assert np.isclose(nem5["q_alpha"], 2.728, atol=0.005)

    nem = nemenyi_posthoc(fr["avg_ranks"], n_blocks=n, alpha=0.05)
    assert nem["cd"] > 0
    sig_pairs = {(p["method_a"], p["method_b"]) for p in nem["pairs"] if p["significant"]}
    assert ("m1", "m4") in sig_pairs

    hw = holm_wilcoxon(M, names, higher_is_better=True, alpha=0.05)
    assert len(hw) == 6
    for row in hw:
        assert 0.0 <= row["p_raw"] <= row["p_holm"] <= 1.0
        assert -1.0 <= row["rank_biserial"] <= 1.0

    cd_path = str(tmp_path / "cd.png")
    plot_cd_diagram(fr["avg_ranks"], nem["cd"], cd_path)
    assert os.path.exists(cd_path) and os.path.getsize(cd_path) > 1000


def test_analyze_runs_generates_artifacts(tmp_path):
    runs = []
    for seed in range(1, 6):
        for cond, bonus in [("baseline", 0.0), ("normalized", 0.05)]:
            res = {
                "fl": {"xgboost": {"upload_bytes": 100000, "broadcast_bytes": 200000}},
                "latency_ms_p95": {"xgboost": 5.0},
                "ensemble": {
                    "variants": {
                        "soft_voting": {
                            "pooled_ap": 0.60 + bonus + 0.005 * seed,
                            "macro_ap": 0.55 + bonus,
                            "recall_at_fpr_001": 0.80,
                            "f1": 0.30,
                            "savings": 0.90,
                            "bayes_savings": 0.85 + bonus,
                            "ece": 0.01,
                            "brier": 0.002,
                        },
                        "weighted_calibrated_bayes": {
                            "pooled_ap": 0.68 + bonus + 0.005 * seed,
                            "macro_ap": 0.63 + bonus,
                            "recall_at_fpr_001": 0.86,
                            "f1": 0.35,
                            "savings": 0.95,
                            "bayes_savings": 0.95 + 0.02 * bonus,
                            "ece": 0.004,
                            "brier": 0.001,
                        },
                    }
                },
                "baselines": {
                    "centralized": {
                        "xgboost": {
                            "pooled_ap": 0.64 + 0.004 * seed,
                            "recall_at_fpr_001": 0.82,
                            "f1": 0.32,
                            "savings": 0.93,
                            "bayes_savings": 0.94,
                            "ece": 0.005,
                            "brier": 0.0015,
                        }
                    }
                },
            }
            runs.append((cond, seed, res))

    out_dir = str(tmp_path / "stats_out")
    rep = analyze_runs(runs, out_dir, primary_metric="pooled_ap")
    assert rep["statistics"]["friedman"]["p_value"] < 0.05
    for fname in ("summary.json", "summary.md", "table.tex", "cd_diagram.png"):
        assert os.path.exists(os.path.join(out_dir, fname))
    with open(os.path.join(out_dir, "table.tex")) as f:
        tex = f.read()
    assert r"\begin{tabular}" in tex


def test_run_suite_cli_helpers(tmp_path):
    assert parse_partition_spec("profile") == ("profile", "profile", None)
    assert parse_partition_spec("dirichlet:0.5") == ("dirichlet_a0p5", "dirichlet", 0.5)

    old = sys.argv
    sys.argv = [
        "run_suite", "--config", "configs/quick.yaml", "--seeds", "1,2",
        "--conditions", "baseline,normalized_fedbn", "--partitions", "profile,dirichlet:0.5",
        "--out", str(tmp_path / "suite"), "--dry-run",
    ]
    try:
        from fl_experiments.run_suite import main
        main()
    finally:
        sys.argv = old
    assert (tmp_path / "suite" / "profile").is_dir()
    assert (tmp_path / "suite" / "dirichlet_a0p5").is_dir()


def test_stats_primary_scope_power_warning_and_multiparts(tmp_path):
    """Regression test: primary scope prevents 20-method dilution, N=5 emits an
    explicit Wilcoxon power warning, N=10 reaches Holm-Wilcoxon significance,
    and load_suite_dir pools multi-partition suites cleanly."""
    def _mk_res(base_ap, seed):
        def _v(val):
            return {
                "pooled_ap": val, "macro_ap": val - 0.02, "recall_at_fpr_001": 0.80,
                "f1": 0.30, "alert_rate": 0.008, "fpr": 0.005,
                "savings": 0.90, "bayes_savings": 0.92, "bayes_savings_at_1pct": 0.88,
                "ece": 0.005, "brier": 0.001,
            }
        return {
            "fl": {"xgboost": {"upload_bytes": 100000, "broadcast_bytes": 200000}},
            "latency_ms_p95": {"xgboost": 4.0},
            "ensemble": {
                "variants": {
                    "soft_voting": _v(base_ap - 0.02 + 0.002 * seed),
                    "weighted_voting": _v(base_ap + 0.002 * seed),
                    "weighted_calibrated_per_silo": _v(base_ap + 0.01 + 0.002 * seed),
                    "weighted_calibrated_bayes": _v(base_ap + 0.02 + 0.002 * seed),
                    "weighted_calibrated_bayes_1pct": _v(base_ap + 0.025 + 0.002 * seed),
                }
            },
            "baselines": {"centralized": {"xgboost": _v(0.66 + 0.002 * seed)}},
        }

    # 1. N=5 seeds: primary scope selects 1 representative/condition + centralized (4 methods, not 16)
    runs_5 = []
    for seed in range(1, 6):
        runs_5.append(("baseline", seed, _mk_res(0.58, seed)))
        runs_5.append(("normalized", seed, _mk_res(0.68, seed)))
        runs_5.append(("normalized_fedbn", seed, _mk_res(0.74, seed)))
    rep5 = analyze_runs(runs_5, str(tmp_path / "out5"), primary_metric="pooled_ap", scope="primary")
    tested = set(rep5["statistics"]["methods"])
    assert tested == {
        "baseline:weighted_voting",
        "normalized:weighted_calibrated_bayes_1pct",
        "normalized_fedbn:weighted_calibrated_bayes_1pct",
        "centralized:xgboost",
    }
    assert len(rep5["summary"]) > len(tested)
    # N=5 has Wilcoxon floor 2/32 = 0.0625 >= 0.05 -> must emit power warning
    assert rep5["statistics"]["power_warning"] is not None
    assert np.isclose(rep5["statistics"]["min_wilcoxon_p"], 0.0625)
    with open(tmp_path / "out5" / "summary.md") as f:
        assert "Warning:" in f.read()

    # 2. Multi-partition suite (2 partitions x 5 seeds = 10 blocks) -> Holm-Wilcoxon significant!
    suite_root = tmp_path / "multi_suite"
    for part, offset in [("profile", 0.0), ("dirichlet_a0p5", -0.01)]:
        for seed in range(1, 6):
            for cond, base_ap in [("baseline", 0.58), ("normalized", 0.68), ("normalized_fedbn", 0.75)]:
                d = suite_root / part / cond / f"seed_{seed}"
                d.mkdir(parents=True)
                with open(d / "results.json", "w") as f:
                    json.dump(_mk_res(base_ap + offset, seed), f)
    pooled_runs = load_suite_dir(str(suite_root))
    assert len(pooled_runs) == 30
    rep10 = analyze_runs(pooled_runs, str(tmp_path / "out10"), primary_metric="pooled_ap", scope="primary")
    assert rep10["statistics"]["n_blocks"] == 10
    assert rep10["statistics"]["power_warning"] is None
    assert any(row["significant"] for row in rep10["statistics"]["holm_wilcoxon"])


def test_select_primary_methods_deduplicates_baselines():
    """Primary scope must pick 1 anchor each for centralized and local_only
    (preferring ensemble) rather than all 7 individual baseline models."""
    complete = [
        "centralized:xgboost", "centralized:catboost", "centralized:mlp", "centralized:ensemble",
        "local_only:xgboost", "local_only:catboost", "local_only:mlp", "local_only:ensemble",
        "baseline:soft_voting", "baseline:weighted_voting", "baseline:weighted_calibrated_bayes_1pct",
        "normalized_fedbn:weighted_voting", "normalized_fedbn:weighted_calibrated_bayes_1pct",
    ]
    sel = set(select_primary_methods(complete))
    assert sel == {
        "centralized:ensemble",
        "local_only:ensemble",
        "baseline:weighted_voting",
        "normalized_fedbn:weighted_calibrated_bayes_1pct",
    }
