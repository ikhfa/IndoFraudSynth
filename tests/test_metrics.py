import numpy as np

from fl_experiments.metrics import (
    ap, recall_at_fpr, fraud_cost, cost_threshold, savings,
    ece, brier, bayes_cost_predict, bayes_savings,
    default_c_fp, alert_rate, fpr_flags,
)
from fl_experiments.ensemble import fit_calibrators


def test_separable_scores():
    y = np.array([0] * 900 + [1] * 100)
    s = np.concatenate([np.random.default_rng(0).uniform(0, 0.4, 900),
                        np.random.default_rng(1).uniform(0.6, 1.0, 100)])
    assert ap(y, s) > 0.99
    assert recall_at_fpr(y, s, 0.01) > 0.99


def test_cost_and_savings():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    amt = np.array([100, 200, 300, 1_000_000, 2_000_000])
    thr, c = cost_threshold(y, s, amt, c_fp=7500)
    assert c <= fraud_cost(y, s, 0.99, amt, 7500)
    sv = savings(y, s, thr, amt, 7500)
    assert -1.0 <= sv <= 1.0
    # flagging nothing costs the full fraud loss
    assert fraud_cost(y, s, 1.1, amt, 7500) == 3_000_000


def test_ap_degenerate():
    assert np.isnan(ap(np.zeros(10), np.random.rand(10)))


def test_ece_brier_and_bayes_cost():
    y = np.array([0, 0, 0, 1, 1])
    p_good = np.array([0.01, 0.02, 0.05, 0.95, 0.90])
    p_bad = np.array([0.8, 0.8, 0.8, 0.2, 0.2])
    assert ece(y, p_good) < ece(y, p_bad)
    assert brier(y, p_good) < brier(y, p_bad)

    # Bayes rule p_i * A_i >= c_fp: flags high-value txn even at p=0.05, ignores tiny txn at p=0.05
    p = np.array([0.05, 0.05, 0.001, 0.9])
    amt = np.array([50_000, 1_000_000, 1_000_000, 500_000])
    flags = bayes_cost_predict(p, amt, c_fp=7500.0)
    assert flags.tolist() == [0, 1, 0, 1]
    # Alert rate cap strictly limits total alerts
    capped = bayes_cost_predict(p, amt, c_fp=7500.0, max_alert_rate=0.25)
    assert int(capped.sum()) == 1
    bs = bayes_savings(np.array([0, 1, 0, 1]), p, amt, c_fp=7500.0)
    assert bs == 1.0


def test_platt_calibrator_activates_for_small_silo():
    from fl_experiments.ensemble import PlattCalibrator
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 12)
    # Deliberately miscalibrated (compressed near 1e-13 like unnormalized GBDT)
    s = np.concatenate([rng.uniform(1e-14, 1e-12, 500), rng.uniform(5e-12, 5e-11, 12)])
    cals, methods, oof = fit_calibrators({"xgb": {"s1": s}}, {"s1": y}, seed=42)
    assert methods["xgb"]["s1"] == "platt"
    assert cals["xgb"]["s1"] is not None
    assert cals["xgb"]["s1"].slope_ > 0.0
    assert np.isfinite(oof["xgb"]["s1"]).all()
    assert (oof["xgb"]["s1"] >= 0).all() and (oof["xgb"]["s1"] <= 1).all()
    # Ranking at 1e-13 scale MUST be preserved (previously destroyed by 1e-7 clip!)
    pred = cals["xgb"]["s1"].predict(s)
    assert np.isclose(ap(y, pred), ap(y, s), atol=1e-6)
    assert ap(y, pred) > 0.95

    # Monotonicity guard: even on anti-correlated noisy small silo, slope stays > 0
    cal_noisy = PlattCalibrator().fit(np.linspace(0.01, 0.99, 20), np.array([1] * 5 + [0] * 15))
    assert cal_noisy.slope_ >= 1e-3


def test_default_c_fp_and_operational_alert_budget():
    assert default_c_fp("indofraud") == 7500.0
    assert default_c_fp("ulb") == 5.0
    assert default_c_fp("ieee_cis") == 5.0
    assert default_c_fp("sparkov") == 5.0

    y = np.array([0] * 990 + [1] * 10)
    # High-value IDR transfers where unconstrained Bayes over-alerts at c_fp=7500
    p = np.concatenate([np.full(990, 0.005), np.full(10, 0.5)])
    amt = np.full(1000, 5_000_000.0)
    unconstrained = bayes_cost_predict(p, amt, c_fp=7500.0)
    capped = bayes_cost_predict(p, amt, c_fp=7500.0, max_alert_rate=0.01)
    assert alert_rate(unconstrained) == 1.0
    assert alert_rate(capped) <= 0.01
    assert fpr_flags(y, capped) == 0.0
    assert bayes_savings(y, p, amt, c_fp=7500.0, max_alert_rate=0.01) == 1.0


def test_hierarchical_platt_global_fallback_and_oof_consistency():
    """Regression test:
    1. Silos with 5..7 positives must NOT fall back to uncalibrated identity inside 3-fold OOF.
    2. Tiny silos (<5 positives) must receive FL-safe `platt_global` fallback so pooled
       scores across silos stay on one calibrated scale (preventing the PR cliff).
    """
    rng = np.random.default_rng(7)
    # Silo A: 20 positives, compressed unnormalized scores (~1e-12)
    ya = np.array([0] * 1000 + [1] * 20)
    sa = np.concatenate([rng.uniform(1e-14, 1e-12, 1000), rng.uniform(5e-12, 5e-11, 20)])
    # Silo B: 6 positives (edge case where 3-fold y[tr] has only 4 positives!)
    yb = np.array([0] * 500 + [1] * 6)
    sb = np.concatenate([rng.uniform(1e-14, 1e-12, 500), rng.uniform(5e-12, 5e-11, 6)])
    # Silo C: 2 positives (<5 threshold -> previously raw identity!)
    yc = np.array([0] * 400 + [1] * 2)
    sc = np.concatenate([rng.uniform(1e-14, 1e-12, 400), rng.uniform(5e-12, 5e-11, 2)])

    cals, methods, oof = fit_calibrators(
        {"xgb": {"sA": sa, "sB": sb, "sC": sc}},
        {"sA": ya, "sB": yb, "sC": yc},
        seed=42,
    )
    assert methods["xgb"]["sA"] == "platt_shrunk"
    assert methods["xgb"]["sB"] == "platt_shrunk"
    assert methods["xgb"]["sC"] == "platt_global"
    assert cals["xgb"]["sC"] is not None
    # Silo B OOF must be on calibrated probability scale (~0.01), NOT raw 1e-12!
    assert oof["xgb"]["sB"].max() > 0.05
    # Pooled calibrated predictions across all 3 silos (incl. tiny sC) must preserve high AP
    p_all = np.concatenate([cals["xgb"][k].predict(s) for k, s in [("sA", sa), ("sB", sb), ("sC", sc)]])
    y_all = np.concatenate([ya, yb, yc])
    assert ap(y_all, p_all) > 0.95
    assert ece(y_all, p_all) < 0.02


def test_exact_cost_threshold_rare_tail_and_medium_silo_platt():
    """Regression tests:
    1. Exact O(N log N) cost_threshold finds rare-event cutoffs above the 99.9th
       percentile (previously missed by 250-quantile grid).
    2. Medium rare-event silos (pos=120 < 500) use smooth Platt scaling rather than
       step-quantized Isotonic regression, preserving exact per-silo AP ranking.
    """
    rng = np.random.default_rng(123)
    n_neg, n_pos = 10_000, 5  # 0.05% fraud -> optimal cutoff at 99.95th percentile!
    y = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
    s = np.concatenate([rng.uniform(0.0, 0.2, n_neg), rng.uniform(0.8, 0.99, n_pos)])
    amt = np.full(n_neg + n_pos, 1_000_000.0)
    thr, best_c = cost_threshold(y, s, amt, c_fp=7500.0)
    # Exact optimal threshold catches all 5 frauds with 0 false positives (cost == 0)
    assert best_c == 0.0
    assert fraud_cost(y, s, thr, amt, c_fp=7500.0) == 0.0
    assert savings(y, s, thr, amt, c_fp=7500.0) == 1.0

    # Medium silo with 120 positives must choose smooth Platt (no Isotonic tie collapse)
    y_med = np.array([0] * 5000 + [1] * 120, dtype=np.int8)
    s_med = np.concatenate([rng.beta(1, 20, 5000), rng.beta(4, 2, 120)])
    cals, methods, _ = fit_calibrators({"xgb": {"s_med": s_med}}, {"s_med": y_med}, seed=42)
    assert methods["xgb"]["s_med"] == "platt"
    pred_med = cals["xgb"]["s_med"].predict(s_med)
    assert np.isclose(ap(y_med, pred_med), ap(y_med, s_med), atol=1e-6)


def test_logit_blend_and_post_global_platt_calibration():
    """Verify calibrated log-odds blending beats linear prob averaging on rare-event
    complementary models and post-blend global Platt preserves exact AP while lowering ECE."""
    from scipy.special import expit
    from fl_experiments.ensemble import (
        combine_scores_logit, fit_global_post_calibrator, search_weights,
    )

    rng = np.random.default_rng(77)
    n_neg, n_pos = 12_000, 30  # 0.25% rare fraud
    y = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
    # Two complementary models in the rare-event regime: each occasionally fires false
    # alarms on negatives that the other model strongly rejects (~1e-4), where linear
    # probability averaging acts like soft-OR while log-odds blending rejects them.
    z1 = np.where(y == 1, rng.normal(-1.5, 1.3, len(y)), rng.normal(-6.2, 2.2, len(y)))
    z2 = np.where(y == 1, rng.normal(-1.5, 1.3, len(y)), rng.normal(-6.2, 2.2, len(y)))
    p1 = expit(z1)
    p2 = expit(z2)

    # Split positives across both silos cleanly
    idx1 = np.concatenate([np.arange(0, 6000), np.arange(n_neg, n_neg + 15)])
    idx2 = np.concatenate([np.arange(6000, n_neg), np.arange(n_neg + 15, n_neg + n_pos)])
    s_val = {"m1": {"s1": p1[idx1], "s2": p1[idx2]}, "m2": {"s1": p2[idx1], "s2": p2[idx2]}}
    y_val = {"s1": y[idx1], "s2": y[idx2]}

    w_lin, _ = search_weights(s_val, y_val, n_draws=100, seed=42, logit_space=False)
    w_lg, _ = search_weights(s_val, y_val, n_draws=100, seed=42, logit_space=True)
    lin_all = np.concatenate([w_lin["m1"] * s_val["m1"][s] + w_lin["m2"] * s_val["m2"][s] for s in ("s1", "s2")])
    lg_map = combine_scores_logit(s_val, w_lg, silos=["s1", "s2"])
    lg_all = np.concatenate([lg_map[s] for s in ("s1", "s2")])
    y_all = np.concatenate([y_val["s1"], y_val["s2"]])

    # Log-odds blend must strictly beat linear prob averaging on rare-event AP
    assert ap(y_all, lg_all) > ap(y_all, lin_all) + 0.05

    # Post-blend global Platt must preserve exact pooled & per-silo AP and keep valid probabilities
    post_cal, oof_map = fit_global_post_calibrator(lg_map, y_val, seed=42)
    assert post_cal is not None
    post_all = np.concatenate([post_cal.predict(lg_map[s]) for s in ("s1", "s2")])
    assert np.isclose(ap(y_all, post_all), ap(y_all, lg_all), atol=1e-6)
    assert ece(y_all, post_all) <= ece(y_all, lg_all) + 1e-4
    for s in ("s1", "s2"):
        assert np.isfinite(oof_map[s]).all()
        assert (oof_map[s] >= 0.0).all() and (oof_map[s] <= 1.0).all()


def test_extreme_dirichlet_platt_prior_shift_prevents_ece_collapse():
    """Regression test: under extreme Dirichlet(0.1) label skew, a positive-dominated
    silo (e.g., 1000 pos / 5 neg) must NOT hijack the federated global Platt fallback
    and push zero-fraud silos to p -> 1.0 (which previously caused ECE ~ 0.96)."""
    from scipy.special import expit
    from fl_experiments.ensemble import fit_global_post_calibrator, safe_cal_predict

    rng = np.random.default_rng(3)

    def _mk(n_neg, n_pos, shift=-14.0):
        y = np.array([0] * n_neg + [1] * n_pos, dtype=np.int8)
        z = np.where(y == 1, rng.normal(shift + 2.0, 1.5, len(y)), rng.normal(shift, 1.5, len(y)))
        return expit(z), y

    s0, y0 = _mk(2000, 15)
    s1, y1 = _mk(50_000, 8)
    s4, y4 = _mk(5, 1000)      # extreme 99.5% fraud silo
    s6, y6 = _mk(100_000, 0)   # giant 0% fraud silo -> gets platt_global
    sc_val = {"c0": s0, "c1": s1, "c4": s4, "c6": s6}
    yv = {"c0": y0, "c1": y1, "c4": y4, "c6": y6}

    cals, methods, _ = fit_calibrators({"m": sc_val}, yv, seed=42)
    assert methods["m"]["c6"] == "platt_global"
    p_map = {k: safe_cal_predict(cals["m"][k], sc_val[k]) for k in sc_val}
    post, _ = fit_global_post_calibrator(p_map, yv, seed=42)
    p_post = {k: safe_cal_predict(post, p_map[k]) for k in sc_val}

    p_all = np.concatenate([p_post[k] for k in sc_val])
    y_all = np.concatenate([yv[k] for k in sc_val])
    assert p_post["c6"].mean() < 0.02, f"Zero-fraud silo pushed too high: {p_post['c6'].mean()}"
    assert ece(y_all, p_all) < 0.02, f"Extreme Dirichlet ECE collapsed: {ece(y_all, p_all)}"
    assert ap(y_all, p_all) > 0.90, f"Extreme Dirichlet pooled AP collapsed: {ap(y_all, p_all)}"

    # Verify _eval_variant clearly separates per-silo vs pooled 1%-capped Bayes savings
    from types import SimpleNamespace
    from fl_experiments.ensemble import _eval_variant
    fake_clients = {
        k: SimpleNamespace(y={"test": yv[k]}, amt={"test": np.full(len(yv[k]), 500_000.0)})
        for k in sc_val
    }
    ev = _eval_variant(p_post, fake_clients, None, c_fp=7500.0, use_bayes=True, max_alert_rate=0.01)
    assert "per_silo_bayes_savings_at_1pct" in ev
    assert np.isclose(ev["savings"], ev["per_silo_bayes_savings_at_1pct"], atol=1e-9)
    # Pooled 1% cap covers all frauds in 99.5%-fraud silo c4, while per-silo 1% cap truncates c4
    assert ev["bayes_savings_at_1pct"] > ev["per_silo_bayes_savings_at_1pct"] + 0.5

