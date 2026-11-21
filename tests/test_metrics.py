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
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 12)
    # Deliberately miscalibrated (compressed near 1e-4 like unnormalized GBDT)
    s = np.concatenate([rng.uniform(1e-6, 1e-4, 500), rng.uniform(5e-4, 5e-3, 12)])
    cals, methods, oof = fit_calibrators({"xgb": {"s1": s}}, {"s1": y}, seed=42)
    assert methods["xgb"]["s1"] == "platt"
    assert cals["xgb"]["s1"] is not None
    assert np.isfinite(oof["xgb"]["s1"]).all()
    assert (oof["xgb"]["s1"] >= 0).all() and (oof["xgb"]["s1"] <= 1).all()


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
