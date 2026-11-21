"""Ensemble evaluation: soft voting, weighted voting (source-paper replication),
and the PROPOSED per-silo calibrated + cost-optimal / Bayes minimum-risk decision layer.

Weight search: Dirichlet random search + coordinate refinement on macro validation
AUC-PR (evaluated via scalar per-silo AP reports). For the calibrated ensemble,
per-silo calibrators are fitted out-of-fold on `val` BEFORE weight search so all
heterogeneous models (XGBoost, CatBoost, MLP) are blended on a common probability
scale without overfitting `val`.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .metrics import (
    ap, recall_at_fpr, f1_at, cost_threshold, fraud_cost, fraud_cost_flags,
    bayes_cost_predict, bayes_savings, ece, brier, alert_rate, fpr_flags,
)

MODEL_TYPES = ["xgboost", "catboost", "mlp"]


class PlattCalibrator:
    """2-parameter Platt / logit-temperature calibrator (low-variance for small silos)."""

    def __init__(self, c: float = 10.0):
        self.lr = LogisticRegression(C=c, solver="lbfgs", max_iter=200)

    @staticmethod
    def _to_logit(s: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(s, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr.fit(self._to_logit(scores), np.asarray(y, dtype=int))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._to_logit(scores))[:, 1]


def _make_calibrator(y: np.ndarray):
    """Choose low-variance Platt scaling for 5..49 positives, Isotonic for >= 50."""
    pos = int(np.asarray(y).sum())
    neg = int(len(y) - pos)
    if pos >= 50 and neg >= 50:
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0), "isotonic"
    if pos >= 5 and neg >= 5:
        return PlattCalibrator(), "platt"
    return None, "identity"


def gather_scores(clients, payloads, split):
    """{model: {silo: scores}} for clients that have a trained payload."""
    first_val = next(iter(clients.values()))
    is_nested = isinstance(first_val, dict)
    out = {}
    for mt, payload in payloads.items():
        if payload is None:
            continue
        cmap = clients[mt] if is_nested else clients
        out[mt] = {name: c.predict(payload, split) for name, c in cmap.items()}
    return out


def _macro_ap(scores_by_silo, y_by_silo):
    vals = [ap(y_by_silo[s], scores_by_silo[s]) for s in scores_by_silo
            if 0 < y_by_silo[s].sum() < len(y_by_silo[s])]
    return float(np.mean(vals)) if vals else float("nan")


def search_weights(val_scores, y_vals, n_draws=300, seed=42):
    models = list(val_scores.keys())
    silos = list(y_vals.keys())
    rng = np.random.default_rng(seed)

    def objective(w):
        per_silo = {}
        for s in silos:
            per_silo[s] = sum(w[i] * val_scores[m][s] for i, m in enumerate(models))
        v = _macro_ap(per_silo, y_vals)
        return 0.0 if np.isnan(v) else v

    best_w, best_v = np.ones(len(models)) / len(models), -1.0
    draws = rng.dirichlet(np.ones(len(models)), n_draws)
    for w in draws:
        v = objective(w)
        if v > best_v:
            best_v, best_w = v, w
    # coordinate refinement
    for _ in range(3):
        for i in range(len(models)):
            for delta in (-0.1, -0.03, 0.03, 0.1):
                w = best_w.copy()
                w[i] = max(w[i] + delta, 0.0)
                if w.sum() == 0:
                    continue
                w = w / w.sum()
                v = objective(w)
                if v > best_v:
                    best_v, best_w = v, w
    return {m: float(w) for m, w in zip(models, best_w)}, float(best_v)


def fit_calibrators(val_scores, y_vals, seed: int = 42):
    """Per-silo calibration (Platt for 5..49 positives, Isotonic for >= 50).

    Returns `(cals, methods, oof_scores)` where `oof_scores` are strictly
    out-of-fold calibrated validation scores (preventing overfitting during
    subsequent ensemble weight and threshold optimization).
    """
    cals, methods, oof = {}, {}, {}
    for m, by_silo in val_scores.items():
        cals[m] = {}; methods[m] = {}; oof[m] = {}
        for s, scores in by_silo.items():
            y = np.asarray(y_vals[s], dtype=int)
            scores = np.asarray(scores, dtype=np.float64)
            cal, kind = _make_calibrator(y)
            methods[m][s] = kind
            if cal is None:
                cals[m][s] = None
                oof[m][s] = scores.copy()
                continue
            cal.fit(scores, y)
            cals[m][s] = cal
            pos = int(y.sum())
            n_splits = min(3, pos, int(len(y) - pos))
            if n_splits >= 2:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                pred = np.empty_like(scores, dtype=np.float64)
                for tr, va in skf.split(scores.reshape(-1, 1), y):
                    fcal, _ = _make_calibrator(y[tr])
                    if fcal is None:
                        pred[va] = scores[va]
                    else:
                        fcal.fit(scores[tr], y[tr])
                        pred[va] = fcal.predict(scores[va])
                oof[m][s] = pred
            else:
                oof[m][s] = cal.predict(scores)
    return cals, methods, oof


def _eval_variant(test_scores, clients, thr_map, c_fp, use_bayes=False,
                  max_alert_rate: float | None = None):
    """Evaluate a test-score dict across silos.

    If `use_bayes=True`, applies Bayes minimum-risk decisions `p_i * A_i >= c_fp`
    directly on calibrated probabilities instead of a flat threshold (optionally
    subject to a per-silo `max_alert_rate` operational budget).
    """
    ys, ss, amts, flags = [], [], [], []
    per_silo_ap = {}
    total_cost, total_loss = 0.0, 0.0
    for s, scores in test_scores.items():
        c = clients[s]
        y = c.y["test"]; amt = c.amt["test"]
        if use_bayes:
            fl = bayes_cost_predict(scores, amt, c_fp=c_fp, max_alert_rate=max_alert_rate)
            cost = fraud_cost_flags(y, fl, amt, c_fp)
        else:
            thr = thr_map[s]
            fl = (scores >= thr).astype(int)
            cost = fraud_cost(y, scores, thr, amt, c_fp)
        ys.append(y); ss.append(scores); amts.append(amt); flags.append(fl)
        a = ap(y, scores)
        if not np.isnan(a):
            per_silo_ap[s] = float(a)
        total_cost += cost
        total_loss += float(amt[y == 1].sum())
    y_all = np.concatenate(ys); s_all = np.concatenate(ss)
    amt_all = np.concatenate(amts); f_all = np.concatenate(flags)
    b_flags_all = bayes_cost_predict(s_all, amt_all, c_fp=c_fp)
    out = {
        "pooled_ap": ap(y_all, s_all),
        "macro_ap": float(np.mean(list(per_silo_ap.values()))) if per_silo_ap else float("nan"),
        "recall_at_fpr_001": recall_at_fpr(y_all, s_all, 0.01),
        "f1": float(f1_score(y_all, f_all, zero_division=0)),
        "alert_rate": alert_rate(f_all),
        "fpr": fpr_flags(y_all, f_all),
        "savings": float(1 - total_cost / total_loss) if total_loss > 0 else float("nan"),
        "bayes_savings": bayes_savings(y_all, s_all, amt_all, c_fp),
        "bayes_alert_rate": alert_rate(b_flags_all),
        "bayes_savings_at_1pct": bayes_savings(y_all, s_all, amt_all, c_fp, max_alert_rate=0.01),
        "bayes_savings_at_01pct": bayes_savings(y_all, s_all, amt_all, c_fp, max_alert_rate=0.001),
        "ece": ece(y_all, s_all),
        "brier": brier(y_all, s_all),
        "per_silo_ap": per_silo_ap,
    }
    if thr_map is not None:
        out["thresholds"] = {s: float(t) for s, t in thr_map.items()}
    return out


def evaluate_ensemble(clients, payloads, c_fp=7500.0, seed=42, n_draws=300):
    first_val = next(iter(clients.values()))
    ref_clients = first_val if isinstance(first_val, dict) else clients
    val_scores = gather_scores(clients, payloads, "val")
    test_scores = gather_scores(clients, payloads, "test")
    y_vals = {s: c.y["val"] for s, c in ref_clients.items()}
    models = list(payloads.keys())

    # 1. Baseline uncalibrated ensemble weights (source-paper replication)
    weights, val_obj = search_weights(val_scores, y_vals, n_draws, seed)

    # 2. Proposed per-silo calibration (OOF on val) BEFORE weight search so all
    # heterogeneous models sit on a common calibrated probability scale.
    cals, cal_methods, cal_val_oof = fit_calibrators(val_scores, y_vals, seed=seed)
    cal_weights, cal_val_obj = search_weights(cal_val_oof, y_vals, n_draws, seed)

    def combine(scores, w):
        return {s: sum(w[m] * scores[m][s] for m in models) for s in ref_clients}

    soft_val = combine(val_scores, {m: 1 / len(models) for m in models})
    soft_test = combine(test_scores, {m: 1 / len(models) for m in models})
    w_val = combine(val_scores, weights)
    w_test = combine(test_scores, weights)

    cal_val = combine(cal_val_oof, cal_weights)
    cal_test_by_m = {
        m: {s: (cals[m][s].predict(test_scores[m][s])
                if cals[m][s] is not None else test_scores[m][s])
            for s in ref_clients}
        for m in models
    }
    cal_test = combine(cal_test_by_m, cal_weights)

    variants = {}
    # individual global models (global cost threshold from pooled val — non-private baseline)
    for m in models:
        yv = np.concatenate([y_vals[s] for s in ref_clients])
        sv = np.concatenate([val_scores[m][s] for s in ref_clients])
        av = np.concatenate([ref_clients[s].amt["val"] for s in ref_clients])
        thr, _ = cost_threshold(yv, sv, av, c_fp)
        variants[m] = _eval_variant(test_scores[m], ref_clients,
                                    {s: thr for s in ref_clients}, c_fp)

    def global_thr_variant(val_comb, test_comb, name):
        yv = np.concatenate([y_vals[s] for s in ref_clients])
        sv = np.concatenate([val_comb[s] for s in ref_clients])
        av = np.concatenate([ref_clients[s].amt["val"] for s in ref_clients])
        thr, _ = cost_threshold(yv, sv, av, c_fp)
        variants[name] = _eval_variant(test_comb, ref_clients, {s: thr for s in ref_clients}, c_fp)

    global_thr_variant(soft_val, soft_test, "soft_voting")
    global_thr_variant(w_val, w_test, "weighted_voting")

    # PROPOSED (a): per-silo calibration + per-silo cost-optimal flat thresholds (strictly local)
    thr_map = {}
    for s in ref_clients:
        thr_map[s], _ = cost_threshold(y_vals[s], cal_val[s], ref_clients[s].amt["val"], c_fp)
    variants["weighted_calibrated_per_silo"] = _eval_variant(cal_test, ref_clients, thr_map, c_fp)

    # PROPOSED (b): per-silo calibration + Bayes minimum-risk rule (p_i * A_i >= c_fp)
    variants["weighted_calibrated_bayes"] = _eval_variant(
        cal_test, ref_clients, None, c_fp, use_bayes=True
    )

    # PROPOSED (c): per-silo calibration + Bayes minimum-risk under 1% operational alert cap
    variants["weighted_calibrated_bayes_1pct"] = _eval_variant(
        cal_test, ref_clients, None, c_fp, use_bayes=True, max_alert_rate=0.01
    )

    # 2-model calibrated ensemble ablations (when >= 3 models are present)
    ablations_2model = {}
    if len(models) >= 3:
        from itertools import combinations
        for pair in combinations(models, 2):
            sub_oof = {m: cal_val_oof[m] for m in pair}
            sub_w, sub_obj = search_weights(sub_oof, y_vals, n_draws, seed)
            sub_val = {s: sum(sub_w[m] * cal_val_oof[m][s] for m in pair) for s in ref_clients}
            sub_test = {s: sum(sub_w[m] * cal_test_by_m[m][s] for m in pair) for s in ref_clients}
            sub_thr = {
                s: cost_threshold(y_vals[s], sub_val[s], ref_clients[s].amt["val"], c_fp)[0]
                for s in ref_clients
            }
            key = "+".join(pair)
            ablations_2model[key] = {
                "weights": sub_w,
                "val_objective_macro_ap": sub_obj,
                "weighted_calibrated_per_silo": _eval_variant(sub_test, ref_clients, sub_thr, c_fp),
                "weighted_calibrated_bayes_1pct": _eval_variant(
                    sub_test, ref_clients, None, c_fp, use_bayes=True, max_alert_rate=0.01
                ),
            }

    return {"weights": weights, "val_objective_macro_ap": val_obj,
            "calibrated_weights": cal_weights, "calibrated_val_objective_macro_ap": cal_val_obj,
            "variants": variants,
            "ablations_2model": ablations_2model,
            "calibrated_silos": {m: [s for s in cals[m] if cals[m][s] is not None]
                                 for m in models},
            "calibration_methods": cal_methods}
