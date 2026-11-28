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
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .metrics import (
    ap, recall_at_fpr, f1_at, cost_threshold, fraud_cost, fraud_cost_flags,
    bayes_cost_predict, bayes_savings, ece, brier, alert_rate, fpr_flags,
)

MODEL_TYPES = ["xgboost", "catboost", "mlp"]


class PlattCalibrator:
    """Monotonic 2-parameter Platt / logit-temperature calibrator (low-variance for small silos).

    Enforces strictly positive slope `a >= 1e-3` via L-BFGS-B so calibration can
    never invert score rankings on small silos, and clips at `1e-15` (float64 safe)
    so severely under-scaled unnormalized GBDT probabilities retain their ranking.
    """

    def __init__(self, reg: float = 0.1, slope_bounds: tuple[float, float] = (1e-3, 50.0)):
        self.reg = float(reg)
        self.slope_bounds = (float(slope_bounds[0]), float(slope_bounds[1]))
        self.slope_: float = 1.0
        self.intercept_: float = 0.0

    @classmethod
    def from_params(cls, slope: float, intercept: float, reg: float = 0.1) -> "PlattCalibrator":
        obj = cls(reg=reg)
        obj.slope_ = max(float(slope), 1e-3)
        obj.intercept_ = float(intercept)
        return obj

    @staticmethod
    def _to_logit(s: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(s, dtype=np.float64), 1e-15, 1.0 - 1e-15)
        return np.log(p / (1.0 - p))

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        z = self._to_logit(scores)
        yf = np.asarray(y, dtype=np.float64)
        reg = self.reg

        def _obj_and_grad(params):
            a, b = float(params[0]), float(params[1])
            m = a * z + b
            # Overflow-safe binary cross-entropy + mild L2 penalty toward identity slope a=1
            loss = float(np.sum(np.logaddexp(0.0, m) - yf * m) + 0.5 * reg * ((a - 1.0) ** 2))
            diff = expit(m) - yf
            grad = np.array([
                float(np.dot(diff, z) + reg * (a - 1.0)),
                float(np.sum(diff)),
            ], dtype=np.float64)
            return loss, grad

        res = minimize(
            _obj_and_grad,
            x0=np.array([1.0, 0.0], dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[self.slope_bounds, (-50.0, 50.0)],
            options={"maxiter": 200},
        )
        self.slope_ = float(res.x[0])
        self.intercept_ = float(res.x[1])
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        z = self._to_logit(scores)
        return expit(self.slope_ * z + self.intercept_)


def _make_calibrator(y: np.ndarray, isotonic_min_pos: int = 500):
    """Choose smooth, rank-preserving Platt scaling for 5..(isotonic_min_pos-1) positives,
    and Isotonic regression only when positives >= isotonic_min_pos (avoids coarse
    piecewise-constant step ties degrading AUC-PR under severe fraud imbalance)."""
    pos = int(np.asarray(y).sum())
    neg = int(len(y) - pos)
    if pos >= isotonic_min_pos and neg >= isotonic_min_pos:
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0), "isotonic"
    if pos >= 5 and neg >= 5:
        return PlattCalibrator(), "platt"
    return None, "identity"


def _build_calibrator(kind: str):
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    if kind == "platt":
        return PlattCalibrator()
    return None


def safe_cal_predict(cal, scores) -> np.ndarray:
    """Predict calibrated probabilities safely, handling `cal=None` and 0-row silos."""
    arr = np.asarray(scores, dtype=np.float64)
    if cal is None or len(arr) == 0:
        return arr
    return np.asarray(cal.predict(arr), dtype=np.float64)


def _prior_logit(pos: float, total: float) -> float:
    p = float(np.clip(float(pos) / max(float(total), 1.0), 1e-6, 1.0 - 1e-6))
    return float(np.log(p / (1.0 - p)))


def _fisher_weight(pos: int, total: int) -> float:
    """Effective binomial Fisher weight `pos * neg / (pos + neg)`.

    Matches `pos` when `neg >> pos` (typical rare-event silos), while preventing
    degenerate extreme-Dirichlet silos (e.g., 1,000 pos / 5 neg) from dominating
    the federated global Platt average.
    """
    neg = max(int(total) - int(pos), 0)
    tot = int(pos) + neg
    if pos <= 0 or neg <= 0 or tot <= 0:
        return 0.0
    return float(pos * neg) / float(tot)


def _agg_platt_params(platt_params: dict[str, tuple[float, float, int, int]],
                      exclude: str | None = None,
                      silo_counts: dict[str, tuple[int, int]] | None = None,
                      prior_shift: bool = True) -> tuple[float, float, float] | None:
    """FL-safe Fisher-weighted, Bayes-prior-adjusted average of local Platt scalars.

    Each silo shares only `(slope, intercept, pos_count, total_count)`.
    When `prior_shift=True`, local intercepts are neutralized by subtracting each
    silo's log-odds prior `logit(pi_k)` before averaging and then re-anchored to
    the federated global prior `logit(pi_global)` — preventing an extreme-Dirichlet
    positive-heavy silo from shifting zero-fraud silos toward p -> 1.0.

    Returns `(a_global, b_global_at_global_prior, b_neutral)` or `None`.
    """
    items = []
    for s, (a, b, pos, tot) in platt_params.items():
        if s == exclude:
            continue
        w = _fisher_weight(pos, tot)
        if w > 0.0:
            items.append((a, b, pos, tot, w))
    if not items:
        return None
    tot_w = float(sum(w for _, _, _, _, w in items))
    a_g = sum(a * w for a, _, _, _, w in items) / tot_w
    if not prior_shift:
        b_g = sum(b * w for _, b, _, _, w in items) / tot_w
        return float(a_g), float(b_g), float(b_g)

    b_neutral_g = sum((b - _prior_logit(pos, tot)) * w for _, b, pos, tot, w in items) / tot_w
    if silo_counts:
        g_pos = sum(p for s, (p, _) in silo_counts.items() if s != exclude)
        g_tot = sum(n for s, (_, n) in silo_counts.items() if s != exclude)
        if g_tot <= 0:
            g_pos = sum(p for p, _ in silo_counts.values())
            g_tot = sum(n for _, n in silo_counts.values())
    else:
        g_pos = sum(pos for _, _, pos, _, _ in items)
        g_tot = sum(tot for _, _, _, tot, _ in items)
    b_global = b_neutral_g + _prior_logit(g_pos, g_tot)
    return float(a_g), float(b_global), float(b_neutral_g)


def _shrink_platt(cal: PlattCalibrator, global_info: tuple[float, float, float] | None,
                  pos: int, total: int, prior_strength: float = 15.0) -> PlattCalibrator:
    """Empirical-Bayes shrinkage of a local Platt fit toward federated global `(a, b_target)`.

    Uses effective Fisher count `eff_n = pos * neg / (pos + neg)` for shrinkage weight
    `gamma = eff_n / (eff_n + prior_strength)` and shifts the global prior-neutral
    intercept back to the silo's own local prior `logit(pi_k)`.
    """
    eff_n = _fisher_weight(pos, total)
    if global_info is None or eff_n >= 500.0:
        return cal
    gamma = float(eff_n) / (float(eff_n) + float(prior_strength))
    a_g, _, b_neutral_g = global_info
    b_target = b_neutral_g + _prior_logit(pos, total)
    return PlattCalibrator.from_params(
        gamma * cal.slope_ + (1.0 - gamma) * a_g,
        gamma * cal.intercept_ + (1.0 - gamma) * b_target,
        reg=cal.reg,
    )


def gather_scores(clients, payloads, split, loaded_models: dict | None = None):
    """{model: {silo: scores}} for clients that have a trained payload."""
    from . import models as M
    first_val = next(iter(clients.values()))
    is_nested = isinstance(first_val, dict)
    out = {}
    for mt, payload in payloads.items():
        if payload is None:
            continue
        cmap = clients[mt] if is_nested else clients
        loaded = None
        if loaded_models is not None and mt in loaded_models:
            loaded = loaded_models[mt]
        elif mt == "xgboost":
            loaded = M.bytes_to_xgb(payload)
        elif mt == "catboost":
            loaded = M.bytes_to_cat(payload)
        out[mt] = {name: c.predict(payload, split, loaded_model=loaded)
                   for name, c in cmap.items()}
    return out


def _macro_ap(scores_by_silo, y_by_silo):
    vals = [ap(y_by_silo[s], scores_by_silo[s]) for s in scores_by_silo
            if 0 < y_by_silo[s].sum() < len(y_by_silo[s])]
    return float(np.mean(vals)) if vals else float("nan")


def _to_logit(scores: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=np.float64), 1e-15, 1.0 - 1e-15)
    return np.log(p / (1.0 - p))


def combine_scores_logit(scores_by_m: dict[str, dict[str, np.ndarray]],
                         weights: dict[str, float],
                         silos: list[str] | None = None) -> dict[str, np.ndarray]:
    """Calibrated log-odds (log-linear opinion pool) ensemble blend:
    `p_ens = sigmoid(sum_m w_m * logit(p_m))`.

    Under severe fraud imbalance (p ~ 1e-3), linear probability averaging acts
    like a soft-OR that ignores strong negative evidence from complementary
    models and amplifies Platt tail-softening distortion. Blending calibrated
    log-odds is multiplicative in odds and preserves rare-event ranking.
    """
    models = list(weights.keys())
    if silos is None:
        silos = list(scores_by_m[models[0]].keys())
    return {
        s: expit(sum(weights[m] * _to_logit(scores_by_m[m][s]) for m in models))
        for s in silos
    }


def fit_global_post_calibrator(val_scores_by_silo: dict[str, np.ndarray],
                               y_vals: dict[str, np.ndarray],
                               seed: int = 42):
    """FL-safe 2-parameter post-blend global Platt recalibrator.

    Blending correlated calibrated logits can slightly under-disperse output
    probabilities; fitting a single global monotonic Platt map `(a_ens, b_ens)`
    across silos (2 floats + 1 int shared/silo) restores sharp probability
    calibration (ECE / Bayes@1%) while preserving 100% of pooled & macro AP
    rankings. Returns `(calibrator_or_None, oof_val_scores_by_silo)`.
    """
    silos = list(val_scores_by_silo.keys())
    local_platt: dict[str, tuple[float, float, int, int]] = {}
    for s in silos:
        y = np.asarray(y_vals[s], dtype=int)
        sc = np.asarray(val_scores_by_silo[s], dtype=np.float64)
        pos = int(y.sum())
        tot = int(len(y))
        neg = tot - pos
        if pos >= 5 and neg >= 5:
            # Inputs are already per-silo calibrated; use regularized bounded slope
            # so post-blend recalibration adjusts dispersion without blowing up under skew.
            cal = PlattCalibrator(reg=1.0, slope_bounds=(0.25, 4.0)).fit(sc, y)
            local_platt[s] = (cal.slope_, cal.intercept_, pos, tot)

    global_ab = _agg_platt_params(local_platt, exclude=None, prior_shift=False)
    if global_ab is None:
        return None, {s: np.asarray(val_scores_by_silo[s], dtype=np.float64).copy() for s in silos}

    post_cal = PlattCalibrator.from_params(global_ab[0], global_ab[1], reg=1.0)
    oof: dict[str, np.ndarray] = {}
    for s in silos:
        sc = np.asarray(val_scores_by_silo[s], dtype=np.float64)
        y = np.asarray(y_vals[s], dtype=int)
        if len(silos) >= 2:
            other_ab = _agg_platt_params(local_platt, exclude=s, prior_shift=False)
            if other_ab is not None:
                oof[s] = PlattCalibrator.from_params(other_ab[0], other_ab[1], reg=1.0).predict(sc)
            else:
                oof[s] = post_cal.predict(sc)
        else:
            pos = int(y.sum())
            neg = int(len(y) - pos)
            n_splits = min(3, pos, neg)
            if n_splits >= 2:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                pred = np.empty_like(sc, dtype=np.float64)
                for tr, va in skf.split(sc.reshape(-1, 1), y):
                    pos_tr = int(y[tr].sum())
                    neg_tr = int(len(y[tr]) - pos_tr)
                    if pos_tr >= 2 and neg_tr >= 2:
                        pred[va] = PlattCalibrator(reg=1.0, slope_bounds=(0.25, 4.0)).fit(sc[tr], y[tr]).predict(sc[va])
                    else:
                        pred[va] = sc[va]
                oof[s] = pred
            else:
                oof[s] = post_cal.predict(sc)
    return post_cal, oof


def search_weights(val_scores, y_vals, n_draws=300, seed=42, logit_space: bool = False):
    models = list(val_scores.keys())
    silos = list(y_vals.keys())
    rng = np.random.default_rng(seed)

    if logit_space:
        score_src = {m: {s: _to_logit(val_scores[m][s]) for s in silos} for m in models}
    else:
        score_src = val_scores

    def objective(w):
        per_silo = {}
        for s in silos:
            per_silo[s] = sum(w[i] * score_src[m][s] for i, m in enumerate(models))
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


def fit_calibrators(val_scores, y_vals, seed: int = 42, prior_strength: float = 100.0):
    """Per-silo calibration with FL-safe hierarchical Platt shrinkage & global fallback.

    - Silos with >=500 positives use Isotonic regression (locking the same family in OOF).
    - Silos with 5..499 positives use smooth, strictly monotonic Platt scaling (avoiding
      piecewise-constant Isotonic step ties under rare-event imbalance); when multiple
      silos exist, local `(a_k, b_k)` parameters are shrunk toward the FL-safe global
      Platt average `(a_global, b_global)` (only 2 scalars + positive count shared/silo).
    - Silos with <5 positives fall back to the federated global Platt calibrator
      (`platt_global`) instead of raw uncalibrated scores, preventing cross-silo
      score-scale mismatches in pooled evaluation and Bayes decisions.

    Returns `(cals, methods, oof_scores)` where `oof_scores` are strictly out-of-fold
    (and leave-one-silo-out for the global shrinkage target) to avoid overfitting `val`.
    """
    cals, methods, oof = {}, {}, {}
    for m, by_silo in val_scores.items():
        cals[m] = {}; methods[m] = {}; oof[m] = {}
        # Pass 1: fit local Platt scalars on every silo with >=5 pos and >=5 neg to build
        # the FL-safe Bayes-prior-adjusted global fallback/shrinkage target
        # (only 2 floats + 2 scalar counts per silo).
        local_platt: dict[str, tuple[float, float, int, int]] = {}
        silo_counts: dict[str, tuple[int, int]] = {}
        raw_local: dict[str, tuple[object | None, str, np.ndarray, np.ndarray, int, int]] = {}
        for s, scores in by_silo.items():
            y = np.asarray(y_vals[s], dtype=int)
            sc = np.asarray(scores, dtype=np.float64)
            pos = int(y.sum())
            tot = int(len(y))
            silo_counts[s] = (pos, tot)
            cal, kind = _make_calibrator(y)
            if cal is not None:
                cal.fit(sc, y)
                if kind == "platt":
                    local_platt[s] = (cal.slope_, cal.intercept_, pos, tot)
                else:
                    # Also fit a lightweight 2-param Platt strictly for global fallback sharing
                    p_share = PlattCalibrator().fit(sc, y)
                    local_platt[s] = (p_share.slope_, p_share.intercept_, pos, tot)
            raw_local[s] = (cal, kind, sc, y, pos, tot)

        global_info = (
            _agg_platt_params(local_platt, exclude=None, silo_counts=silo_counts, prior_shift=True)
            if len(by_silo) >= 2 else None
        )

        # Pass 2: assemble final per-silo calibrators and strictly leak-free OOF predictions
        for s, (cal, kind, sc, y, pos, tot) in raw_local.items():
            other_info = (
                _agg_platt_params(local_platt, exclude=s, silo_counts=silo_counts, prior_shift=True)
                if len(by_silo) >= 2 else None
            )

            if cal is None:
                # Low-positive or single-class silo (<5 pos or <5 neg): use FL-safe global Platt
                # anchored at the federated global prior pi_global.
                if global_info is not None:
                    gcal = PlattCalibrator.from_params(global_info[0], global_info[1])
                    cals[m][s] = gcal
                    methods[m][s] = "platt_global"
                    oof[m][s] = gcal.predict(sc)
                else:
                    cals[m][s] = None
                    methods[m][s] = "identity"
                    oof[m][s] = sc.copy()
                continue

            if kind == "platt" and other_info is not None:
                final_cal = _shrink_platt(cal, other_info, pos, tot, prior_strength=prior_strength)
                cals[m][s] = final_cal
                methods[m][s] = "platt_shrunk"
            else:
                final_cal = cal
                cals[m][s] = final_cal
                methods[m][s] = kind

            n_splits = min(3, pos, int(tot - pos))
            if n_splits >= 2:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                pred = np.empty_like(sc, dtype=np.float64)
                for tr, va in skf.split(sc.reshape(-1, 1), y):
                    pos_tr = int(y[tr].sum())
                    tot_tr = int(len(y[tr]))
                    neg_tr = tot_tr - pos_tr
                    # Lock inner OOF calibrator family to outer `kind` (prevents silent
                    # identity fallback when pos in {5,6,7} or family flip at pos in {50..74})
                    if pos_tr >= 2 and neg_tr >= 2:
                        fcal = _build_calibrator(kind)
                        fcal.fit(sc[tr], y[tr])
                        if kind == "platt" and other_info is not None:
                            fcal = _shrink_platt(fcal, other_info, pos_tr, tot_tr,
                                                 prior_strength=prior_strength)
                        pred[va] = fcal.predict(sc[va])
                    elif other_info is not None:
                        pred[va] = PlattCalibrator.from_params(other_info[0], other_info[1]).predict(sc[va])
                    else:
                        pred[va] = sc[va]
                oof[m][s] = pred
            else:
                oof[m][s] = final_cal.predict(sc)
    return cals, methods, oof


def _eval_variant(test_scores, clients, thr_map, c_fp, use_bayes=False,
                  max_alert_rate: float | None = None):
    """Evaluate a test-score dict across silos.

    If `use_bayes=True`, applies Bayes minimum-risk decisions `p_i * A_i >= c_fp`
    directly on calibrated probabilities instead of a flat threshold (optionally
    subject to a per-silo `max_alert_rate` operational budget).

    Metric semantics:
      - `savings` reflects this variant's actual deployed decision rule (`thr_map`
        or per-silo Bayes rule with `max_alert_rate`, matching independent per-institution
        SOC alert budgets).
      - `per_silo_bayes_savings_at_1pct` always reports per-silo 1%-capped Bayes savings.
      - `bayes_savings_at_1pct` reports pooled 1%-capped Bayes savings across all silos
        (useful under extreme Dirichlet alpha=0.1 where a single silo may have >80% fraud
        and a 1% per-silo cap mechanically truncates true positives).
    """
    ys, ss, amts, flags = [], [], [], []
    per_silo_ap = {}
    total_cost, total_loss = 0.0, 0.0
    per_silo_b1_cost = 0.0
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
        fl_b1 = bayes_cost_predict(scores, amt, c_fp=c_fp, max_alert_rate=0.01)
        per_silo_b1_cost += fraud_cost_flags(y, fl_b1, amt, c_fp)
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
        "per_silo_bayes_savings_at_1pct": float(1 - per_silo_b1_cost / total_loss) if total_loss > 0 else float("nan"),
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


def evaluate_ensemble(clients, payloads, c_fp=7500.0, seed=42, n_draws=300,
                      out_plot_scores: dict | None = None):
    from . import models as M
    first_val = next(iter(clients.values()))
    ref_clients = first_val if isinstance(first_val, dict) else clients
    loaded_models = {}
    for mt, payload in payloads.items():
        if payload is None:
            continue
        if mt == "xgboost":
            loaded_models[mt] = M.bytes_to_xgb(payload)
        elif mt == "catboost":
            loaded_models[mt] = M.bytes_to_cat(payload)
    val_scores = gather_scores(clients, payloads, "val", loaded_models=loaded_models)
    test_scores = gather_scores(clients, payloads, "test", loaded_models=loaded_models)
    y_vals = {s: c.y["val"] for s, c in ref_clients.items()}
    models = list(payloads.keys())

    # 1. Baseline uncalibrated ensemble weights (source-paper replication: linear prob space)
    weights, val_obj = search_weights(val_scores, y_vals, n_draws, seed, logit_space=False)

    # 2. Proposed per-silo calibration (OOF on val) BEFORE logit-space weight search
    # so all heterogeneous models are combined on a calibrated log-odds scale, followed
    # by a rank-preserving 2-parameter FL-safe global Platt recalibrator.
    cals, cal_methods, cal_val_oof = fit_calibrators(val_scores, y_vals, seed=seed)
    cal_weights, cal_val_obj = search_weights(cal_val_oof, y_vals, n_draws, seed, logit_space=True)

    silo_list = list(ref_clients.keys())

    def combine(scores, w):
        return {s: sum(w[m] * scores[m][s] for m in models) for s in silo_list}

    soft_val = combine(val_scores, {m: 1 / len(models) for m in models})
    soft_test = combine(test_scores, {m: 1 / len(models) for m in models})
    w_val = combine(val_scores, weights)
    w_test = combine(test_scores, weights)

    cal_val_raw = combine_scores_logit(cal_val_oof, cal_weights, silos=silo_list)
    cal_test_by_m = {
        m: {s: safe_cal_predict(cals[m][s], test_scores[m][s]) for s in silo_list}
        for m in models
    }
    cal_test_raw = combine_scores_logit(cal_test_by_m, cal_weights, silos=silo_list)
    post_cal, cal_val = fit_global_post_calibrator(cal_val_raw, y_vals, seed=seed)
    cal_test = {s: safe_cal_predict(post_cal, cal_test_raw[s]) for s in silo_list}
    if out_plot_scores is not None:
        out_plot_scores["weighted"] = w_test
        out_plot_scores["calibrated"] = cal_test

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

    # PROPOSED (a): per-silo calibration + per-silo cost-optimal flat thresholds (strictly local;
    # zero-positive val silos fall back to the FL-safe median threshold of positive silos)
    def _per_silo_thresholds(val_comb):
        raw_t = {}
        pos_thrs = []
        for s in ref_clients:
            t, _ = cost_threshold(y_vals[s], val_comb[s], ref_clients[s].amt["val"], c_fp)
            raw_t[s] = t
            if int(np.asarray(y_vals[s]).sum()) > 0:
                pos_thrs.append(t)
        med_t = float(np.median(pos_thrs)) if pos_thrs else 0.5
        return {s: (raw_t[s] if int(np.asarray(y_vals[s]).sum()) > 0 else med_t)
                for s in ref_clients}

    thr_map = _per_silo_thresholds(cal_val)
    variants["weighted_calibrated_per_silo"] = _eval_variant(cal_test, ref_clients, thr_map, c_fp)

    # PROPOSED (b): per-silo calibration + Bayes minimum-risk rule (p_i * A_i >= c_fp)
    variants["weighted_calibrated_bayes"] = _eval_variant(
        cal_test, ref_clients, None, c_fp, use_bayes=True
    )

    # PROPOSED (c): per-silo calibration + Bayes minimum-risk under 1% operational alert cap
    variants["weighted_calibrated_bayes_1pct"] = _eval_variant(
        cal_test, ref_clients, None, c_fp, use_bayes=True, max_alert_rate=0.01
    )

    # 2-model calibrated logit-space ensemble ablations (when >= 3 models are present)
    ablations_2model = {}
    if len(models) >= 3:
        from itertools import combinations
        for pair in combinations(models, 2):
            sub_oof = {m: cal_val_oof[m] for m in pair}
            sub_w, sub_obj = search_weights(sub_oof, y_vals, n_draws, seed, logit_space=True)
            sub_val_raw = combine_scores_logit(sub_oof, sub_w, silos=silo_list)
            sub_test_raw = combine_scores_logit(
                {m: cal_test_by_m[m] for m in pair}, sub_w, silos=silo_list
            )
            sub_post, sub_val = fit_global_post_calibrator(sub_val_raw, y_vals, seed=seed)
            sub_test = {s: safe_cal_predict(sub_post, sub_test_raw[s]) for s in silo_list}
            sub_thr = _per_silo_thresholds(sub_val)
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
