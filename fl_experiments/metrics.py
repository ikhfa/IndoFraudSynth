"""Metrics: AUC-PR, recall@FPR, calibration (ECE/Brier), cost model (IDR), latency."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, f1_score, brier_score_loss

# Currency-appropriate default false-positive review/friction costs:
#   indofraud: IDR (~Rp 7,500 automated/manual triage friction)
#   ulb:       EUR (~€5 review/friction cost vs €88 mean ticket)
#   ieee_cis / sparkov: USD (~$5 review/friction cost)
DEFAULT_C_FP: dict[str, float] = {
    "indofraud": 7500.0,
    "ulb": 5.0,
    "ieee_cis": 5.0,
    "sparkov": 5.0,
}


def default_c_fp(dataset: str) -> float:
    return float(DEFAULT_C_FP.get(dataset, 7500.0))


def alert_rate(flagged) -> float:
    arr = np.asarray(flagged, dtype=bool)
    return float(arr.mean()) if len(arr) else float("nan")


def fpr_flags(y, flagged) -> float:
    y = np.asarray(y)
    fl = np.asarray(flagged, dtype=bool)
    neg = int((y == 0).sum())
    if neg == 0:
        return float("nan")
    return float(((y == 0) & fl).sum() / neg)


def ap(y, s) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(average_precision_score(y, s))


def recall_at_fpr(y, s, max_fpr: float = 0.01) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(fpr, max_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


def brier(y, p) -> float:
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return float(brier_score_loss(y, p))


def ece(y, p, n_bins: int = 15) -> float:
    """Expected Calibration Error (equal-width bins in [0, 1])."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    n = len(y)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            err += (cnt / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(err)


def f1_at(y, s, thr: float) -> float:
    return float(f1_score(y, (np.asarray(s) >= thr).astype(int), zero_division=0))


def best_f1_threshold(y, s):
    thrs = np.quantile(s, np.linspace(0.5, 0.999, 120))
    best_t, best_f = 0.5, -1.0
    for t in thrs:
        f = f1_at(y, s, t)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t, best_f


def fraud_cost(y, s, thr, amounts, c_fp: float = 7500.0) -> float:
    """Expected operational cost: missed-fraud losses + false-alarm review/friction."""
    y = np.asarray(y); s = np.asarray(s); amounts = np.asarray(amounts, dtype=float)
    flagged = s >= thr
    fn_loss = amounts[(y == 1) & ~flagged].sum()
    fp_cost = c_fp * int(((y == 0) & flagged).sum())
    return float(fn_loss + fp_cost)


def fraud_cost_flags(y, flagged, amounts, c_fp: float = 7500.0) -> float:
    """Operational cost given boolean/binary alert flags."""
    y = np.asarray(y); flagged = np.asarray(flagged, dtype=bool)
    amounts = np.asarray(amounts, dtype=float)
    fn_loss = amounts[(y == 1) & ~flagged].sum()
    fp_cost = c_fp * int(((y == 0) & flagged).sum())
    return float(fn_loss + fp_cost)


def bayes_cost_predict(p, amounts, c_fp: float = 7500.0, max_alert_rate: float | None = None) -> np.ndarray:
    """Bayes minimum-risk decision rule: flag when expected loss p_i * A_i >= c_fp.

    Optionally enforces an operational alert-rate budget (top-q by expected savings
    p_i * A_i - c_fp among transactions with positive expected value).
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    amounts = np.asarray(amounts, dtype=float)
    exp_net = p * amounts - c_fp
    flagged = exp_net >= 0.0
    if max_alert_rate is not None and len(p) > 0:
        cap = max(int(np.floor(max_alert_rate * len(p))), 1)
        if int(flagged.sum()) > cap:
            order = np.argsort(-exp_net)
            keep = np.zeros(len(p), dtype=bool)
            keep[order[:cap]] = True
            flagged = flagged & keep
    return flagged.astype(np.int8)


def bayes_savings(y, p, amounts, c_fp: float = 7500.0, max_alert_rate: float | None = None) -> float:
    """Fraction of no-model fraud losses avoided under Bayes minimum-risk decisions."""
    y = np.asarray(y)
    amounts = np.asarray(amounts, dtype=float)
    total_loss = float(amounts[y == 1].sum())
    if total_loss <= 0:
        return float("nan")
    flags = bayes_cost_predict(p, amounts, c_fp=c_fp, max_alert_rate=max_alert_rate)
    return float(1.0 - fraud_cost_flags(y, flags, amounts, c_fp) / total_loss)


def cost_threshold(y, s, amounts, c_fp: float = 7500.0):
    """Threshold minimizing cost on validation data."""
    s = np.asarray(s)
    if len(s) == 0:
        return 0.5, 0.0
    thrs = np.unique(np.quantile(s, np.linspace(0.001, 0.999, 250)))
    best_t, best_c = 0.5, float("inf")
    for t in thrs:
        c = fraud_cost(y, s, t, amounts, c_fp)
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t, best_c


def savings(y, s, thr, amounts, c_fp: float = 7500.0) -> float:
    """Fraction of no-model fraud losses avoided (1 - cost_model / cost_nomodel)."""
    total_loss = float(np.asarray(amounts)[np.asarray(y) == 1].sum())
    if total_loss <= 0:
        return float("nan")
    return 1.0 - fraud_cost(y, s, thr, amounts, c_fp) / total_loss


def latency_p95_ms(predict_fn, X, n_iter: int = 150, warmup: int = 5) -> float:
    """Single-transaction inference latency p95 (in-memory model; excludes load time)."""
    if len(X) == 0:
        return float("nan")
    for i in range(min(warmup, len(X))):
        predict_fn(X[i:i + 1])
    times = []
    for i in range(min(n_iter, len(X))):
        row = X[i:i + 1]
        t0 = time.perf_counter()
        predict_fn(row)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95)) if times else float("nan")
