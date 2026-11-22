"""Feature matrices under a standardized canonical cross-institution schema
(analogous to BI SNAP / ISO 20022 reporting; institutional heterogeneity enters
via rail-mix, feature-distribution, quantity, and label skew)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAIL_NAMES = ["QRIS", "EWALLET", "BIFAST", "VA", "CARD"]
MERCHANT_CATS = list(range(-1, 7))

NUM_COLS = [
    "amount_log", "hour_sin", "hour_cos", "dow", "is_payday", "is_promo", "is_night",
    "device_change", "new_payee", "region_mismatch", "tenure_log",
    "velocity_1h", "velocity_24h", "gap_prev_log", "amount_z",
    "payee_indegree_24h", "merchant_velocity_24h", "merchant_vel_z",
]


_NON_FEATURE_COLS = {
    "txn_id", "ts", "split", "silo_id", "is_fraud", "isFraud", "Class",
    "typology", "campaign_id", "TransactionID", "TransactionDT", "Time",
    "amount", "Amount", "amt", "TransactionAmt",
}


def build_matrix(df: pd.DataFrame, dataset: str = "indofraud"):
    """Return (X float32 ndarray, feature_names). No ids/labels/typology ever enter X."""
    if dataset == "ulb":
        cols = [c for c in df.columns if c.startswith("V")] + ["Amount_log", "hour"]
        X = df[cols].to_numpy(np.float32)
        return X, cols

    if dataset in ("ieee_cis", "sparkov"):
        cols = [
            c for c in df.columns
            if c not in _NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[cols].to_numpy(np.float32)
        return X, cols

    n = len(df)
    names = NUM_COLS + [f"rail_{r}" for r in RAIL_NAMES] + [f"mcat_{c}" for c in MERCHANT_CATS]
    X = np.zeros((n, len(names)), dtype=np.float32)

    # Fill directly into float32 buffer (avoids multi-GB float64 DataFrame/concat intermediates)
    hour = df.hour.to_numpy(np.float64)
    col_values = {
        "amount_log": np.log1p(df.amount.to_numpy(np.float64)).astype(np.float32),
        "hour_sin": np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32),
        "hour_cos": np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32),
        "tenure_log": np.log1p(df.tenure_days.to_numpy(np.float64)).astype(np.float32),
        "gap_prev_log": np.log1p(df.gap_prev_sec.to_numpy(np.float64)).astype(np.float32),
    }
    for j, c in enumerate(NUM_COLS):
        if c in col_values:
            X[:, j] = col_values[c]
        else:
            X[:, j] = df[c].to_numpy(np.float32, copy=False)

    off = len(NUM_COLS)
    rail = df.rail.to_numpy()
    for k in range(len(RAIL_NAMES)):
        X[:, off + k] = (rail == k)

    off += len(RAIL_NAMES)
    mcat = df.merchant_cat.to_numpy()
    for k, cat in enumerate(MERCHANT_CATS):
        X[:, off + k] = (mcat == cat)

    return X, names


class MLPPreprocessor:
    """Standardization + NaN imputation for MLP inputs.

    Supports both:
      - `fit(X)`: local standardization (for standalone/centralized/local-only use)
      - `from_federated_stats(stats_list)`: FL-safe global standardization computed
        solely from per-client sufficient statistics `(count, sum, sum_sq)` so all
        federated clients share identical feature coordinate scales without ever
        sharing raw rows.
    """

    def __init__(self, clip: float = 10.0):
        self.clip = clip
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    @staticmethod
    def local_stats(X: np.ndarray) -> dict:
        X = np.asarray(X)
        n_cols = X.shape[1] if X.ndim == 2 else 0
        cnt = np.zeros(n_cols, dtype=np.int64)
        s = np.zeros(n_cols, dtype=np.float64)
        sq = np.zeros(n_cols, dtype=np.float64)
        for j in range(n_cols):
            col = np.asarray(X[:, j], dtype=np.float64)
            valid = np.isfinite(col)
            cnt[j] = int(valid.sum())
            if cnt[j] > 0:
                vals = col[valid]
                s[j] = float(vals.sum())
                sq[j] = float(np.dot(vals, vals))
        return {"count": cnt, "sum": s, "sum_sq": sq}

    def fit(self, X: np.ndarray) -> "MLPPreprocessor":
        return self.fit_from_stats([self.local_stats(X)])

    def fit_from_stats(self, stats_list: list[dict]) -> "MLPPreprocessor":
        cnt = np.sum([s["count"] for s in stats_list], axis=0).astype(np.float64)
        total = np.sum([s["sum"] for s in stats_list], axis=0).astype(np.float64)
        total_sq = np.sum([s["sum_sq"] for s in stats_list], axis=0).astype(np.float64)
        safe_cnt = np.maximum(cnt, 1.0)
        mean = total / safe_cnt
        var = np.maximum(total_sq / safe_cnt - mean * mean, 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-6, 1.0, std)
        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None, "call fit() first"
        out = np.array(X, dtype=np.float32, copy=True)
        mean32 = self.mean_.astype(np.float32)
        inv_std32 = (1.0 / self.std_).astype(np.float32)
        for j in range(out.shape[1]):
            col = out[:, j]
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = mean32[j]
            col -= mean32[j]
            col *= inv_std32[j]
            if self.clip is not None:
                np.clip(col, -self.clip, self.clip, out=col)
        return out
