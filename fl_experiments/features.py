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

    d = pd.DataFrame(index=df.index)
    d["amount_log"] = np.log1p(df.amount.to_numpy(np.float64))
    hour = df.hour.to_numpy(np.float64)
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    for c in ["dow", "is_payday", "is_promo", "is_night", "device_change", "new_payee",
              "region_mismatch", "velocity_1h", "velocity_24h", "amount_z",
              "payee_indegree_24h", "merchant_velocity_24h", "merchant_vel_z"]:
        d[c] = df[c].to_numpy(np.float64)
    d["tenure_log"] = np.log1p(df.tenure_days.to_numpy(np.float64))
    d["gap_prev_log"] = np.log1p(df.gap_prev_sec.to_numpy(np.float64))

    rail = pd.Categorical(df.rail.to_numpy(), categories=range(len(RAIL_NAMES)))
    rail_oh = pd.get_dummies(rail, prefix="rail").to_numpy(np.float64)
    mcat = pd.Categorical(df.merchant_cat.to_numpy(), categories=MERCHANT_CATS)
    mcat_oh = pd.get_dummies(mcat, prefix="mcat").to_numpy(np.float64)

    X = np.concatenate([d[NUM_COLS].to_numpy(np.float64), rail_oh, mcat_oh], axis=1)
    names = NUM_COLS + [f"rail_{r}" for r in RAIL_NAMES] + [f"mcat_{c}" for c in MERCHANT_CATS]
    return X.astype(np.float32), names


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
        X = np.asarray(X, dtype=np.float64)
        valid = np.isfinite(X)
        cnt = valid.sum(axis=0).astype(np.int64)
        clean = np.where(valid, X, 0.0)
        return {"count": cnt, "sum": clean.sum(axis=0), "sum_sq": (clean * clean).sum(axis=0)}

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
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isfinite(X), X, self.mean_)
        z = (filled - self.mean_) / self.std_
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)
