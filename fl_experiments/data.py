"""Dataset loading, temporal splits, client partitioning, train subsampling."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from indofraud_synth.silos import partition


def load_indofraud(data_dir: str) -> pd.DataFrame:
    tx = pd.read_csv(os.path.join(data_dir, "transactions.csv.gz"))
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tx.attrs["rail_names"] = meta["rail_names"]
    tx.attrs["silo_names"] = meta["silo_names"]
    return tx


def load_ulb(csv_path: str) -> pd.DataFrame:
    """European cardholder dataset (Kaggle mlg-ulb/creditcardfraud).

    Download manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    and pass the path to creditcard.csv.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Class": "is_fraud"})
    df["is_fraud"] = df["is_fraud"].astype(np.int8)
    df["ts"] = df["Time"].astype(np.int64)
    df["amount"] = df["Amount"].astype(np.float64)
    df["Amount_log"] = np.log1p(np.maximum(df["amount"], 0.0))
    df["hour"] = ((df["ts"] // 3600) % 24).astype(np.int8)
    df["silo_id"] = 0
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.attrs["silo_names"] = ["pooled"]
    return df


def _encode_object_cols(df: pd.DataFrame, cols: list[str],
                        fit_prefix_frac: float = 0.7) -> pd.DataFrame:
    """Deterministic integer encoding for categorical/string columns (-1 for missing
    or unseen test categories).

    Vocabularies are learned strictly from the chronological training prefix
    (`fit_prefix_frac`, assuming `df` is already sorted by `ts`) so future test
    categories never leak into encoding tables.
    """
    n_fit = max(int(round(len(df) * fit_prefix_frac)), 1)
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        fit_vals = s.iloc[:n_fit].dropna().astype(str).unique().tolist()
        uniq = sorted(fit_vals)
        lut = {v: i for i, v in enumerate(uniq)}
        df[c] = s.map(lambda x: lut.get(str(x), -1) if pd.notna(x) else -1).astype(np.int32)
    return df


def load_ieee_cis(path: str) -> pd.DataFrame:
    """IEEE-CIS Fraud Detection (Kaggle ieee-fraud-detection).

    Accepts either:
      - a directory containing `train_transaction.csv` (and optional `train_identity.csv`), or
      - the direct path to `train_transaction.csv` (auto-merges sibling `train_identity.csv`).

    Always paired with `temporal_split` (TransactionDT -> ts) to avoid the source
    paper's random-split temporal leakage.
    """
    if os.path.isdir(path):
        tx_path = os.path.join(path, "train_transaction.csv")
        id_path = os.path.join(path, "train_identity.csv")
    else:
        tx_path = path
        id_path = os.path.join(os.path.dirname(path), "train_identity.csv")

    if not os.path.exists(tx_path):
        raise FileNotFoundError(
            f"IEEE-CIS transaction file not found at {tx_path}. Download "
            "train_transaction.csv (+ train_identity.csv) from "
            "https://www.kaggle.com/c/ieee-fraud-detection"
        )

    df = pd.read_csv(tx_path)
    if os.path.exists(id_path):
        ident = pd.read_csv(id_path)
        df = df.merge(ident, on="TransactionID", how="left")

    df["is_fraud"] = df["isFraud"].astype(np.int8)
    df["ts"] = df["TransactionDT"].astype(np.int64)
    df["amount"] = df["TransactionAmt"].astype(np.float64)
    df["amount_log"] = np.log1p(np.maximum(df["amount"], 0.0))
    df["hour"] = ((df["ts"] // 3600) % 24).astype(np.int8)
    df["dow"] = ((df["ts"] // 86400) % 7).astype(np.int8)
    df["silo_id"] = 0

    drop_cols = [c for c in ("TransactionID", "isFraud", "TransactionDT") if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df = _encode_object_cols(df, obj_cols, fit_prefix_frac=0.7)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.attrs["silo_names"] = ["pooled"]
    return df


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load_sparkov(path: str) -> pd.DataFrame:
    """Sparkov credit-card fraud simulation (Kaggle kartik2112/fraud-detection).

    Accepts either:
      - a directory containing `fraudTrain.csv` (and optional `fraudTest.csv`,
        concatenated in chronological order), or
      - the direct path to a Sparkov CSV file.
    """
    if os.path.isdir(path):
        tr_path = os.path.join(path, "fraudTrain.csv")
        te_path = os.path.join(path, "fraudTest.csv")
        if not os.path.exists(tr_path):
            raise FileNotFoundError(
                f"Sparkov file not found at {tr_path}. Download fraudTrain.csv / "
                "fraudTest.csv from https://www.kaggle.com/datasets/kartik2112/fraud-detection"
            )
        frames = [pd.read_csv(tr_path)]
        if os.path.exists(te_path):
            frames.append(pd.read_csv(te_path))
        df = pd.concat(frames, ignore_index=True)
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sparkov CSV not found at {path}")
        df = pd.read_csv(path)

    df["is_fraud"] = df["is_fraud"].astype(np.int8)
    if "unix_time" in df.columns:
        df["ts"] = df["unix_time"].astype(np.int64)
        dt = pd.to_datetime(df["ts"], unit="s")
    else:
        dt = pd.to_datetime(df["trans_date_trans_time"])
        df["ts"] = (dt.astype("int64") // 10**9).astype(np.int64)

    df["amount"] = df["amt"].astype(np.float64)
    df["amount_log"] = np.log1p(np.maximum(df["amount"], 0.0))
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["dow"] = dt.dt.day_of_week.astype(np.int8)
    df["is_night"] = (df["hour"] < 6).astype(np.int8)
    if "city_pop" in df.columns:
        df["city_pop_log"] = np.log1p(np.maximum(df["city_pop"].astype(np.float64), 0.0))
    if "dob" in df.columns:
        dob = pd.to_datetime(df["dob"], errors="coerce")
        df["age_years"] = ((dt - dob).dt.days / 365.25).astype(np.float64)
    if {"lat", "long", "merch_lat", "merch_long"} <= set(df.columns):
        df["geo_dist_km"] = _haversine_km(
            df["lat"].to_numpy(np.float64), df["long"].to_numpy(np.float64),
            df["merch_lat"].to_numpy(np.float64), df["merch_long"].to_numpy(np.float64),
        )

    df["silo_id"] = 0
    # Drop raw PII / high-cardinality identifiers that must never enter features.
    pii_drop = [
        "Unnamed: 0", "trans_date_trans_time", "cc_num", "merchant",
        "first", "last", "street", "city", "trans_num", "dob", "unix_time",
    ]
    df = df.drop(columns=[c for c in pii_drop if c in df.columns])

    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    obj_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    df = _encode_object_cols(df, obj_cols, fit_prefix_frac=0.7)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.attrs["silo_names"] = ["pooled"]
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    t1 = df.ts.quantile(train_frac)
    t2 = df.ts.quantile(train_frac + val_frac)
    split = np.where(df.ts <= t1, "train", np.where(df.ts <= t2, "val", "test"))
    df = df.copy()
    df["split"] = split
    return df, {"mode": "temporal", "t1": float(t1), "t2": float(t2)}


def random_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15,
                 seed: int = 42):
    """Stratified random train/val/test split (for source-paper replication only;
    temporal_split is always primary to prevent future leakage)."""
    rng = np.random.default_rng(seed)
    y = df.is_fraud.to_numpy()
    split = np.full(len(df), "test", dtype=object)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls).copy()
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * train_frac))
        n_va = int(round(len(idx) * val_frac))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
    df = df.copy()
    df["split"] = split
    return df, {"mode": "random", "seed": int(seed),
                "train_frac": float(train_frac), "val_frac": float(val_frac)}


def subsample_train(train: pd.DataFrame, cap: int, seed: int = 42) -> pd.DataFrame:
    """Stratified training cap: preserves the true fraud prior (avoids prior distortion)."""
    if len(train) <= cap:
        return train
    rng = np.random.default_rng(seed)
    fraud = train[train.is_fraud == 1]
    legit = train[train.is_fraud == 0]
    frac = cap / len(train)
    n_fraud = min(len(fraud), max(int(round(len(fraud) * frac)), 1 if len(fraud) > 0 else 0))
    n_legit = min(len(legit), max(cap - n_fraud, 0))
    f_idx = rng.choice(len(fraud), n_fraud, replace=False) if n_fraud > 0 else np.array([], dtype=int)
    l_idx = rng.choice(len(legit), n_legit, replace=False) if n_legit > 0 else np.array([], dtype=int)
    out = pd.concat([fraud.iloc[np.sort(f_idx)], legit.iloc[np.sort(l_idx)]])
    return out.sort_values("ts").reset_index(drop=True)


def make_clients(df: pd.DataFrame, mode: str, k: int, alpha: float = 0.5,
                 seed: int = 42, train_cap: int = 200_000):
    """Return (clients dict, pooled dict).

    `pooled['train']` is the exact union of all clients' training splits so the
    centralized baseline sees identical rows and class priors as the federation.
    """
    parts = partition(df, mode, k, alpha, seed)
    clients = {}
    for i, (name, part) in enumerate(parts.items()):
        tr = subsample_train(part[part.split == "train"], train_cap, seed + i)
        clients[name] = {
            "train": tr,
            "val": part[part.split == "val"],
            "test": part[part.split == "test"],
        }
    pooled_train = (
        pd.concat([c["train"] for c in clients.values()], ignore_index=True)
        .sort_values("ts", kind="stable")
        .reset_index(drop=True)
    )
    pooled = {
        "train": pooled_train,
        "val": df[df.split == "val"],
        "test": df[df.split == "test"],
    }
    return clients, pooled
