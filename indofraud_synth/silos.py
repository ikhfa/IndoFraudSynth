"""Silo assignment (institution profiles) and alternative partitions.

Silos are the servicing institutions:
  BIFAST/VA/CARD -> customer's primary bank
  EWALLET        -> customer's e-wallet provider (bank fallback if no wallet)
  QRIS           -> merchant's acquirer (bank or wallet)

This produces realistic quantity skew (big banks >> BPDs), rail-mix skew (wallet
silos see wallet/QRIS traffic; bank silos see transfers/VA/cards) and label skew.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BANK_WEIGHTS, WALLET_WEIGHTS


def assign_silos(df: pd.DataFrame, agents: pd.DataFrame, merchants: pd.DataFrame,
                 meta: dict, rail_names: list) -> pd.DataFrame:
    banks = meta["banks"]; wallets = meta["wallets"]
    silo_names = banks + [w for w in wallets if w != "none"]
    silo_idx = {s: i for i, s in enumerate(silo_names)}
    ridx = {r: i for i, r in enumerate(rail_names)}

    cust_bank = agents.bank.to_numpy()
    cust_wallet = agents.wallet.to_numpy()          # index into wallets incl. 'none'
    n_banks = len(banks)
    none_idx = wallets.index("none")

    silo = np.zeros(len(df), dtype=np.int32)
    rail = df.rail.to_numpy()
    cid = df.customer_id.to_numpy()

    bank_rails = np.isin(rail, [ridx[r] for r in ("BIFAST", "VA", "CARD") if r in ridx])
    silo[bank_rails] = cust_bank[cid[bank_rails]]

    if "EWALLET" in ridx:
        w_mask = rail == ridx["EWALLET"]
        w = cust_wallet[cid[w_mask]]
        has_wallet = w != none_idx
        sw = np.where(has_wallet, n_banks + w, cust_bank[cid[w_mask]])
        # wallet index maps directly: wallets list order = ewallet_1..3, none last
        silo[w_mask] = sw

    if "QRIS" in ridx:
        q_mask = rail == ridx["QRIS"]
        mids = df.merchant_id.to_numpy()[q_mask]
        # acquirer already encodes banks [0..n_banks-1] and wallets [n_banks..]
        silo[q_mask] = merchants.acquirer.to_numpy()[mids]

    df = df.copy()
    df["silo_id"] = silo
    df.attrs["silo_names"] = silo_names
    return df


def profile_partition(df: pd.DataFrame, k: int, seed: int = 42) -> dict:
    """Partition by institution.

    - If k < n_raw_silos: keep top k-1 institutions and merge the rest into 'silo_other'.
    - If k == n_raw_silos: keep all institutions under their own names.
    - If k > n_raw_silos (e.g. K=20): deterministically sub-partition the largest
      institutions by customer_id (keeping each customer within one branch silo)
      until exactly k silos are formed.
    """
    names = df.attrs.get("silo_names", [str(i) for i in sorted(df.silo_id.unique())])
    sizes = df.groupby("silo_id").size().sort_values(ascending=False)
    n_raw = len(sizes)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k == 1:
        return {"silo_all": df.copy()}
    if k < n_raw:
        keep = list(sizes.index[: k - 1])
        parts = {}
        for s in keep:
            parts[names[s]] = df[df.silo_id == s]
        rest = df[~df.silo_id.isin(keep)]
        if len(rest):
            parts["silo_other"] = rest
        return parts

    # k >= n_raw: keep every institution cleanly named, then split largest silos if k > n_raw
    parts = {names[s]: df[df.silo_id == s] for s in sizes.index}
    if k == n_raw:
        return parts

    extra = k - n_raw
    # Allocate sub-silo counts greedily to the largest institutions (branch/subsidiary splits)
    split_counts = {names[s]: 1 for s in sizes.index}
    base_sizes = {names[s]: int(sizes[s]) for s in sizes.index}
    for _ in range(extra):
        target = max(split_counts, key=lambda nm: base_sizes[nm] / split_counts[nm])
        split_counts[target] += 1

    out = {}
    for nm, sub in parts.items():
        m = split_counts[nm]
        if m == 1:
            out[nm] = sub
            continue
        if "customer_id" in sub.columns:
            # Keep every customer's stream intact inside a single branch sub-silo
            buckets = (sub["customer_id"].to_numpy(np.int64) * 2654435761 + seed) % m
        else:
            buckets = np.arange(len(sub)) % m
        for b in range(m):
            chunk = sub[buckets == b]
            if len(chunk):
                out[f"{nm}_b{b + 1}"] = chunk
    return out


def equal_partition(df: pd.DataFrame, k: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    parts = {}
    for i, chunk in enumerate(np.array_split(idx, k)):
        parts[f"client_{i}"] = df.iloc[np.sort(chunk)]
    return parts


def dirichlet_partition(df: pd.DataFrame, k: int, alpha: float, seed: int = 42,
                        max_retries: int = 30) -> dict:
    """Label-skew Dirichlet partition (standard non-IID benchmark).

    Resamples extreme draws up to `max_retries` times so every client receives a
    non-empty partition (and non-empty training split when `split` is present),
    while preserving genuine Dirichlet(alpha) label and quantity skew.
    """
    rng = np.random.default_rng(seed)
    y = df.is_fraud.to_numpy()
    classes = np.unique(y)
    has_split = "split" in df.columns
    is_train = (df["split"].to_numpy() == "train") if has_split else None
    best_parts = None
    best_nonempty = -1

    for _ in range(max(max_retries, 1)):
        parts = {f"client_{i}": [] for i in range(k)}
        for cls in classes:
            idx = np.flatnonzero(y == cls).copy()
            rng.shuffle(idx)
            props = rng.dirichlet(np.full(k, alpha))
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                if len(chunk):
                    parts[f"client_{i}"].append(chunk)
        idx_map = {
            name: np.sort(np.concatenate(chunks))
            for name, chunks in parts.items() if chunks
        }
        if has_split:
            n_ok = sum(
                1 for arr in idx_map.values()
                if len(arr) > 0 and int(is_train[arr].sum()) >= 2
            )
        else:
            n_ok = sum(1 for arr in idx_map.values() if len(arr) > 0)
        if n_ok > best_nonempty:
            best_nonempty = n_ok
            best_parts = idx_map
        if n_ok == k:
            break

    return {
        name: df.iloc[arr]
        for name, arr in best_parts.items() if len(arr) > 0
    }


def partition(df: pd.DataFrame, mode: str, k: int, alpha: float = 0.5,
              seed: int = 42) -> dict:
    if mode == "profile":
        return profile_partition(df, k, seed=seed)
    if mode == "equal":
        return equal_partition(df, k, seed)
    if mode == "dirichlet":
        return dirichlet_partition(df, k, alpha, seed)
    raise ValueError(mode)
