"""Leakage-safe, SILO-LOCAL window features.

Important FL semantics: each institution can only compute features over its OWN
transaction stream. All velocity / in-degree / z-score features here are computed
within (silo, key) groups using only PAST events (two-pointer windows + expanding
statistics). Cross-silo patterns (e.g., a mule receiving across institutions) are
deliberately NOT visible in features — that is precisely what federated model
sharing should help with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _customer_features(ts, amt, payee_id, pre_known_row=None):
    """Per-customer chronologically ordered arrays -> velocity/gap/z/new_payee features.

    `new_payee` is derived strictly from the chronological `(silo, customer, payee)`
    history (warm-started with pre-horizon recurring contacts for established
    accounts so day-1 has no artificial cold-start spike).
    """
    n = len(ts)
    v1 = np.zeros(n, np.int32); v24 = np.zeros(n, np.int32)
    gap = np.full(n, 7 * 86400, np.float64); z = np.zeros(n, np.float64)
    new_p = np.zeros(n, np.int8)
    seen = set()
    if pre_known_row is not None:
        for p in pre_known_row:
            ip = int(p)
            if ip >= 0:
                seen.add(ip)
    l1 = l24 = 0
    s = sq = 0.0; cnt = 0
    for i in range(n):
        t = ts[i]
        while ts[l1] < t - 3600:
            l1 += 1
        while ts[l24] < t - 86400:
            l24 += 1
        v1[i] = i - l1
        v24[i] = i - l24
        if i > 0:
            gap[i] = t - ts[i - 1]
        if cnt >= 3:
            mean = s / cnt
            var = max(sq / cnt - mean * mean, 0.0)
            std = np.sqrt(var)
            z[i] = np.clip((amt[i] - mean) / (std + 1e-6), -10, 10)
        s += amt[i]; sq += amt[i] * amt[i]; cnt += 1
        pid = int(payee_id[i])
        if pid in seen:
            new_p[i] = 0
        else:
            new_p[i] = 1
            seen.add(pid)
    return v1, v24, gap, z, new_p


def _payee_indegree(ts, payer):
    n = len(ts)
    out = np.zeros(n, np.int32)
    from collections import defaultdict
    counter = defaultdict(int)
    left = 0
    for i in range(n):
        t = ts[i]
        while ts[left] < t - 86400:
            p = payer[left]
            counter[p] -= 1
            if counter[p] == 0:
                del counter[p]
            left += 1
        out[i] = len(counter)
        counter[payer[i]] += 1
    return out


def _merchant_features(ts):
    n = len(ts)
    v24 = np.zeros(n, np.int32); z = np.zeros(n, np.float64)
    left = 0
    s = sq = 0.0; cnt = 0
    for i in range(n):
        t = ts[i]
        while ts[left] < t - 86400:
            left += 1
        c = i - left
        v24[i] = c
        if cnt >= 5:
            mean = s / cnt
            var = max(sq / cnt - mean * mean, 0.0)
            z[i] = np.clip((c - mean) / (np.sqrt(var) + 1e-6), -10, 10)
        s += c; sq += c * c; cnt += 1
    return v24, z


def add_window_features(df: pd.DataFrame, pre_known_payees: np.ndarray | None = None) -> pd.DataFrame:
    if pre_known_payees is None:
        pre_known_payees = df.attrs.get("pre_known_payees")
    df = df.copy()
    n = len(df)
    v1 = np.zeros(n, np.int32); v24 = np.zeros(n, np.int32)
    gap = np.full(n, 7 * 86400.0); az = np.zeros(n)
    new_p = np.zeros(n, np.int8)
    indeg = np.zeros(n, np.int32)
    mv24 = np.zeros(n, np.int32); mz = np.zeros(n)

    ts_all = df.ts.to_numpy(np.int64)
    amt_all = df.amount.to_numpy(np.float64)
    cid_all = df.customer_id.to_numpy(np.int64)
    pid_all = df.payee_id.to_numpy(np.int64)
    n_known = len(pre_known_payees) if pre_known_payees is not None else 0

    # customer-level (within silo)
    order = np.lexsort((ts_all, cid_all, df.silo_id.to_numpy()))
    keys = df.silo_id.to_numpy()[order] * (cid_all.max() + 1) + cid_all[order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    for chunk in np.split(order, bounds):
        c_id = int(cid_all[chunk[0]])
        pk_row = pre_known_payees[c_id] if c_id < n_known else None
        a, b, c, d, np_c = _customer_features(
            ts_all[chunk], amt_all[chunk], pid_all[chunk], pre_known_row=pk_row
        )
        v1[chunk] = a; v24[chunk] = b; gap[chunk] = c; az[chunk] = d; new_p[chunk] = np_c

    # payee in-degree (within silo)
    order = np.lexsort((ts_all, df.payee_id.to_numpy(), df.silo_id.to_numpy()))
    keys = df.silo_id.to_numpy()[order] * (df.payee_id.max() + 1) + df.payee_id.to_numpy()[order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    payers = df.customer_id.to_numpy()
    for chunk in np.split(order, bounds):
        indeg[chunk] = _payee_indegree(ts_all[chunk], payers[chunk])

    # merchant velocity (within silo, merchant txns only)
    m_mask = df.merchant_id.to_numpy() >= 0
    m_idx = np.flatnonzero(m_mask)
    if len(m_idx):
        mids = df.merchant_id.to_numpy()
        order = m_idx[np.lexsort((ts_all[m_idx], mids[m_idx], df.silo_id.to_numpy()[m_idx]))]
        keys = df.silo_id.to_numpy()[order] * (mids.max() + 1) + mids[order]
        bounds = np.flatnonzero(np.diff(keys)) + 1
        for chunk in np.split(order, bounds):
            a, b = _merchant_features(ts_all[chunk])
            mv24[chunk] = a; mz[chunk] = b

    df["velocity_1h"] = v1
    df["velocity_24h"] = v24
    df["gap_prev_sec"] = gap
    df["amount_z"] = az
    df["new_payee"] = new_p
    df["payee_indegree_24h"] = indeg
    df["merchant_velocity_24h"] = mv24
    df["merchant_vel_z"] = mz
    return df
