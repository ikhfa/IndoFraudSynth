"""Vectorized legitimate transaction generation with Indonesian seasonality."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    SynthConfig, HOUR_WEIGHTS, INCOME_WEIGHTS, INCOME_AMOUNT_MULT, INCOME_ACTIVITY_MULT,
    REGIONS,
)

RAIL_NAMES = ["QRIS", "EWALLET", "BIFAST", "VA", "CARD"]
MERCHANT_RAILS = {"QRIS", "VA", "CARD"}          # payee is a merchant
P2P_RAILS = {"EWALLET", "BIFAST"}                # payee is a peer account


def calendar_weights(cfg: SynthConfig):
    start = pd.Timestamp(cfg.start_date)
    days = pd.date_range(start, periods=cfg.horizon_days, freq="D")
    w = np.ones(len(days))
    dom = days.day.to_numpy()
    w[np.isin(dom, [25, 26, 27, 28, 29, 30, 31, 1])] *= cfg.payday_mult
    w[days.day_of_week.to_numpy() >= 5] *= cfg.weekend_mult
    promo = pd.to_datetime(cfg.promo_days)
    for p in promo:
        idx = (days >= p - pd.Timedelta(days=0)) & (days <= p)
        w[idx] *= cfg.promo_mult
        pre = (days >= p - pd.Timedelta(days=3)) & (days < p)
        w[pre] *= 1.3
    if cfg.thr_window:
        a, b = pd.to_datetime(cfg.thr_window[0]), pd.to_datetime(cfg.thr_window[1])
        w[(days >= a) & (days <= b)] *= cfg.thr_mult
    is_payday = np.isin(dom, [25, 26, 27, 28, 29, 30, 31, 1]).astype(np.int8)
    is_promo = np.zeros(len(days), dtype=np.int8)
    for p in promo:
        is_promo[(days >= p) & (days <= p)] = 1
    return days, w / w.sum(), is_payday, is_promo


def gen_legit(cfg: SynthConfig, rng: np.random.Generator, agents: pd.DataFrame,
              merchants: pd.DataFrame, meta: dict) -> pd.DataFrame:
    n = cfg.n_customers
    n_mules = meta.get("n_mules", 0)
    n_active = n + n_mules
    active = agents.iloc[:n_active]
    act = np.array(INCOME_ACTIVITY_MULT)[active.income.to_numpy()]
    # Mule accounts also have legitimate background activity (at lower frequency)
    # so they are not trivially separable as fraud-only accounts.
    is_mule_flag = active.is_mule.to_numpy() == 1
    act = np.where(is_mule_flag, act * 0.35, act)
    lam = cfg.base_activity * act * cfg.horizon_days
    counts = rng.poisson(lam)
    N = int(counts.sum())

    days, day_p, is_payday_d, is_promo_d = calendar_weights(cfg)
    hour_p = np.array(HOUR_WEIGHTS); hour_p = hour_p / hour_p.sum()

    cidx = np.repeat(np.arange(n_active), counts)
    rail_names = list(cfg.rails.keys())
    qris_idx = rail_names.index("QRIS") if "QRIS" in rail_names else -1
    rail = rng.choice(len(rail_names), N, p=[cfg.rails[r]["share"] for r in rail_names])
    day = rng.choice(len(days), N, p=day_p)
    hour = rng.choice(24, N, p=hour_p)
    start_epoch = int(pd.Timestamp(cfg.start_date).timestamp())
    ts = (start_epoch + day * 86400 + hour * 3600
          + rng.integers(0, 60, N) * 60 + rng.integers(0, 60, N)).astype(np.int64)

    # Amounts: lognormal calibrated to per-rail mean tickets, income-scaled.
    means = np.array([cfg.rails[r]["mean"] for r in rail_names])
    sigmas = np.array([cfg.rails[r]["sigma"] for r in rail_names])
    caps = np.array([cfg.rails[r]["cap"] for r in rail_names])
    inc_mult = np.array(INCOME_AMOUNT_MULT)[active.income.to_numpy()[cidx]]
    expected_inc_mult = (
        sum(w * a * m for w, a, m in zip(INCOME_WEIGHTS, INCOME_ACTIVITY_MULT, INCOME_AMOUNT_MULT))
        / sum(w * a for w, a in zip(INCOME_WEIGHTS, INCOME_ACTIVITY_MULT))
    )
    mu = np.log(means[rail]) - sigmas[rail] ** 2 / 2 + np.log(inc_mult / expected_inc_mult)
    amount = np.exp(rng.normal(mu, sigmas[rail]))
    amount = np.clip(amount, 1_000, caps[rail])
    amount = np.where(rail == qris_idx, np.round(amount / 500) * 500, np.round(amount / 1000) * 1000)
    amount = np.maximum(amount, 500)

    # Payees: regional-affinity recurring merchants for QRIS; national recurring
    # merchants for VA/CARD; regionally anchored recurring peer circles for P2P.
    merch_pop = merchants.popularity.to_numpy()
    merch_regions = merchants.region.to_numpy()
    n_regions = len(REGIONS)
    reg_merch_ids = []
    reg_merch_pop = []
    for r in range(n_regions):
        ids_r = np.flatnonzero(merch_regions == r)
        if len(ids_r) == 0:
            ids_r = np.arange(len(merchants))
        p_r = merch_pop[ids_r]
        p_r = p_r / p_r.sum()
        reg_merch_ids.append(ids_r)
        reg_merch_pop.append(p_r)

    active_regions = active.region.to_numpy()
    active_tenure0 = active.tenure0_days.to_numpy()
    cust_regions = agents.region.to_numpy()[:n]
    reg_custs = []
    for r in range(n_regions):
        cr = np.flatnonzero(cust_regions == r)
        reg_custs.append(cr if len(cr) > 0 else np.arange(n))

    # Per-customer recurring merchant favorites (4 local for QRIS, 3 national for VA/CARD)
    fav_local = np.empty((n_active, 4), dtype=np.int64)
    for r in range(n_regions):
        c_in_r = np.flatnonzero(active_regions == r)
        if len(c_in_r):
            fav_local[c_in_r] = rng.choice(
                reg_merch_ids[r], size=(len(c_in_r), 4), p=reg_merch_pop[r]
            )
    fav_online = rng.choice(len(merchants), size=(n_active, 3), p=merch_pop)

    merchant_id = np.full(N, -1, dtype=np.int64)
    payee_id = np.full(N, -1, dtype=np.int64)
    merch_acct_base = len(agents)

    # 1. QRIS: strongly local (60% recurring local warung/F&B, 22% other local, 18% travel/global)
    q_mask = (rail == qris_idx) if qris_idx >= 0 else np.zeros(N, dtype=bool)
    n_q = int(q_mask.sum())
    if n_q:
        q_cidx = cidx[q_mask]
        q_reg = active_regions[q_cidx]
        u = rng.random(n_q)
        q_m = np.empty(n_q, dtype=np.int64)
        fav_m = u < 0.60
        loc_m = (u >= 0.60) & (u < 0.82)
        glb_m = u >= 0.82
        if fav_m.any():
            q_m[fav_m] = fav_local[q_cidx[fav_m], rng.integers(0, 4, int(fav_m.sum()))]
        if loc_m.any():
            for r in range(n_regions):
                rm = loc_m & (q_reg == r)
                cnt_r = int(rm.sum())
                if cnt_r:
                    q_m[rm] = rng.choice(reg_merch_ids[r], cnt_r, p=reg_merch_pop[r])
        if glb_m.any():
            q_m[glb_m] = rng.choice(len(merchants), int(glb_m.sum()), p=merch_pop)
        merchant_id[q_mask] = q_m
        payee_id[q_mask] = merch_acct_base + q_m

    # 2. Online merchant rails (VA, CARD): 55% recurring national merchants, 45% global
    online_ridxs = [rail_names.index(r) for r in ("VA", "CARD") if r in rail_names]
    o_mask = np.isin(rail, online_ridxs)
    n_o = int(o_mask.sum())
    if n_o:
        o_cidx = cidx[o_mask]
        o_fav = rng.random(n_o) < 0.55
        o_m = np.empty(n_o, dtype=np.int64)
        if o_fav.any():
            o_m[o_fav] = fav_online[o_cidx[o_fav], rng.integers(0, 3, int(o_fav.sum()))]
        if (~o_fav).any():
            o_m[~o_fav] = rng.choice(len(merchants), int((~o_fav).sum()), p=merch_pop)
        merchant_id[o_mask] = o_m
        payee_id[o_mask] = merch_acct_base + o_m

    m_mask = q_mask | o_mask

    # 3. P2P rails (EWALLET, BIFAST): regionally anchored recurring peer circles + 25% bills
    p_mask = ~m_mask
    n_p = int(p_mask.sum())
    if n_p:
        p_cidx = cidx[p_mask]
        p_reg = active_regions[p_cidx]
        # recurring payee set: k ~ geometric -> small stable circle
        k = np.minimum(rng.geometric(0.35, n_p) - 1, 12)
        # k % 3 != 2 (~70% of volume) -> same-region peer; k % 3 == 2 -> cross-region remittance
        peer = (p_cidx * 7 + k * 13 + 3) % n
        same_reg_peer = (k % 3) != 2
        for r in range(n_regions):
            rm = same_reg_peer & (p_reg == r)
            if rm.any():
                pool = reg_custs[r]
                peer[rm] = pool[(p_cidx[rm] * 7 + k[rm] * 13 + 3) % len(pool)]
        # 25% of P2P goes to merchants as bill/payment (60% recurring favorites, 40% global)
        bill = rng.random(n_p) < 0.25
        n_bill = int(bill.sum())
        m_p = np.full(n_p, -1, dtype=np.int64)
        if n_bill:
            b_cidx = p_cidx[bill]
            b_fav = rng.random(n_bill) < 0.60
            b_idx = np.empty(n_bill, dtype=np.int64)
            if b_fav.any():
                b_idx[b_fav] = fav_online[b_cidx[b_fav], rng.integers(0, 3, int(b_fav.sum()))]
            if (~b_fav).any():
                b_idx[~b_fav] = rng.choice(len(merchants), int((~b_fav).sum()), p=merch_pop)
            peer[bill] = merch_acct_base + b_idx
            m_p[bill] = b_idx
        payee_id[p_mask] = peer
        merchant_id[p_mask] = m_p

    # Pre-horizon known payees for established accounts (tenure0 >= 30d):
    # top 3 local merchants + top 2 online merchants + top 3 recurring P2P peers (k=0,1,2).
    # Used by `add_window_features` to compute graph-consistent `new_payee` without day-1 cold start.
    pre_known = np.full((n_active, 8), -1, dtype=np.int64)
    est = active_tenure0 >= 30
    if est.any():
        est_ids = np.flatnonzero(est)
        pre_known[est_ids, 0:3] = merch_acct_base + fav_local[est_ids, 0:3]
        pre_known[est_ids, 3:5] = merch_acct_base + fav_online[est_ids, 0:2]
        # k=0, 1 (same-region peers) and k=2 (recurring cross-region peer)
        pre_known[est_ids, 7] = (est_ids * 7 + 2 * 13 + 3) % n
        for r in range(n_regions):
            er = est_ids[active_regions[est_ids] == r]
            if len(er):
                pool = reg_custs[r]
                pre_known[er, 5] = pool[(er * 7 + 3) % len(pool)]
                pre_known[er, 6] = pool[(er * 7 + 13 + 3) % len(pool)]

    # Placeholder (overwritten chronologically from actual payee graph in add_window_features)
    new_payee = np.zeros(N, dtype=np.int8)
    device_change = (rng.random(N) < cfg.device_change_noise).astype(np.int8)

    # Region mismatch between payer and payee region.
    payer_region = active_regions[cidx]
    payee_region = np.zeros(N, dtype=int)
    payee_region[m_mask] = merch_regions[merchant_id[m_mask]]
    if n_p:
        peer_ids = payee_id[p_mask]
        is_peer = peer_ids < len(agents)
        pr = np.zeros(n_p, dtype=int)
        pr[is_peer] = agents.region.to_numpy()[peer_ids[is_peer]]
        pr[~is_peer] = merch_regions[peer_ids[~is_peer] - merch_acct_base]
        payee_region[p_mask] = pr
    region_mismatch = (payer_region != payee_region).astype(np.int8)

    df = pd.DataFrame({
        "ts": ts, "customer_id": cidx, "payee_id": payee_id, "merchant_id": merchant_id,
        "rail": rail, "amount": amount.astype(np.int64),
        "hour": hour.astype(np.int8), "dow": days[day].day_of_week.to_numpy().astype(np.int8),
        "is_payday": is_payday_d[day], "is_promo": is_promo_d[day],
        "is_night": (hour < 6).astype(np.int8),
        "device_change": device_change, "new_payee": new_payee,
        "region_mismatch": region_mismatch,
        "tenure_days": active_tenure0[cidx] + day,
        "merchant_cat": np.where(merchant_id >= 0,
                                 merchants.category.to_numpy()[np.maximum(merchant_id, 0)], -1),
        "is_fraud": np.zeros(N, dtype=np.int8),
        "typology": "legit", "campaign_id": -1,
    })
    df.attrs["rail_names"] = rail_names
    df.attrs["pre_known_payees"] = pre_known
    return df
