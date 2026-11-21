"""Fraud campaign injection (agent-level campaigns, not i.i.d. label flips).

Typologies (informed by Satgas PASTI / OJK IASC modus operandi and PPATK patterns):
  ATO        account takeover: device change + rapid drain to fresh mule payees
  APP        authorized push payment (social engineering): NORMAL device, new payee,
             urgency gaps; victim authorizes the transfer themselves
  MULE       fan-out legs: mule accounts disperse received funds to aggregators fast
  QRIS_ABUSE merchant-side static-QR tampering: many distinct payers, small tickets,
             short window at one merchant
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SynthConfig

RAIL_NAMES_DEFAULT = ["QRIS", "EWALLET", "BIFAST", "VA", "CARD"]


def _base_rows(n, ts, customer_id, payee_id, merchant_id, rail, amount, meta_arrays):
    hour = ((ts % 86400) // 3600).astype(np.int8)
    return pd.DataFrame({
        "ts": ts.astype(np.int64), "customer_id": customer_id, "payee_id": payee_id,
        "merchant_id": merchant_id, "rail": rail, "amount": amount.astype(np.int64),
        "hour": hour, "dow": meta_arrays["dow"], "is_payday": meta_arrays["is_payday"],
        "is_promo": meta_arrays["is_promo"], "is_night": (hour < 6).astype(np.int8),
        "device_change": meta_arrays["device_change"], "new_payee": meta_arrays["new_payee"],
        "region_mismatch": meta_arrays["region_mismatch"], "tenure_days": meta_arrays["tenure"],
        "merchant_cat": meta_arrays["merchant_cat"],
        "is_fraud": np.ones(n, dtype=np.int8), "typology": meta_arrays["typology"],
        "campaign_id": meta_arrays["campaign_id"],
    })


def inject_fraud(cfg: SynthConfig, rng: np.random.Generator, legit: pd.DataFrame,
                 agents: pd.DataFrame, merchants: pd.DataFrame, meta: dict) -> pd.DataFrame:
    rail_names = legit.attrs.get("rail_names", RAIL_NAMES_DEFAULT)
    ridx = {r: i for i, r in enumerate(rail_names)}
    n = cfg.n_customers
    mule_ids = meta["mule_ids"]; agg_ids = meta["agg_ids"]
    start_epoch = int(pd.Timestamp(cfg.start_date).timestamp())
    horizon_sec = cfg.horizon_days * 86400
    drift_start = start_epoch + int(0.70 * horizon_sec)
    caps = {r: cfg.rails[r]["cap"] for r in rail_names}
    means = {r: cfg.rails[r]["mean"] for r in rail_names}
    # Causal mule intake tracker: mule_id -> list of (intake_ts, amount)
    received: dict[int, list[tuple[int, int]]] = {int(m): [] for m in mule_ids}

    # Precompute calendar metadata once (instead of inside every campaign).
    days = pd.date_range(pd.Timestamp(cfg.start_date), periods=cfg.horizon_days, freq="D")
    dom = days.day.to_numpy()
    dow_arr = days.day_of_week.to_numpy().astype(np.int8)
    payday_arr = np.isin(dom, [25, 26, 27, 28, 29, 30, 31, 1]).astype(np.int8)
    promo_arr = np.zeros(len(days), dtype=np.int8)
    for p in pd.to_datetime(cfg.promo_days):
        promo_arr[(days >= p) & (days <= p)] = 1

    agent_regions = agents.region.to_numpy()
    agent_tenure0 = agents.tenure0_days.to_numpy()
    merch_regions = merchants.region.to_numpy()
    merch_cats = merchants.category.to_numpy()
    # E-commerce / pulsa / services merchants used by VA/CARD scam flows.
    va_card_merch = merchants.index[merchants.category.isin([3, 4, 6])].to_numpy()
    if len(va_card_merch) == 0:
        va_card_merch = merchants.index.to_numpy()
    static_merch = merchants.index[merchants.qris_static == 1].to_numpy()
    if len(static_merch) == 0:
        static_merch = merchants.index.to_numpy()
    merch_acct_base = len(agents)

    frames = []
    cid = 0

    horizon_end = start_epoch + horizon_sec

    def day_meta(ts):
        day_idx = np.clip(((ts - start_epoch) // 86400).astype(int),
                          0, cfg.horizon_days - 1)
        return dow_arr[day_idx], payday_arr[day_idx], promo_arr[day_idx]

    def rand_ts(k, night_bias=0.0, tail_margin_sec: int = 7200):
        # Leave a campaign-duration buffer before horizon_end so multi-leg
        # campaigns do not clamp and pile up on the final second of test.
        max_margin = max(horizon_sec // 5, 1)
        margin = min(max(int(tail_margin_sec), 1), max_margin)
        hi = max(horizon_sec - margin, 1)
        t = start_epoch + rng.integers(0, hi, k)
        if night_bias > 0:
            night = rng.random(k) < night_bias
            # shift into 00:00-05:00 of the same day (clipped inside horizon)
            shifted = (t[night] // 86400) * 86400 + rng.integers(0, 5 * 3600, int(night.sum()))
            t[night] = np.clip(shifted, start_epoch, start_epoch + hi - 1)
        return t

    def bounded_leg_ts(t0: int, raw_offsets: np.ndarray) -> np.ndarray:
        """Place leg timestamps inside [t0, horizon_end - 1] without end-second pileup."""
        ts = t0 + raw_offsets
        if len(ts) == 0 or int(ts[-1]) < horizon_end:
            return ts.astype(np.int64)
        avail = max(horizon_end - 1 - t0, len(ts))
        span = max(int(raw_offsets[-1]), 1)
        scaled = np.floor(raw_offsets * (avail / span)).astype(np.int64)
        # Ensure strictly non-colliding spacing when multiple legs fit in `avail`
        for i in range(1, len(scaled)):
            if scaled[i] <= scaled[i - 1]:
                scaled[i] = min(scaled[i - 1] + 1, avail)
        return np.minimum(t0 + scaled, horizon_end - 1).astype(np.int64)

    def tenure(cids, ts):
        return agent_tenure0[cids] + ((ts - start_epoch) // 86400)

    def drift_scale(t0: int) -> tuple[float, float]:
        """In the final 30% of the horizon, fraudsters adapt (stealthier amounts,
        wider inter-leg spacing) to create realistic out-of-time concept drift."""
        d = float(getattr(cfg, "drift_strength", 0.0))
        if d <= 0 or t0 < drift_start:
            return 1.0, 1.0
        frac = min((t0 - drift_start) / max(horizon_sec * 0.30, 1.0), 1.0)
        amt_mult = max(0.5, 1.0 - 0.35 * d * frac)
        gap_mult = 1.0 + 0.8 * d * frac
        return amt_mult, gap_mult

    # Ensure MULE runs after ATO/APP within each rail so intake events exist.
    typ_order = {"ATO": 0, "APP": 1, "QRIS_ABUSE": 2, "MULE": 3}

    for rail in rail_names:
        n_legit_rail = int((legit.rail == ridx[rail]).sum())
        target = int(round(cfg.fraud_rates.get(rail, 0.0) * n_legit_rail))
        if target <= 0:
            continue
        mix = cfg.typology_mix.get(rail, {})
        ordered_mix = sorted(mix.items(), key=lambda kv: typ_order.get(kv[0], 99))
        for typ, share in ordered_mix:
            need = int(round(target * share))
            produced = 0
            guard = 0
            while produced < need and guard < 10000:
                guard += 1
                cid += 1
                if typ == "ATO":
                    if produced > 0 and need - produced < 2:
                        break
                    legs = int(rng.integers(3, 9))
                    legs = min(legs, max(need - produced, 2 if produced == 0 else 3))
                    if need < 3 and produced == 0:
                        legs = max(need, 1)
                    victim = int(rng.integers(0, n))
                    t0 = int(rand_ts(1, night_bias=0.45, tail_margin_sec=6 * 3600)[0])
                    amt_d, gap_d = drift_scale(t0)
                    ts = bounded_leg_ts(
                        t0, np.sort(rng.integers(60, int(2400 * gap_d), legs)).cumsum(),
                    )
                    payees = rng.choice(mule_ids, legs)
                    base_amt = max(means[rail] * 4.5, 300_000) * amt_d
                    amt = np.clip(np.exp(rng.normal(np.log(base_amt), 0.85, legs)),
                                  50_000, caps[rail])
                    for p, t_leg, a in zip(payees, ts, amt):
                        received[int(p)].append((int(t_leg), int(a)))
                    dow, pd_, pr = day_meta(ts)
                    # Most ATO triggers device_change; ~15% session/SIM hijack does not.
                    dev_chg = (rng.random(legs) < 0.85).astype(np.int8)
                    new_p = (rng.random(legs) < 0.90).astype(np.int8)
                    frames.append(_base_rows(
                        legs, ts, np.full(legs, victim), payees, np.full(legs, -1),
                        np.full(legs, ridx[rail]), amt,
                        {"dow": dow, "is_payday": pd_, "is_promo": pr,
                         "device_change": dev_chg,
                         "new_payee": new_p,
                         "region_mismatch": (agent_regions[payees] !=
                                             agent_regions[victim]).astype(np.int8),
                         "tenure": tenure(np.full(legs, victim), ts),
                         "merchant_cat": np.full(legs, -1),
                         "typology": "ATO", "campaign_id": np.full(legs, cid)}))
                    produced += legs

                elif typ == "APP":
                    legs = int(rng.integers(1, 5))
                    legs = min(legs, max(need - produced, 1))
                    victim = int(rng.integers(0, n))
                    t0 = int(rand_ts(1, tail_margin_sec=4 * 86400)[0])
                    amt_d, gap_d = drift_scale(t0)
                    ts = bounded_leg_ts(
                        t0, np.sort(rng.integers(300, int(86400 * gap_d), legs)).cumsum(),
                    )
                    base_amt = max(means[rail] * 3.5, 250_000) * amt_d
                    amt = np.clip(np.exp(rng.normal(np.log(base_amt), 0.95, legs)),
                                  50_000, min(50_000_000, caps[rail]))
                    dow, pd_, pr = day_meta(ts)
                    # On merchant rails (VA, CARD), APP scams route through shell/marketplace
                    # merchants rather than merchant_id=-1 (avoiding a deterministic leak).
                    if rail in ("VA", "CARD"):
                        merch = rng.choice(va_card_merch, legs)
                        payees = merch_acct_base + merch
                        m_ids = merch
                        m_cats = merch_cats[merch]
                        reg_mm = (merch_regions[merch] != agent_regions[victim]).astype(np.int8)
                        new_p = (rng.random(legs) < 0.78).astype(np.int8)
                    else:
                        payees = rng.choice(mule_ids, legs)
                        for p, t_leg, a in zip(payees, ts, amt):
                            received[int(p)].append((int(t_leg), int(a)))
                        m_ids = np.full(legs, -1)
                        m_cats = np.full(legs, -1)
                        reg_mm = (agent_regions[payees] != agent_regions[victim]).astype(np.int8)
                        new_p = (rng.random(legs) < 0.85).astype(np.int8)
                    frames.append(_base_rows(
                        legs, ts, np.full(legs, victim), payees, m_ids,
                        np.full(legs, ridx[rail]), amt,
                        {"dow": dow, "is_payday": pd_, "is_promo": pr,
                         "device_change": np.zeros(legs, np.int8),   # social engineering!
                         "new_payee": new_p,
                         "region_mismatch": reg_mm,
                         "tenure": tenure(np.full(legs, victim), ts),
                         "merchant_cat": m_cats,
                         "typology": "APP", "campaign_id": np.full(legs, cid)}))
                    produced += legs

                elif typ == "MULE":
                    funded = [m for m, evs in received.items() if evs]
                    if not funded:
                        break
                    if produced > 0 and need - produced < 2:
                        break
                    mule = int(rng.choice(funded))
                    events = received[mule]
                    legs = int(rng.integers(2, 6))
                    legs = min(legs, max(need - produced, 1 if need < 2 else 2))
                    # Causal fan-out: strictly occurs AFTER the mule receives scam funds.
                    last_in_ts = max(t_in for t_in, _ in events)
                    total_in = sum(a_in for _, a_in in events)
                    t0 = min(last_in_ts + int(rng.integers(300, 7200)),
                             horizon_end - max(legs + 1, 600))
                    t0 = min(max(t0, last_in_ts + 1), horizon_end - 1)
                    _, gap_d = drift_scale(t0)
                    ts = bounded_leg_ts(
                        t0, np.sort(rng.integers(180, int(7200 * gap_d), legs)).cumsum(),
                    )
                    total = total_in * 0.95
                    split = rng.dirichlet(np.ones(legs)) * total
                    amt = np.clip(split, 50_000, caps[rail])
                    payees = rng.choice(agg_ids, legs)
                    received[mule] = []
                    dow, pd_, pr = day_meta(ts)
                    frames.append(_base_rows(
                        legs, ts, np.full(legs, mule), payees, np.full(legs, -1),
                        np.full(legs, ridx[rail]), amt,
                        {"dow": dow, "is_payday": pd_, "is_promo": pr,
                         "device_change": (rng.random(legs) < 0.05).astype(np.int8),
                         "new_payee": (rng.random(legs) < 0.85).astype(np.int8),
                         "region_mismatch": (agent_regions[payees] !=
                                             agent_regions[mule]).astype(np.int8),
                         "tenure": tenure(np.full(legs, mule), ts),
                         "merchant_cat": np.full(legs, -1),
                         "typology": "MULE", "campaign_id": np.full(legs, cid)}))
                    produced += legs

                elif typ == "QRIS_ABUSE":
                    if produced > 0 and need - produced < 10:
                        break
                    merch = int(rng.choice(static_merch))
                    if need < 12 and produced == 0:
                        legs = max(need, 1)
                    else:
                        legs = int(rng.integers(12, 35))
                        if produced > 0:
                            legs = min(legs, max(need - produced, 12))
                        elif need < legs:
                            legs = max(need, 12)
                    t0 = int(rand_ts(1, tail_margin_sec=3 * 86400)[0])
                    amt_d, gap_d = drift_scale(t0)
                    ts = bounded_leg_ts(
                        t0, np.sort(rng.integers(0, int(2 * 86400 * gap_d), legs)),
                    )
                    payers = rng.choice(n, legs, replace=False) if legs <= n else rng.choice(n, legs)
                    amt = np.round(rng.uniform(10_000, 120_000 * amt_d, legs) / 500) * 500
                    amt = np.maximum(amt, 5_000)
                    dow, pd_, pr = day_meta(ts)
                    merch_acct = merch_acct_base + merch
                    frames.append(_base_rows(
                        legs, ts, payers, np.full(legs, merch_acct), np.full(legs, merch),
                        np.full(legs, ridx[rail]), amt,
                        {"dow": dow, "is_payday": pd_, "is_promo": pr,
                         "device_change": np.zeros(legs, np.int8),
                         "new_payee": (rng.random(legs) < 0.70).astype(np.int8),
                         "region_mismatch": (agent_regions[payers] !=
                                             merch_regions[merch]).astype(np.int8),
                         "tenure": tenure(payers, ts),
                         "merchant_cat": np.full(legs, merch_cats[merch]),
                         "typology": "QRIS_ABUSE", "campaign_id": np.full(legs, cid)}))
                    produced += legs
                else:
                    raise ValueError(f"unknown typology {typ}")

    if not frames:
        return legit
    fraud = pd.concat(frames, ignore_index=True)
    out = pd.concat([legit, fraud], ignore_index=True)
    out = out.sort_values("ts", kind="stable").reset_index(drop=True)
    out["txn_id"] = np.arange(len(out))
    out.attrs["rail_names"] = rail_names
    if "pre_known_payees" in legit.attrs:
        out.attrs["pre_known_payees"] = legit.attrs["pre_known_payees"]
    return out
