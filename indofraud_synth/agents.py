"""Agent population: customers, merchants, mule accounts, aggregators."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    SynthConfig, BANK_WEIGHTS, WALLET_WEIGHTS, REGIONS, REGION_WEIGHTS,
    INCOME_WEIGHTS, MERCHANT_CATEGORIES, MERCHANT_CAT_WEIGHTS,
)


def make_agents(cfg: SynthConfig, rng: np.random.Generator):
    n = cfg.n_customers
    banks = list(BANK_WEIGHTS)
    wallets = list(WALLET_WEIGHTS)

    # ~5% newly opened customer accounts (<30d) so low tenure is informative
    # without being a 100% deterministic mule-account shortcut.
    cust_tenure0 = rng.integers(30, 1500, n)
    fresh_cust = rng.random(n) < 0.05
    cust_tenure0[fresh_cust] = rng.integers(1, 30, int(fresh_cust.sum()))

    customers = pd.DataFrame({
        "customer_id": np.arange(n),
        "region": rng.choice(len(REGIONS), n, p=REGION_WEIGHTS),
        "income": rng.choice(3, n, p=INCOME_WEIGHTS),
        "bank": rng.choice(len(banks), n, p=list(BANK_WEIGHTS.values())),
        "wallet": rng.choice(len(wallets), n, p=list(WALLET_WEIGHTS.values())),
        "tenure0_days": cust_tenure0,
        "is_mule": np.zeros(n, dtype=np.int8),
    })

    # Mule pool: accounts that receive scam proceeds and fan them out.
    n_mules = max(60, n // 80)
    mules = pd.DataFrame({
        "customer_id": np.arange(n, n + n_mules),
        "region": rng.choice(len(REGIONS), n_mules, p=REGION_WEIGHTS),
        "income": np.zeros(n_mules, dtype=int),
        "bank": rng.choice(len(banks), n_mules, p=list(BANK_WEIGHTS.values())),
        "wallet": rng.choice(len(wallets) - 1, n_mules,
                             p=np.array(list(WALLET_WEIGHTS.values()))[:-1] /
                               (1 - WALLET_WEIGHTS["none"])),
        "tenure0_days": rng.integers(5, 200, n_mules),   # freshly opened accounts
        "is_mule": np.ones(n_mules, dtype=np.int8),
    })

    # Aggregator accounts (top of the laundering chain).
    n_agg = 5
    aggs = pd.DataFrame({
        "customer_id": np.arange(n + n_mules, n + n_mules + n_agg),
        "region": rng.choice(len(REGIONS), n_agg, p=REGION_WEIGHTS),
        "income": np.full(n_agg, 2),
        "bank": rng.choice(len(banks), n_agg, p=list(BANK_WEIGHTS.values())),
        "wallet": rng.choice(len(wallets) - 1, n_agg,
                             p=np.array(list(WALLET_WEIGHTS.values()))[:-1] /
                               (1 - WALLET_WEIGHTS["none"])),
        "tenure0_days": rng.integers(100, 800, n_agg),
        "is_mule": np.full(n_agg, 2, dtype=np.int8),    # 2 = aggregator
    })

    agents = pd.concat([customers, mules, aggs], ignore_index=True)

    m = cfg.n_merchants
    # Merchant acquirer: 60% bank, 40% e-wallet (QRIS acquirer mix, assumption).
    acq_is_bank = rng.random(m) < 0.6
    acquirer = np.where(
        acq_is_bank,
        rng.choice(len(banks), m, p=list(BANK_WEIGHTS.values())),
        len(banks) + rng.choice(len(wallets) - 1, m,
                                p=np.array(list(WALLET_WEIGHTS.values()))[:-1] /
                                  (1 - WALLET_WEIGHTS["none"])),
    )
    merchants = pd.DataFrame({
        "merchant_id": np.arange(m),
        "category": rng.choice(len(MERCHANT_CATEGORIES), m, p=MERCHANT_CAT_WEIGHTS),
        "region": rng.choice(len(REGIONS), m, p=REGION_WEIGHTS),
        "acquirer": acquirer,                 # index into banks + wallets list
        "acquirer_is_bank": acq_is_bank.astype(np.int8),
        "qris_static": (rng.random(m) < 0.7).astype(np.int8),
    })
    # Zipf-ish popularity: a few marketplaces dominate volume.
    ranks = rng.permutation(m) + 1
    w = 1.0 / np.power(ranks + 10, 0.8)
    merchants["popularity"] = w / w.sum()

    return agents, merchants, {"banks": banks, "wallets": wallets,
                               "n_mules": n_mules, "n_agg": n_agg,
                               "mule_ids": mules.customer_id.to_numpy(),
                               "agg_ids": aggs.customer_id.to_numpy()}
