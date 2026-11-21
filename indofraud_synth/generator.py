"""Generator orchestrator: agents -> legit behavior -> fraud campaigns -> silos
-> silo-local window features -> save + dataset card."""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import __version__
from .agents import make_agents
from .behavior import gen_legit
from .calibration import calibration_markdown
from .config import SynthConfig
from .features import add_window_features
from .fraud import inject_fraud
from .silos import assign_silos


def generate(cfg: SynthConfig):
    rng = np.random.default_rng(cfg.seed)
    agents, merchants, meta = make_agents(cfg, rng)
    legit = gen_legit(cfg, rng, agents, merchants, meta)
    tx = inject_fraud(cfg, rng, legit, agents, merchants, meta)
    pre_known = tx.attrs.pop("pre_known_payees", None)
    rail_names = tx.attrs.get("rail_names", list(cfg.rails.keys()))
    tx = assign_silos(tx, agents, merchants, meta, rail_names)
    tx = add_window_features(tx, pre_known_payees=pre_known)
    tx.attrs["rail_names"] = rail_names
    tx.attrs["silo_names"] = meta["banks"] + [w for w in meta["wallets"] if w != "none"]
    info = {
        "version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "config": cfg.to_dict(),
        "n_transactions": int(len(tx)),
        "n_fraud": int(tx.is_fraud.sum()),
        "fraud_rate": float(tx.is_fraud.mean()),
        "rail_names": rail_names,
        "silo_names": tx.attrs["silo_names"],
        "n_agents": int(len(agents)),
        "n_merchants": int(len(merchants)),
        "typology_counts": {k: int(v) for k, v in tx.typology.value_counts().items()},
    }
    return {"transactions": tx, "agents": agents, "merchants": merchants, "info": info}


DATASET_CARD_TEMPLATE = """# IndoFraud-Synth {version} — Dataset Card

**SYNTHETIC DATA.** No real customer data were used. Generated {generated_utc}.

## Summary
- Transactions: {n_transactions:,} ({n_fraud:,} fraudulent, {fraud_rate_pct:.3f}%)
- Horizon: {horizon_days} days from {start_date}
- Agents: {n_agents:,} accounts (incl. mules/aggregators), {n_merchants:,} merchants
- Rails: {rails}
- Silos (institutions): {silos}
- Typologies: {typologies}

## Intended use
Federated / cross-silo fraud-detection research under realistic Indonesian payment
heterogeneity (quantity skew, rail-mix skew, label skew, temporal drift).
Methodological precedent: PaySim (Lopez-Rojas et al., 2016).

## Calibration targets and sources
{calibration_table}

## Known limitations (must be cited in any publication using this data)
1. Per-rail fraud rates, e-wallet/VA/card ticket sizes, rail shares and institution
   market shares are ASSUMPTIONS within plausible ranges, not official statistics.
2. Fraud campaigns are stylized (burst/urgency/fan-out patterns); real fraudsters
   adapt. Do not claim operational validity without real-data validation.
3. Window features are silo-local by design (no cross-institution visibility).
4. Balance dynamics, KYC fields and device fingerprint richness are simplified.
5. Sensitivity analyses over fraud rates and skew parameters are required.

## Columns
`txn_id, ts, customer_id, payee_id, merchant_id, silo_id, rail, amount, hour, dow,
is_payday, is_promo, is_night, device_change, new_payee, region_mismatch, tenure_days,
merchant_cat, velocity_1h, velocity_24h, gap_prev_sec, amount_z, payee_indegree_24h,
merchant_velocity_24h, merchant_vel_z, is_fraud, typology, campaign_id`

`is_fraud` is the label; `typology`/`campaign_id` are metadata (NEVER features).
"""


def save_dataset(data: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    tx = data["transactions"]
    tx.to_csv(os.path.join(out_dir, "transactions.csv.gz"), index=False,
              compression="gzip")
    data["agents"].to_csv(os.path.join(out_dir, "agents.csv.gz"), index=False,
                          compression="gzip")
    data["merchants"].to_csv(os.path.join(out_dir, "merchants.csv.gz"), index=False,
                             compression="gzip")
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(data["info"], f, indent=2)
    info = data["info"]; cfg = info["config"]
    card = DATASET_CARD_TEMPLATE.format(
        version=info["version"], generated_utc=info["generated_utc"],
        n_transactions=info["n_transactions"], n_fraud=info["n_fraud"],
        fraud_rate_pct=100 * info["fraud_rate"], horizon_days=cfg["horizon_days"],
        start_date=cfg["start_date"], n_agents=info["n_agents"],
        n_merchants=info["n_merchants"], rails=", ".join(info["rail_names"]),
        silos=", ".join(info["silo_names"]),
        typologies=json.dumps(info["typology_counts"]),
        calibration_table=calibration_markdown(),
    )
    with open(os.path.join(out_dir, "DATASET_CARD.md"), "w") as f:
        f.write(card)
    return out_dir
