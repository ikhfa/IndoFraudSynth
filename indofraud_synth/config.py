"""Generator configuration (dataclasses + YAML loading)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


# Rail definitions. `mean` = target mean ticket in IDR; `sigma` = lognormal sigma;
# `cap` = per-transaction regulatory/product ceiling (documented in calibration.py).
DEFAULT_RAILS = {
    "QRIS":    {"share": 0.45, "mean": 48_000,    "sigma": 0.9, "cap": 10_000_000},
    "EWALLET": {"share": 0.25, "mean": 150_000,   "sigma": 1.0, "cap": 20_000_000},
    "BIFAST":  {"share": 0.15, "mean": 2_470_000, "sigma": 1.1, "cap": 250_000_000},
    "VA":      {"share": 0.12, "mean": 250_000,   "sigma": 0.9, "cap": 50_000_000},
    "CARD":    {"share": 0.03, "mean": 500_000,   "sigma": 1.0, "cap": 100_000_000},
}

# Target fraud rate per rail (fraction of transactions). Plausible-range assumptions,
# NOT official statistics; sensitivity analysis is required in any publication.
DEFAULT_FRAUD_RATES = {
    "QRIS": 0.0008, "EWALLET": 0.0025, "BIFAST": 0.0040, "VA": 0.0015, "CARD": 0.0020,
}

# Typology mix per rail (shares of fraudulent transactions on that rail).
# ATO  = account-takeover drain (device change + burst to new payees)
# APP  = authorized push payment / social-engineering scam (normal device!)
# MULE = money-mule fan-out legs (fast pass-through to aggregators)
# QRIS_ABUSE = merchant-side static-QR tampering / quishing
DEFAULT_TYPOLOGY_MIX = {
    "QRIS":    {"QRIS_ABUSE": 1.0},
    "EWALLET": {"ATO": 0.35, "APP": 0.30, "MULE": 0.35},
    "BIFAST":  {"ATO": 0.40, "APP": 0.35, "MULE": 0.25},
    "VA":      {"APP": 1.0},
    "CARD":    {"APP": 1.0},
}

# Institutional silo profiles (customer primary-bank market shares, assumed;
# inspired by the KBMI structure: 2 dominant banks, mid banks, regional BPDs).
BANK_WEIGHTS = {
    "bank_big_1": 0.36, "bank_big_2": 0.25, "bank_mid_1": 0.15,
    "bank_mid_2": 0.11, "bpd_1": 0.075, "bpd_2": 0.035, "bank_small_1": 0.02,
}
WALLET_WEIGHTS = {"ewallet_1": 0.45, "ewallet_2": 0.32, "ewallet_3": 0.18, "none": 0.05}

REGIONS = ["jabodetabek", "kota_besar", "kota_kecil", "rural"]
REGION_WEIGHTS = [0.25, 0.30, 0.25, 0.20]
INCOME_WEIGHTS = [0.45, 0.40, 0.15]          # low / mid / high
INCOME_AMOUNT_MULT = [0.6, 1.0, 2.2]
INCOME_ACTIVITY_MULT = [0.7, 1.0, 1.6]

MERCHANT_CATEGORIES = ["warung", "fnb", "retail", "marketplace", "pulsa", "transport", "services"]
MERCHANT_CAT_WEIGHTS = [0.35, 0.20, 0.15, 0.10, 0.10, 0.07, 0.03]

# Diurnal activity weights (00:00 .. 23:00), peaks at lunch and evening.
HOUR_WEIGHTS = [0.5, 0.3, 0.2, 0.2, 0.3, 0.8, 1.5, 2.5, 3.0, 3.2, 3.5, 4.0,
                4.5, 3.8, 3.2, 3.0, 3.2, 3.8, 4.5, 4.8, 4.2, 3.2, 2.0, 1.0]


@dataclass
class SynthConfig:
    seed: int = 42
    n_customers: int = 5_000
    n_merchants: int = 800
    horizon_days: int = 180
    start_date: str = "2025-07-01"          # covers 11.11 / 12.12 Harbolnas promos
    base_activity: float = 0.7              # txns / customer / day (mid segment)
    rails: dict = field(default_factory=lambda: DEFAULT_RAILS.copy())
    fraud_rates: dict = field(default_factory=lambda: DEFAULT_FRAUD_RATES.copy())
    typology_mix: dict = field(default_factory=lambda: DEFAULT_TYPOLOGY_MIX.copy())
    promo_days: list = field(default_factory=lambda: ["2025-11-11", "2025-12-12"])
    thr_window: Optional[list] = None       # optional ["YYYY-MM-DD", "YYYY-MM-DD"]
    payday_mult: float = 1.6                # gajian uplift (25th..1st)
    weekend_mult: float = 1.15
    promo_mult: float = 2.6                 # Harbolnas uplift
    thr_mult: float = 1.4
    device_change_noise: float = 0.005      # benign device switches / reinstalls
    drift_strength: float = 0.25            # test-window fraud-campaign adaptation (0 = stationary)
    out_dir: str = "data/indofraud_v0"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_yaml(path: str) -> "SynthConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        section = raw.get("synth", raw)
        cfg = SynthConfig()
        for k, v in section.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # normalize rail share sums
        total = sum(r["share"] for r in cfg.rails.values())
        for r in cfg.rails.values():
            r["share"] = r["share"] / total
        return cfg
