"""CLI: python -m indofraud_synth --config configs/quick.yaml [--out DIR] [--seed N]"""
from __future__ import annotations

import argparse
import json

from .config import SynthConfig
from .generator import generate, save_dataset
from .validation import calibration_report, save_report, report_markdown


def main():
    ap = argparse.ArgumentParser(description="IndoFraud-Synth generator")
    ap.add_argument("--config", default=None, help="YAML config (synth section)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-customers", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = SynthConfig.from_yaml(args.config) if args.config else SynthConfig()
    if args.out:
        cfg.out_dir = args.out
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n_customers is not None:
        cfg.n_customers = args.n_customers
    if args.days is not None:
        cfg.horizon_days = args.days

    data = generate(cfg)
    report = calibration_report(data["transactions"], cfg)
    print(report_markdown(report))
    if not args.no_save:
        save_dataset(data, cfg.out_dir)
        save_report(report, cfg.out_dir)
        print(f"\nSaved to {cfg.out_dir}")
    print(json.dumps(data["info"]["typology_counts"], indent=2))
    if not report["ok"]:
        raise SystemExit("Calibration FAILED — inspect report before using data.")


if __name__ == "__main__":
    main()
