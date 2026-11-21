"""Multi-seed experiment suite runner + automatic Friedman-Nemenyi statistical report.

Example (quick smoke across 3 seeds):
  python -m fl_experiments.run_suite --config configs/quick.yaml \
      --seeds 1,2,3 --conditions baseline,normalized,normalized_fedbn \
      --partitions profile --out results/suite_quick

Example (full publication matrix):
  python -m fl_experiments.run_suite --config configs/default.yaml \
      --seeds 1,2,3,4,5 \
      --conditions baseline,normalized,normalized_fedprox,normalized_fedbn,normalized_size_fedbn \
      --partitions equal,profile,dirichlet:0.1,dirichlet:0.5,dirichlet:5.0 \
      --out results/suite_full --skip-existing
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys

from .stats import load_suite_dir, analyze_runs


CONDITION_PRESETS: dict[str, list[str]] = {
    "baseline": ["--tree-agg", "baseline", "--mlp-agg", "fedavg"],
    "normalized_unbudgeted": ["--tree-agg", "normalized", "--tree-weight", "uniform",
                              "--tree-budget", "0", "--mlp-agg", "fedavg"],
    "normalized": ["--tree-agg", "normalized", "--tree-weight", "uniform", "--mlp-agg", "fedavg"],
    "normalized_fedprox": ["--tree-agg", "normalized", "--tree-weight", "uniform", "--mlp-agg", "fedprox"],
    "normalized_fedbn": ["--tree-agg", "normalized", "--tree-weight", "uniform", "--mlp-agg", "fedbn"],
    "normalized_size_fedbn": ["--tree-agg", "normalized", "--tree-weight", "size", "--mlp-agg", "fedbn"],
}


def parse_partition_spec(spec: str) -> tuple[str, str, float | None]:
    """Parse 'profile', 'equal', or 'dirichlet:0.5' -> (tag, mode, alpha)."""
    spec = spec.strip()
    if ":" in spec:
        mode, a_str = spec.split(":", 1)
        alpha = float(a_str)
        tag = f"{mode}_a{a_str.replace('.', 'p')}"
        return tag, mode, alpha
    return spec, spec, None


def build_run_cmd(args, part_mode: str, part_alpha: float | None,
                  condition: str, seed: int, run_dir: str,
                  run_baselines: bool, model_cache_dir: str | None = None) -> list[str]:
    if condition not in CONDITION_PRESETS:
        raise ValueError(f"Unknown condition '{condition}'. Choose from {sorted(CONDITION_PRESETS)}")
    cmd = [sys.executable, "-m", "fl_experiments.run_fl"]
    if args.config:
        cmd += ["--config", args.config]
    if args.data:
        cmd += ["--data", args.data]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    if args.engine:
        cmd += ["--engine", args.engine]
    if args.models:
        cmd += ["--models", args.models]
    if args.k is not None:
        cmd += ["--k", str(args.k)]
    if args.rounds is not None:
        cmd += ["--rounds", str(args.rounds)]
    if getattr(args, "patience", None) is not None:
        cmd += ["--patience", str(args.patience)]
    if getattr(args, "split", None):
        cmd += ["--split", args.split]
    if getattr(args, "train_cap", None) is not None:
        cmd += ["--train-cap", str(args.train_cap)]
    if getattr(args, "device", None):
        cmd += ["--device", args.device]
    if getattr(args, "threads", None) is not None:
        cmd += ["--threads", str(args.threads)]
    if getattr(args, "mlp_batch_size", None) is not None:
        cmd += ["--mlp-batch-size", str(args.mlp_batch_size)]
    if model_cache_dir:
        cmd += ["--model-cache-dir", model_cache_dir]
    cmd += ["--partition", part_mode]
    if part_alpha is not None:
        cmd += ["--alpha", str(part_alpha)]
    cmd += CONDITION_PRESETS[condition]
    cmd += ["--seed", str(seed), "--out", run_dir]
    if not run_baselines:
        cmd += ["--no-baselines"]
    return cmd


def main():
    ap = argparse.ArgumentParser(description="Multi-seed FL experiment suite + statistics")
    ap.add_argument("--config", default="configs/quick.yaml")
    ap.add_argument("--data", default=None)
    ap.add_argument("--dataset", default=None, choices=["indofraud", "ulb", "ieee_cis", "sparkov"])
    ap.add_argument("--engine", default=None, choices=["local", "flower"])
    ap.add_argument("--models", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--train-cap", type=int, default=None)
    ap.add_argument("--split", default=None, choices=["temporal", "random"])
    ap.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=None,
                    help="Threads per worker process (default: auto; auto-divided if --workers > 1)")
    ap.add_argument("--mlp-batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of seeds to run concurrently (keep 1 when using a single GPU)")
    ap.add_argument("--no-model-cache", action="store_true",
                    help="Disable cross-condition FL model payload caching")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--conditions", default="baseline,normalized,normalized_fedbn")
    ap.add_argument("--partitions", default="profile",
                    help="Comma-separated: profile, equal, dirichlet:0.1, dirichlet:0.5, dirichlet:5.0")
    ap.add_argument("--metric", default="pooled_ap", help="Primary metric for Friedman-Nemenyi")
    ap.add_argument("--scope", default="primary", choices=["primary", "all"],
                    help="Statistical test scope: 1 canonical method/condition (primary) or all")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    ap.add_argument("--out", default="results/suite")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs whose results.json already exists")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = ap.parse_args()

    # If parallel seed workers are requested on CPU without explicit --threads,
    # divide CPU cores evenly across workers to prevent thread oversubscription.
    if args.workers > 1 and args.threads is None:
        args.threads = max(1, (os.cpu_count() or 2) // args.workers)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    part_specs = [parse_partition_spec(p) for p in args.partitions.split(",") if p.strip()]

    def _run_seed(part_dir: str, pmode: str, palpha: float | None, seed: int,
                  model_cache_dir: str | None):
        has_baselines_for_seed = False
        if args.skip_existing:
            for prev_cond in conditions:
                prev_path = os.path.join(part_dir, prev_cond, f"seed_{seed}", "results.json")
                if os.path.exists(prev_path):
                    try:
                        with open(prev_path) as f:
                            if "baselines" in json.load(f):
                                has_baselines_for_seed = True
                                break
                    except Exception:
                        pass
        for cond in conditions:
            run_dir = os.path.join(part_dir, cond, f"seed_{seed}")
            res_path = os.path.join(run_dir, "results.json")
            if args.skip_existing and os.path.exists(res_path):
                print(f"[skip] {res_path}")
                continue
            # Compute centralized + local-only baselines once per (partition, seed)
            need_base = not has_baselines_for_seed
            cmd = build_run_cmd(args, pmode, palpha, cond, seed, run_dir,
                                run_baselines=need_base, model_cache_dir=model_cache_dir)
            print("[run]", " ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
                if need_base:
                    has_baselines_for_seed = True

    for ptag, pmode, palpha in part_specs:
        part_dir = os.path.join(args.out, ptag)
        os.makedirs(part_dir, exist_ok=True)
        model_cache_dir = (None if args.no_model_cache
                           else os.path.join(part_dir, "_model_cache"))
        if args.workers > 1 and not args.dry_run and len(seeds) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(_run_seed, part_dir, pmode, palpha, seed, model_cache_dir)
                        for seed in seeds]
                for fut in as_completed(futs):
                    fut.result()
        else:
            for seed in seeds:
                _run_seed(part_dir, pmode, palpha, seed, model_cache_dir)

        if not args.dry_run:
            runs = load_suite_dir(part_dir)
            if runs:
                analyze_runs(runs, part_dir, primary_metric=args.metric,
                             alpha=args.alpha, scope=args.scope)
                print(f"[stats] Wrote summary.md / table.tex / cd_diagram.png -> {part_dir}")


if __name__ == "__main__":
    main()
