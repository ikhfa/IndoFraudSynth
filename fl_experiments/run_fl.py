"""Main experiment runner.

Engines:
  flower : real Flower simulation (flwr.simulation + Ray) — the publication path
  local  : in-process loop with IDENTICAL client/aggregation semantics —
           sandbox/CI-friendly fallback (no Ray), used for quick iteration

Example:
  python -m fl_experiments.run_fl --data data/quick --engine local \
      --models xgboost,catboost,mlp --partition profile --k 5 --rounds 3 \
      --tree-agg normalized --tree-budget 400 --out results/quick_normalized
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
import time

import numpy as np
import yaml

from .aggregation import (
    merge_trees, tree_count, merge_mlp, compute_r1_t0, compute_round_local_iters,
)
from .baselines import run_baselines
from .client import FraudClient, ClientCfg, make_flower_client
from .data import (
    load_indofraud, load_ulb, load_ieee_cis, load_sparkov,
    temporal_split, random_split, make_clients,
)
from .ensemble import evaluate_ensemble
from .features import build_matrix, MLPPreprocessor
from .metrics import latency_p95_ms, default_c_fp
from . import models as M


def run_local_engine(cores, model_type, rounds, normalize, budget,
                     size_weighted=False, patience: int = 0, verbose=True):
    payload = None
    n_prev = foundation = 0
    best_ap, best_payload, best_round = -1.0, None, 0
    stale = 0
    stopped_early = False
    logs = []
    up_bytes = down_bytes = 0
    use_plan = model_type != "mlp" and normalize and budget is not None
    r1_t0 = (compute_r1_t0(cores[0].cfg.t0, len(cores), rounds, cores[0].cfg.local_iters, budget)
             if use_plan else None)
    eff_local_iters = (
        compute_round_local_iters(cores[0].cfg.t0, cores[0].cfg.local_iters,
                                  len(cores), rounds, budget)
        if use_plan else None
    )
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        fits = [
            c.fit(
                payload, r,
                t0_override=(r1_t0 if r == 1 else None),
                local_iters_override=(eff_local_iters if r > 1 else None),
                n_prev=(n_prev if model_type != "mlp" else 0),
            )
            for c in cores
        ]
        payloads = [f[0] for f in fits]
        weights = [f[1] for f in fits]
        if model_type == "mlp":
            payload = merge_mlp(payloads, weights)
            size_metric = int(sum(a.nbytes for a in payload))
        else:
            c_weights = weights if (normalize and size_weighted) else None
            payload = merge_trees(model_type, payloads, global_payload=payload,
                                  normalize=normalize, n_prev=n_prev, budget=budget,
                                  foundation=foundation, client_weights=c_weights,
                                  is_delta=(n_prev > 0))
            n_prev = tree_count(model_type, payload)
            if r == 1:
                foundation = n_prev
            size_metric = n_prev
        up = sum(f[2]["payload_bytes"] for f in fits)
        down = (len(payload) if model_type != "mlp"
                else int(sum(a.nbytes for a in payload))) * len(cores)
        up_bytes += up; down_bytes += down
        evals = [c.evaluate(payload, server_round=r) for c in cores]
        aps = []
        valid_aps = []
        for e in evals:
            a = float(e[2].get("ap", -1.0))
            has_pos = float(e[2].get("has_pos", 1.0)) > 0.5
            if has_pos and np.isfinite(a) and a >= 0.0:
                aps.append(a)
                valid_aps.append(a)
            else:
                aps.append(None)
        macro = float(np.mean(valid_aps)) if valid_aps else 0.0
        if macro > best_ap:
            best_ap, best_round = macro, r
            best_payload = ([a.copy() for a in payload] if model_type == "mlp"
                            else payload)
            stale = 0
        else:
            stale += 1
        logs.append({"round": r, "macro_val_ap": macro, "per_client_ap": aps,
                     "size": size_metric, "upload_bytes": up, "broadcast_bytes": down,
                     "seconds": round(time.perf_counter() - t0, 2)})
        if verbose:
            print(f"  [{model_type}] round {r}: macro_val_AP={macro:.4f} "
                  f"size={size_metric} ({logs[-1]['seconds']}s)")
        if patience > 0 and stale >= patience:
            stopped_early = True
            if verbose:
                print(f"  [{model_type}] early stopping after round {r} "
                      f"(no improvement for {patience} rounds)")
            break
    if model_type == "mlp" and best_round > 0:
        for c in cores:
            c.restore_bn_round(best_round)
    final_round = logs[-1]["round"] if logs else 0
    final_val_ap = logs[-1]["macro_val_ap"] if logs else float("nan")
    return best_payload, {"best_round": best_round, "best_val_ap": best_ap,
                          "final_round": final_round, "final_val_ap": final_val_ap,
                          "stopped_early": stopped_early,
                          "rounds": logs, "upload_bytes": up_bytes,
                          "broadcast_bytes": down_bytes}


def _run_flower_inprocess(client_fn, num_clients, rounds, strategy):
    """Run the real Flower Server + Strategy + ClientProxy protocol without Ray."""
    import flwr as fl
    from flwr.server import Server, SimpleClientManager
    from flwr.server.client_proxy import ClientProxy

    class _LocalProxy(ClientProxy):
        def __init__(self, cid: str):
            super().__init__(cid)

        def get_properties(self, ins, timeout, group_id):
            return client_fn(self.cid).get_properties(ins)

        def get_parameters(self, ins, timeout, group_id):
            return client_fn(self.cid).get_parameters(ins)

        def fit(self, ins, timeout, group_id):
            return client_fn(self.cid).fit(ins)

        def evaluate(self, ins, timeout, group_id):
            return client_fn(self.cid).evaluate(ins)

        def reconnect(self, ins, timeout, group_id):
            return fl.common.DisconnectRes(reason="")

    cm = SimpleClientManager()
    for i in range(num_clients):
        cm.register(_LocalProxy(str(i)))
    server = Server(client_manager=cm, strategy=strategy)
    server.set_max_workers(1)
    server.fit(num_rounds=rounds, timeout=None)


def run_flower_engine(cores, model_type, rounds, normalize, budget,
                      size_weighted=False, patience: int = 0, verbose=True):
    import importlib.util
    import flwr as fl
    from .strategy import TreeBaggingStrategy, MLPStrategy

    use_plan = model_type != "mlp" and normalize and budget is not None
    r1_t0 = (compute_r1_t0(cores[0].cfg.t0, len(cores), rounds, cores[0].cfg.local_iters, budget)
             if use_plan else None)
    eff_local_iters = (
        compute_round_local_iters(cores[0].cfg.t0, cores[0].cfg.local_iters,
                                  len(cores), rounds, budget)
        if use_plan else None
    )
    if model_type == "mlp":
        strategy = MLPStrategy(initial_arrays=cores[0]._initial_mlp(),
                               min_clients=len(cores), patience=patience)
    else:
        strategy = TreeBaggingStrategy(model_kind=model_type, normalize=normalize,
                                       budget=budget, min_clients=len(cores),
                                       r1_t0=r1_t0, eff_local_iters=eff_local_iters,
                                       size_weighted=size_weighted, patience=patience)

    def client_fn(cid: str):
        return make_flower_client(cores[int(cid)]).to_client()

    has_ray = importlib.util.find_spec("ray") is not None
    t0 = time.perf_counter()
    if has_ray:
        from flwr.simulation import start_simulation
        ray_tmp = os.path.join(tempfile.gettempdir(), "flwr_ray")
        os.makedirs(ray_tmp, exist_ok=True)
        ray_args = {
            "num_cpus": min(4, os.cpu_count() or 2),
            "include_dashboard": False,
            "_temp_dir": ray_tmp,
            "object_store_memory": 300 * 1024 * 1024,
            "log_to_driver": verbose,
        }
        start_simulation(
            client_fn=client_fn, num_clients=len(cores),
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args=ray_args,
        )
    else:
        _run_flower_inprocess(client_fn, len(cores), rounds, strategy)

    if model_type == "mlp" and strategy.best_round > 0:
        for c in cores:
            c.restore_bn_round(strategy.best_round)

    info = {"best_round": strategy.best_round, "best_val_ap": strategy.best_ap,
            "final_round": strategy.final_round, "final_val_ap": strategy.final_val_ap,
            "stopped_early": strategy.stopped_early,
            "rounds": strategy.round_logs,
            "upload_bytes": strategy.total_client_bytes,
            "broadcast_bytes": strategy.total_broadcast_bytes,
            "flower_backend": "ray" if has_ray else "inprocess_server",
            "seconds": round(time.perf_counter() - t0, 2)}
    return strategy.best_payload, info


def _pr_curve_plot(clients, payloads, ensemble_res, pooled, dataset, out_path,
                   plot_scores: dict | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    first_val = next(iter(clients.values()))
    is_nested = isinstance(first_val, dict)
    ref_clients = first_val if is_nested else clients
    silo_order = list(ref_clients.keys())
    y_order = np.concatenate([ref_clients[s].y["test"] for s in silo_order])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    per_model_scores = {}
    for mt, payload in payloads.items():
        if payload is None:
            continue
        cmap = clients[mt] if is_nested and mt in clients else clients
        s = np.concatenate([cmap[name].predict(payload, "test") for name in silo_order])
        per_model_scores[mt] = s
        p, r, _ = precision_recall_curve(y_order, s)
        ax.plot(r, p, lw=1.4, alpha=0.85, label=f"FL {mt}")

    if ensemble_res and len(per_model_scores) >= 2:
        w = ensemble_res.get("weights", {})
        if w and all(m in per_model_scores for m in w):
            ens_s = sum(w[m] * per_model_scores[m] for m in w)
            p, r, _ = precision_recall_curve(y_order, ens_s)
            ax.plot(r, p, lw=1.8, color="#444444", linestyle="--",
                    label="FL Weighted Ensemble (uncalibrated)")
        if plot_scores and "calibrated" in plot_scores:
            cal_s = np.concatenate([plot_scores["calibrated"][name] for name in silo_order])
            p, r, _ = precision_recall_curve(y_order, cal_s)
            ax.plot(r, p, lw=2.3, color="black", linestyle="-",
                    label="FL Proposed Calibrated Ensemble")

    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Test Precision–Recall: Federated Models & Ensemble")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    ap_ = argparse.ArgumentParser(description="Federated ensemble fraud detection")
    ap_.add_argument("--config", default=None)
    ap_.add_argument("--data", default="data/quick")
    ap_.add_argument("--dataset", default="indofraud",
                     choices=["indofraud", "ulb", "ieee_cis", "sparkov"])
    ap_.add_argument("--engine", default="local", choices=["local", "flower"])
    ap_.add_argument("--models", default="xgboost,catboost,mlp")
    ap_.add_argument("--partition", default="profile",
                     choices=["profile", "equal", "dirichlet"])
    ap_.add_argument("--split", default="temporal", choices=["temporal", "random"],
                     help="temporal (default, leak-free) or random (source-paper replication only)")
    ap_.add_argument("--k", type=int, default=5)
    ap_.add_argument("--alpha", type=float, default=0.5)
    ap_.add_argument("--rounds", type=int, default=3)
    ap_.add_argument("--patience", type=int, default=0,
                     help="Federated early stopping patience on macro val AP (0 = run all rounds)")
    ap_.add_argument("--t0", type=int, default=120)
    ap_.add_argument("--local-iters", type=int, default=15)
    ap_.add_argument("--mlp-epochs", type=int, default=2)
    ap_.add_argument("--mlp-agg", default="fedavg", choices=["fedavg", "fedprox", "fedbn"])
    ap_.add_argument("--prox-mu", type=float, default=0.01)
    ap_.add_argument("--tree-agg", default="normalized", choices=["baseline", "normalized"])
    ap_.add_argument("--tree-weight", default="uniform", choices=["uniform", "size"])
    ap_.add_argument("--tree-budget", type=int, default=400)
    ap_.add_argument("--train-cap", type=int, default=120_000)
    ap_.add_argument("--cent-iters", type=int, default=300)
    ap_.add_argument("--cent-epochs", type=int, default=6)
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--c-fp", type=float, default=None,
                     help="False-positive cost in dataset currency (defaults per dataset)")
    ap_.add_argument("--out", default="results/run")
    ap_.add_argument("--no-baselines", action="store_true")
    ap_.add_argument("--no-ensemble", action="store_true")
    return ap_


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Two-pass CLI parsing: hardcoded defaults < YAML `--config` < explicit CLI flags.

    Prevents YAML values (e.g. `mlp_agg: fedbn` in default.yaml) from silently
    overwriting explicit CLI flags that happen to match argparse defaults
    (e.g. `--mlp-agg fedavg` or `--engine local`), and avoids leaking an IDR
    `c_fp` from a YAML config onto a non-IndoFraud `--dataset`.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--dataset", default=None)
    pre.add_argument("--c-fp", type=float, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    ap_ = build_arg_parser()
    if pre_args.config:
        with open(pre_args.config) as f:
            exp = (yaml.safe_load(f) or {}).get("experiment", {})
        valid_dests = {a.dest for a in ap_._actions}
        yaml_defaults = {
            k.replace("-", "_"): v for k, v in exp.items()
            if k.replace("-", "_") in valid_dests
        }
        yaml_dataset = yaml_defaults.get("dataset", "indofraud")
        if (pre_args.dataset is not None
                and pre_args.dataset != yaml_dataset
                and pre_args.c_fp is None):
            yaml_defaults.pop("c_fp", None)
        ap_.set_defaults(**yaml_defaults)

    args = ap_.parse_args(argv)
    if args.c_fp is None:
        args.c_fp = default_c_fp(args.dataset)
    return args


def main():
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    model_types = [m.strip() for m in args.models.split(",")]
    normalize = args.tree_agg == "normalized"
    budget = args.tree_budget if (normalize and args.tree_budget and args.tree_budget > 0) else None

    print(f"Loading {args.data} ({args.dataset}) ...")
    loaders = {
        "indofraud": load_indofraud,
        "ulb": load_ulb,
        "ieee_cis": load_ieee_cis,
        "sparkov": load_sparkov,
    }
    df = loaders[args.dataset](args.data)
    if args.split == "random":
        df, cuts = random_split(df, seed=args.seed)
    else:
        df, cuts = temporal_split(df)
    if args.dataset != "indofraud" and args.partition == "profile":
        args.partition = "dirichlet"
    clients_df, pooled = make_clients(df, args.partition, args.k, args.alpha,
                                      args.seed, args.train_cap)
    del df
    print(f"Clients: { {k: len(v['train']) for k, v in clients_df.items()} }")

    # FL-safe global MLP standardization & pos_weight from per-client scalar sufficient
    # statistics ((count, sum, sum_sq) and (n_pos, n_total)) — zero raw rows shared.
    fed_pre = None
    fed_mlp_pw = None
    if "mlp" in model_types:
        stats = [MLPPreprocessor.local_stats(build_matrix(fr["train"], args.dataset)[0])
                 for fr in clients_df.values()]
        fed_pre = MLPPreprocessor().fit_from_stats(stats)
        tot_pos = sum(int(fr["train"].is_fraud.sum()) for fr in clients_df.values())
        tot_n = sum(len(fr["train"]) for fr in clients_df.values())
        fed_mlp_pw = M.compute_mlp_pos_weight(tot_pos, tot_n)

    bn_dir = (os.path.join(args.out, "_bn_state")
              if (args.mlp_agg == "fedbn" and args.engine == "flower") else None)
    if bn_dir and os.path.isdir(bn_dir):
        shutil.rmtree(bn_dir, ignore_errors=True)
    ccfg = ClientCfg(t0=args.t0, local_iters=args.local_iters, mlp_epochs=args.mlp_epochs,
                     mlp_pos_weight=fed_mlp_pw, prox_mu=args.prox_mu, mlp_agg=args.mlp_agg,
                     seed=args.seed, dataset=args.dataset, bn_state_dir=bn_dir)
    # Build unscaled feature matrices once per client and share across models (halves RAM)
    shared_by_silo = {}
    for name, frames in clients_df.items():
        sx, sy, sa = {}, {}, {}
        for sp in ("train", "val", "test"):
            df_sp = frames[sp]
            sx[sp], _ = build_matrix(df_sp, args.dataset)
            sy[sp] = df_sp.is_fraud.to_numpy(np.int8)
            if "amount" in df_sp:
                sa[sp] = df_sp["amount"].to_numpy(np.float64)
            elif "Amount" in df_sp:
                sa[sp] = df_sp["Amount"].to_numpy(np.float64)
            else:
                sa[sp] = np.zeros(len(df_sp))
        shared_by_silo[name] = (sx, sy, sa)

    cores = {mt: [FraudClient(name, mt, frames, ccfg,
                              pre=(fed_pre if mt == "mlp" else None),
                              shared_raw=shared_by_silo[name])
                  for name, frames in clients_df.items()] for mt in model_types}
    del shared_by_silo

    engine = run_flower_engine if args.engine == "flower" else run_local_engine
    size_weighted = args.tree_weight == "size"
    payloads, fl_info = {}, {}
    for mt in model_types:
        print(f"Training federated {mt} ({args.engine}, {args.tree_agg}) ...")
        payload, info = engine(cores[mt], mt, args.rounds, normalize, budget,
                               size_weighted=size_weighted, patience=args.patience)
        payloads[mt] = payload
        fl_info[mt] = info

    results = {"args": vars(args), "temporal_cuts": cuts,
               "silo_stats": {name: {"n_train": len(fr["train"]),
                                     "n_test": len(fr["test"]),
                                     "fraud_rate_test": float(fr["test"].is_fraud.mean())}
                              for name, fr in clients_df.items()},
               "fl": fl_info}

    cores_by_mt = {mt: {c.name: c for c in lst} for mt, lst in cores.items()}

    if not args.no_baselines:
        print("Running centralized + local-only baselines ...")
        results["baselines"] = run_baselines(
            pooled, cores_by_mt, model_types, dataset=args.dataset,
            cent_iters=args.cent_iters, cent_epochs=args.cent_epochs,
            local_t0=args.t0, c_fp=args.c_fp, seed=args.seed,
            clients_df=clients_df)

    plot_scores = {}
    if not args.no_ensemble and len(payloads) >= 2:
        print("Evaluating ensembles ...")
        results["ensemble"] = evaluate_ensemble(cores_by_mt, payloads,
                                                c_fp=args.c_fp, seed=args.seed,
                                                out_plot_scores=plot_scores)

    # latency (in-memory loaded models — excludes JSON/cbm deserialization!) + model size
    sizes = {}
    for mt, p in payloads.items():
        if p is None:
            continue
        if mt == "mlp":
            sizes[mt] = {"bytes": int(sum(a.nbytes for a in p))}
        else:
            sizes[mt] = {"n_trees": tree_count(mt, p), "bytes": len(p)}
    results["global_model_sizes"] = sizes
    lat = {}
    loaded_predictors = []
    if "xgboost" in payloads and payloads["xgboost"] is not None:
        c0 = cores["xgboost"][0]
        bst_loaded = M.bytes_to_xgb(payloads["xgboost"])
        xgb_fn = lambda X, _b=bst_loaded: M.xgb_predict_fast(_b, X)
        lat["xgboost"] = latency_p95_ms(xgb_fn, c0.X["test"], n_iter=200, warmup=15)
        loaded_predictors.append((xgb_fn, c0.X["test"]))
    if "catboost" in payloads and payloads["catboost"] is not None:
        c0 = cores["catboost"][0]
        cat_loaded = M.bytes_to_cat(payloads["catboost"])
        cat_fn = lambda X, _m=cat_loaded: M.cat_predict_loaded(_m, X)
        lat["catboost"] = latency_p95_ms(cat_fn, c0.X["test"], n_iter=200, warmup=15)
        loaded_predictors.append((cat_fn, c0.X["test"]))
    if "mlp" in payloads and payloads["mlp"] is not None:
        c0 = cores["mlp"][0]
        mlp_loaded = c0._mlp_model_from(payloads["mlp"]).eval()
        mlp_shift = c0._mlp_logit_shift()
        mlp_fn = lambda X, _m=mlp_loaded, _s=mlp_shift: M.mlp_predict(_m, X, logit_shift=_s)
        lat["mlp"] = latency_p95_ms(mlp_fn, c0.X["test"], n_iter=200, warmup=15)
        loaded_predictors.append((mlp_fn, c0.X["test"]))
    if len(loaded_predictors) >= 2:
        n_rows = min(len(X) for _, X in loaded_predictors)
        if n_rows > 0:
            idx_arr = np.arange(n_rows)
            def _ens_predict(idx_slice):
                i = int(idx_slice[0])
                total = 0.0
                for fn, Xm in loaded_predictors:
                    total += float(fn(Xm[i:i + 1])[0])
                return total
            lat["ensemble"] = latency_p95_ms(_ens_predict, idx_arr, n_iter=200, warmup=15)
    if lat:
        results["latency_ms_p95"] = lat

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    try:
        _pr_curve_plot(cores_by_mt, payloads, results.get("ensemble"),
                       pooled, args.dataset, os.path.join(args.out, "pr_curve.png"),
                       plot_scores=plot_scores)
    except Exception as e:  # plotting must never kill a run
        print(f"(plot skipped: {e})")

    # console summary
    print("\n=== SUMMARY ===")
    for mt in model_types:
        info = fl_info[mt]
        line = f"FL {mt:9s} best_round={info['best_round']} val_AP={info['best_val_ap']:.4f}"
        if mt in sizes:
            line += f" size={sizes[mt]}"
        print(line)
    if "ensemble" in results:
        for name, v in results["ensemble"]["variants"].items():
            print(f"{name:32s} test_AP={v['pooled_ap']:.4f} macro_AP={v['macro_ap']:.4f} "
                  f"savings={v['savings']:.3f}")
    if "baselines" in results:
        for mt, v in results["baselines"]["centralized"].items():
            print(f"centralized {mt:20s} test_AP={v['pooled_ap']:.4f} savings={v['savings']:.3f}")
    print(f"\nResults -> {args.out}/results.json")


if __name__ == "__main__":
    main()
