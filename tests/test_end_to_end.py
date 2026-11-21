"""End-to-end smoke test: tiny generator -> local FL engine -> ensemble results."""
import json
import os
import sys

import numpy as np
import pytest

from indofraud_synth.config import SynthConfig
from indofraud_synth.generator import generate, save_dataset


def test_end_to_end_local(tmp_path):
    cfg = SynthConfig(seed=42, n_customers=1200, n_merchants=150, horizon_days=60,
                      start_date="2025-10-15")
    data = generate(cfg)
    data_dir = str(tmp_path / "data")
    save_dataset(data, data_dir)

    out_dir = str(tmp_path / "results")
    argv = ["run_fl", "--data", data_dir, "--engine", "local",
            "--models", "xgboost,catboost,mlp", "--partition", "profile",
            "--k", "3", "--rounds", "2", "--t0", "40", "--local-iters", "5",
            "--mlp-epochs", "1", "--tree-agg", "normalized", "--tree-budget", "120",
            "--train-cap", "20000", "--cent-iters", "60", "--cent-epochs", "2",
            "--out", out_dir]
    old = sys.argv
    sys.argv = argv
    try:
        from fl_experiments.run_fl import main
        main()
    finally:
        sys.argv = old

    with open(os.path.join(out_dir, "results.json")) as f:
        res = json.load(f)
    assert "ensemble" in res and "baselines" in res
    v = res["ensemble"]["variants"]
    assert 0.0 <= v["weighted_voting"]["pooled_ap"] <= 1.0
    assert 0.0 <= v["weighted_calibrated_per_silo"]["pooled_ap"] <= 1.0
    assert 0.0 <= v["weighted_calibrated_bayes"]["pooled_ap"] <= 1.0
    assert 0.0 <= v["weighted_calibrated_bayes_1pct"]["pooled_ap"] <= 1.0
    assert 0.0 <= v["weighted_calibrated_bayes_1pct"]["alert_rate"] <= 0.02
    assert np.isfinite(v["weighted_calibrated_bayes"]["alert_rate"])
    assert np.isfinite(v["weighted_calibrated_bayes"]["fpr"])
    assert np.isfinite(v["weighted_calibrated_bayes"]["bayes_savings_at_1pct"])
    assert np.isfinite(v["weighted_calibrated_bayes"]["ece"])
    assert np.isfinite(v["weighted_calibrated_bayes"]["brier"])
    assert np.isfinite(res["baselines"]["centralized"]["xgboost"]["bayes_savings_at_1pct"])
    assert abs(sum(res["ensemble"]["weights"].values()) - 1.0) < 1e-6
    assert abs(sum(res["ensemble"]["calibrated_weights"].values()) - 1.0) < 1e-6
    # Round 2 must genuinely add trees (not be frozen by foundation >= budget) and obey budget!
    xgb_rounds = res["fl"]["xgboost"]["rounds"]
    cat_rounds = res["fl"]["catboost"]["rounds"]
    assert xgb_rounds[1]["size"] > xgb_rounds[0]["size"]
    # Delta-tree upload: round 2 (5 new trees/client) uploads strictly fewer bytes than round 1
    assert xgb_rounds[1]["upload_bytes"] < xgb_rounds[0]["upload_bytes"]
    assert cat_rounds[1]["upload_bytes"] < cat_rounds[0]["upload_bytes"]
    assert res["global_model_sizes"]["xgboost"]["n_trees"] <= 120
    assert res["global_model_sizes"]["catboost"]["n_trees"] <= 120
    assert res["fl"]["xgboost"]["upload_bytes"] > 0
    # Calibration must activate on at least one silo via Platt/Isotonic
    assert len(res["ensemble"]["calibrated_silos"]["xgboost"]) >= 1
    # 2-model calibrated ablations and heuristic rules + centralized ensemble + full local_only metrics
    assert set(res["ensemble"]["ablations_2model"].keys()) == {
        "xgboost+catboost", "xgboost+mlp", "catboost+mlp"
    }
    assert np.isfinite(res["baselines"]["rules"]["pooled_ap"])
    assert np.isfinite(res["baselines"]["centralized"]["xgboost"]["macro_ap"])
    cent_ens = res["baselines"]["centralized"]["ensemble"]
    assert 0.0 <= cent_ens["pooled_ap"] <= 1.0
    assert np.isfinite(cent_ens["macro_ap"])
    assert np.isfinite(cent_ens["bayes_savings_at_1pct"])
    assert abs(sum(cent_ens["weights"].values()) - 1.0) < 1e-6
    loc_xgb = res["baselines"]["local_only"]["xgboost"]
    assert np.isfinite(loc_xgb["pooled_ap"])
    assert np.isfinite(loc_xgb["savings"])
    assert np.isfinite(loc_xgb["bayes_savings_at_1pct"])
    assert np.isfinite(loc_xgb["ece"])
    loc_ens = res["baselines"]["local_only"]["ensemble"]
    assert 0.0 <= loc_ens["pooled_ap"] <= 1.0
    assert np.isfinite(loc_ens["macro_ap"])
    assert np.isfinite(loc_ens["bayes_savings_at_1pct"])
    # In-memory inference latency measured for all 3 models + end-to-end ensemble
    assert set(res["latency_ms_p95"].keys()) == {"xgboost", "catboost", "mlp", "ensemble"}
    assert res["latency_ms_p95"]["xgboost"] < 100.0
    assert res["latency_ms_p95"]["ensemble"] < 300.0


def test_end_to_end_flower(tmp_path):
    """Flower simulation smoke test: xgboost + catboost + FedBN MLP via Ray."""
    cfg = SynthConfig(seed=42, n_customers=900, n_merchants=120, horizon_days=50,
                      start_date="2025-10-15")
    data = generate(cfg)
    data_dir = str(tmp_path / "data")
    save_dataset(data, data_dir)

    out_dir = str(tmp_path / "results_flower")
    argv = ["run_fl", "--data", data_dir, "--engine", "flower",
            "--models", "xgboost,catboost,mlp", "--partition", "profile",
            "--k", "3", "--rounds", "2", "--t0", "30", "--local-iters", "5",
            "--mlp-epochs", "1", "--mlp-agg", "fedbn",
            "--tree-agg", "normalized", "--tree-budget", "90",
            "--train-cap", "15000", "--no-baselines", "--out", out_dir]
    old = sys.argv
    sys.argv = argv
    try:
        from fl_experiments.run_fl import main
        main()
    finally:
        sys.argv = old

    with open(os.path.join(out_dir, "results.json")) as f:
        res = json.load(f)
    assert "ensemble" in res
    for mt in ("xgboost", "catboost", "mlp"):
        assert len(res["fl"][mt]["rounds"]) == 2
        assert res["fl"][mt]["upload_bytes"] > 0
    # Budget + Round-2 tree growth verified under Flower strategy too
    assert res["fl"]["xgboost"]["rounds"][1]["size"] > res["fl"]["xgboost"]["rounds"][0]["size"]
    assert res["global_model_sizes"]["xgboost"]["n_trees"] <= 90
    assert res["global_model_sizes"]["catboost"]["n_trees"] <= 90
    # FedBN persisted BN state files for all 3 clients
    bn_dir = os.path.join(out_dir, "_bn_state")
    assert os.path.isdir(bn_dir)
    assert len([f for f in os.listdir(bn_dir) if f.endswith("_bn.npz")]) == 3


def test_ieee_cis_and_sparkov_loaders_end_to_end(tmp_path):
    """Verify IEEE-CIS & Sparkov loaders, feature matrices, temporal splits, and FL run."""
    import pandas as pd
    from fl_experiments.data import load_ieee_cis, load_sparkov, temporal_split, make_clients
    from fl_experiments.features import build_matrix

    rng = np.random.default_rng(123)
    n = 600

    # --- Synthetic IEEE-CIS fixture (train_transaction.csv + train_identity.csv) ---
    ieee_dir = tmp_path / "ieee"
    ieee_dir.mkdir()
    # Inject a future-only category in the last 10% of rows to verify no vocab leak
    card4 = rng.choice(["visa", "mastercard", None], n).astype(object)
    card4[int(0.92 * n):] = "future_only_brand"
    tx = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1),
        "isFraud": (rng.random(n) < 0.08).astype(int),
        "TransactionDT": np.sort(rng.integers(86400, 86400 * 30, n)),
        "TransactionAmt": np.round(np.exp(rng.normal(4.5, 1.0, n)), 2),
        "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n),
        "card1": rng.integers(1000, 9999, n),
        "card4": card4,
        "C1": rng.poisson(2, n).astype(float),
        "D1": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 100, n)),
        "V1": rng.normal(0, 1, n),
    })
    ident = pd.DataFrame({
        "TransactionID": np.arange(1, n + 1, 2),   # partial identity coverage (realistic)
        "id_01": rng.normal(-5, 3, n // 2),
        "DeviceType": rng.choice(["desktop", "mobile"], n // 2),
    })
    tx.to_csv(ieee_dir / "train_transaction.csv", index=False)
    ident.to_csv(ieee_dir / "train_identity.csv", index=False)

    df_ieee = load_ieee_cis(str(ieee_dir))
    assert len(df_ieee) == n
    assert (df_ieee["ts"].diff().dropna() >= 0).all()
    # Future-only category must map to -1 (not leaked from test into train vocab)
    assert (df_ieee["card4"].iloc[int(0.92 * n):] == -1).all()
    X_ieee, cols_ieee = build_matrix(df_ieee, "ieee_cis")
    assert X_ieee.shape[0] == n and X_ieee.dtype == np.float32
    assert "is_fraud" not in cols_ieee and "TransactionID" not in cols_ieee and "ts" not in cols_ieee
    assert "DeviceType" in cols_ieee and "amount_log" in cols_ieee

    # Run quick FL smoke on IEEE-CIS
    out_ieee = str(tmp_path / "res_ieee")
    old = sys.argv
    sys.argv = ["run_fl", "--data", str(ieee_dir), "--dataset", "ieee_cis",
                "--engine", "local", "--models", "xgboost,mlp",
                "--partition", "dirichlet", "--k", "2", "--alpha", "5.0",
                "--rounds", "2", "--t0", "15", "--local-iters", "5",
                "--mlp-epochs", "1", "--tree-budget", "40",
                "--no-baselines", "--out", out_ieee]
    try:
        from fl_experiments.run_fl import main
        main()
    finally:
        sys.argv = old
    with open(os.path.join(out_ieee, "results.json")) as f:
        res_ieee = json.load(f)
    assert "ensemble" in res_ieee
    assert res_ieee["args"]["c_fp"] == 5.0

    # --- Synthetic Sparkov fixture (fraudTrain.csv + fraudTest.csv) ---
    sp_dir = tmp_path / "sparkov"
    sp_dir.mkdir()
    base_ts = 1577836800
    ts_all = np.sort(base_ts + rng.integers(0, 86400 * 60, n))
    dt_str = pd.to_datetime(ts_all, unit="s").strftime("%Y-%m-%d %H:%M:%S")
    sp = pd.DataFrame({
        "Unnamed: 0": np.arange(n),
        "trans_date_trans_time": dt_str,
        "cc_num": rng.integers(10**11, 10**12, n, dtype=np.int64),
        "merchant": rng.choice(["fraud_A", "fraud_B", "legit_C"], n),
        "category": rng.choice(["grocery_pos", "shopping_net", "gas_transport"], n),
        "amt": np.round(np.exp(rng.normal(3.8, 0.9, n)), 2),
        "first": "Alice", "last": "Smith", "gender": rng.choice(["F", "M"], n),
        "street": "123 Main", "city": "Austin", "state": rng.choice(["TX", "CA", "NY"], n),
        "zip": rng.integers(10000, 99999, n),
        "lat": rng.uniform(30, 42, n), "long": rng.uniform(-120, -75, n),
        "city_pop": rng.integers(500, 500000, n),
        "job": rng.choice(["Engineer", "Teacher", "Nurse"], n),
        "dob": "1985-06-15",
        "trans_num": [f"t{i}" for i in range(n)],
        "unix_time": ts_all,
        "merch_lat": rng.uniform(30, 42, n), "merch_long": rng.uniform(-120, -75, n),
        "is_fraud": (rng.random(n) < 0.08).astype(int),
    })
    sp.iloc[:400].to_csv(sp_dir / "fraudTrain.csv", index=False)
    sp.iloc[400:].to_csv(sp_dir / "fraudTest.csv", index=False)

    df_sp = load_sparkov(str(sp_dir))
    assert len(df_sp) == n
    assert (df_sp["ts"].diff().dropna() >= 0).all()
    X_sp, cols_sp = build_matrix(df_sp, "sparkov")
    assert X_sp.shape[0] == n and X_sp.dtype == np.float32
    assert "geo_dist_km" in cols_sp and "age_years" in cols_sp and "amount_log" in cols_sp
    for pii in ("cc_num", "first", "last", "trans_num", "merchant", "is_fraud", "ts"):
        assert pii not in cols_sp

    df_sp_s, _ = temporal_split(df_sp)
    clients_sp, pooled_sp = make_clients(df_sp_s, mode="dirichlet", k=2, alpha=5.0, seed=42)
    assert len(pooled_sp["train"]) == sum(len(c["train"]) for c in clients_sp.values())

    # Stratified random_split (replication anchor) must populate all splits with positives
    from fl_experiments.data import random_split
    df_sp_rand, rand_meta = random_split(df_sp, seed=42)
    assert rand_meta["mode"] == "random"
    assert set(df_sp_rand["split"].unique()) == {"train", "val", "test"}
    for sp_name in ("train", "val", "test"):
        assert df_sp_rand.loc[df_sp_rand["split"] == sp_name, "is_fraud"].sum() > 0


def test_cli_config_precedence_and_extreme_dirichlet():
    """Regression test: explicit CLI flags matching argparse defaults must NOT be
    overwritten by YAML `--config`, and extreme Dirichlet(0.1) must yield non-empty silos."""
    import pandas as pd
    from fl_experiments.run_fl import parse_args
    from indofraud_synth.silos import dirichlet_partition

    # default.yaml sets mlp_agg=fedbn, engine=flower, k=10, rounds=10, c_fp=7500
    a1 = parse_args([
        "--config", "configs/default.yaml",
        "--mlp-agg", "fedavg", "--engine", "local", "--k", "5",
    ])
    assert a1.mlp_agg == "fedavg"
    assert a1.engine == "local"
    assert a1.k == 5
    assert a1.rounds == 10           # inherited from default.yaml
    assert a1.patience == 5          # inherited from default.yaml
    assert a1.tree_budget == 2000    # inherited from default.yaml
    assert a1.c_fp == 7500.0

    # Switching --dataset to ulb with default.yaml must NOT leak IDR c_fp=7500
    a2 = parse_args(["--config", "configs/default.yaml", "--dataset", "ulb"])
    assert a2.c_fp == 5.0

    # Explicit --c-fp always overrides both YAML and dataset default
    a3 = parse_args(["--config", "configs/default.yaml", "--dataset", "ulb", "--c-fp", "12.5"])
    assert a3.c_fp == 12.5

    # Extreme Dirichlet(0.1) with pre-split df yields non-empty clients with train rows
    rng = np.random.default_rng(7)
    n = 2000
    df = pd.DataFrame({
        "ts": np.arange(n),
        "is_fraud": (rng.random(n) < 0.03).astype(np.int8),
        "split": np.where(np.arange(n) < 1400, "train",
                          np.where(np.arange(n) < 1700, "val", "test")),
    })
    parts = dirichlet_partition(df, k=6, alpha=0.1, seed=42)
    assert len(parts) == 6
    assert sum(len(p) for p in parts.values()) == n
    for p in parts.values():
        assert len(p) > 0 and int((p["split"] == "train").sum()) >= 2


def test_model_cache_exact_equivalence(tmp_path):
    """Cross-condition model cache must produce identical ensemble metrics (incl. FedBN)."""
    cfg = SynthConfig(seed=42, n_customers=1200, n_merchants=150, horizon_days=60,
                      start_date="2025-10-15")
    data = generate(cfg)
    data_dir = str(tmp_path / "data")
    save_dataset(data, data_dir)

    cache_dir = str(tmp_path / "_model_cache")
    out_cold = str(tmp_path / "cold")
    out_warm = str(tmp_path / "warm")
    base_argv = [
        "run_fl", "--data", data_dir, "--engine", "local",
        "--models", "xgboost,catboost,mlp", "--partition", "profile",
        "--k", "3", "--rounds", "2", "--t0", "30", "--local-iters", "5",
        "--mlp-epochs", "1", "--mlp-agg", "fedbn",
        "--tree-agg", "normalized", "--tree-budget", "90",
        "--train-cap", "15000", "--no-baselines",
        "--device", "cpu", "--threads", "2",
        "--model-cache-dir", cache_dir,
    ]
    from fl_experiments.run_fl import main
    old = sys.argv
    try:
        sys.argv = base_argv + ["--out", out_cold]
        main()
        # Cache directory must now contain 3 model entries
        assert len(os.listdir(cache_dir)) == 3

        sys.argv = base_argv + ["--out", out_warm]
        main()
    finally:
        sys.argv = old

    with open(os.path.join(out_cold, "results.json")) as f:
        r_cold = json.load(f)
    with open(os.path.join(out_warm, "results.json")) as f:
        r_warm = json.load(f)

    for vname in ("soft_voting", "weighted_voting", "weighted_calibrated_per_silo",
                  "weighted_calibrated_bayes", "weighted_calibrated_bayes_1pct"):
        assert np.isfinite(r_cold["ensemble"]["variants"][vname]["pooled_ap"])
        assert r_warm["ensemble"]["variants"][vname]["pooled_ap"] == pytest.approx(
            r_cold["ensemble"]["variants"][vname]["pooled_ap"], abs=1e-12
        )
        assert r_warm["ensemble"]["variants"][vname]["macro_ap"] == pytest.approx(
            r_cold["ensemble"]["variants"][vname]["macro_ap"], abs=1e-12
        )
        assert r_warm["ensemble"]["variants"][vname]["ece"] == pytest.approx(
            r_cold["ensemble"]["variants"][vname]["ece"], abs=1e-12
        )
    for mt in ("xgboost", "catboost", "mlp"):
        assert r_warm["fl"][mt]["best_val_ap"] == pytest.approx(r_cold["fl"][mt]["best_val_ap"], abs=1e-12)
