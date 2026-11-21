import numpy as np
import pytest
import xgboost as xgb

from fl_experiments import models as M
from fl_experiments.aggregation import (
    merge_xgb, merge_cat, fedavg, tree_count, compute_r1_t0,
    compute_round_local_iters, _xgb_tree_leaf_rms, _prune_by_magnitude,
    extract_xgb_delta, extract_cat_delta,
)
from fl_experiments.client import FraudClient, ClientCfg
from fl_experiments.features import MLPPreprocessor


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=600) > 0.8).astype(int)
    return X, y


def _margins(payload, X):
    return M.bytes_to_xgb(payload).predict(xgb.DMatrix(X), output_margin=True)


def test_xgb_merge_additivity(data):
    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 25))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 25))
    merged = merge_xgb([b1, b2], normalize=False)
    assert tree_count("xgboost", merged) == 50
    m = _margins(merged, X)
    expect = _margins(b1, X) + _margins(b2, X)   # base_score=0.5 -> logit 0
    assert np.allclose(m, expect, atol=1e-4)


def test_xgb_merge_normalized(data):
    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 25))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 25))
    merged = merge_xgb([b1, b2], normalize=True)
    m = _margins(merged, X)
    expect = (_margins(b1, X) + _margins(b2, X)) / 2
    assert np.allclose(m, expect, atol=1e-4)


def test_xgb_append_and_budget(data):
    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 20))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 20))
    g = merge_xgb([b1, b2], normalize=True)
    n_prev = tree_count("xgboost", g)          # 40
    gb = M.bytes_to_xgb(g)
    c1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 10, xgb_model=gb))
    c2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 10, xgb_model=gb))
    assert tree_count("xgboost", c1) == n_prev + 10
    appended = merge_xgb([c1, c2], global_payload=g, normalize=True, n_prev=n_prev)
    assert tree_count("xgboost", appended) == n_prev + 20
    capped = merge_xgb([c1, c2], global_payload=g, normalize=True, n_prev=n_prev,
                       budget=n_prev + 10, foundation=n_prev)
    assert tree_count("xgboost", capped) == n_prev + 10
    # capped model must still predict valid probabilities
    p = M.xgb_predict(capped, X)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_cat_merge_additivity(data):
    X, y = data
    m1 = M.cat_to_bytes(M.train_cat(X[:300], y[:300], 20))
    m2 = M.cat_to_bytes(M.train_cat(X[300:], y[300:], 20))
    merged = merge_cat([m1, m2], normalize=False)
    assert tree_count("catboost", merged) == 40
    r = M.bytes_to_cat(merged).predict(X, prediction_type="RawFormulaVal")
    r1 = M.bytes_to_cat(m1).predict(X, prediction_type="RawFormulaVal")
    r2 = M.bytes_to_cat(m2).predict(X, prediction_type="RawFormulaVal")
    assert np.allclose(r, r1 + r2, atol=1e-3)
    rn = M.bytes_to_cat(merge_cat([m1, m2], normalize=True)).predict(
        X, prediction_type="RawFormulaVal")
    assert np.allclose(rn, (r1 + r2) / 2, atol=1e-3)


def test_cat_append_and_budget(data):
    X, y = data
    import os, tempfile
    m1 = M.cat_to_bytes(M.train_cat(X[:300], y[:300], 15))
    m2 = M.cat_to_bytes(M.train_cat(X[300:], y[300:], 15))
    g = merge_cat([m1, m2], normalize=True)
    n_prev = tree_count("catboost", g)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "g.cbm")
        with open(p, "wb") as f:
            f.write(g)
        c1 = M.cat_to_bytes(M.train_cat(X[:300], y[:300], 8, init_model_path=p))
        c2 = M.cat_to_bytes(M.train_cat(X[300:], y[300:], 8, init_model_path=p))
    assert tree_count("catboost", c1) == n_prev + 8
    appended = merge_cat([c1, c2], global_payload=g, normalize=True, n_prev=n_prev)
    assert tree_count("catboost", appended) == n_prev + 16
    capped = merge_cat([c1, c2], global_payload=g, normalize=True, n_prev=n_prev,
                       budget=n_prev + 8, foundation=n_prev)
    assert tree_count("catboost", capped) == n_prev + 8
    pr = M.cat_predict(capped, X)
    assert np.isfinite(pr).all()


def test_fedavg():
    a = [np.array([1.0, 2.0]), np.array([3, 4], dtype=np.int64)]
    b = [np.array([3.0, 6.0]), np.array([5, 6], dtype=np.int64)]
    out = fedavg([a, b], weights=[1, 3])
    assert np.allclose(out[0], [2.5, 5.0])
    assert out[1].dtype == np.int64


def test_xgb_round1_budget_and_foundation_guard(data):
    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 25))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 25))
    # Round 1 must enforce budget via valid per-client prefixes
    r1 = merge_xgb([b1, b2], normalize=True, budget=30)
    assert tree_count("xgboost", r1) == 30
    assert compute_r1_t0(120, k=5, rounds=3, local_iters=15, budget=400) == 50

    # Even if caller passes foundation >= budget, new trees must NOT be completely dropped
    gb = M.bytes_to_xgb(r1)
    c1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 8, xgb_model=gb))
    c2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 8, xgb_model=gb))
    r2 = merge_xgb([c1, c2], global_payload=r1, normalize=True, n_prev=30,
                   budget=30, foundation=30)
    assert tree_count("xgboost", r2) == 30
    # Predictions must change because newly trained round-2 trees were incorporated
    assert not np.allclose(_margins(r1, X), _margins(r2, X))


def test_tree_size_weighted_merge(data):
    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 15))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 15))
    merged = merge_xgb([b1, b2], normalize=True, client_weights=[1, 3])
    expect = 0.25 * _margins(b1, X) + 0.75 * _margins(b2, X)
    assert np.allclose(_margins(merged, X), expect, atol=1e-4)

    m1 = M.cat_to_bytes(M.train_cat(X[:300], y[:300], 15))
    m2 = M.cat_to_bytes(M.train_cat(X[300:], y[300:], 15))
    mc = merge_cat([m1, m2], normalize=True, client_weights=[1, 3])
    r = M.bytes_to_cat(mc).predict(X, prediction_type="RawFormulaVal")
    r1 = M.bytes_to_cat(m1).predict(X, prediction_type="RawFormulaVal")
    r2 = M.bytes_to_cat(m2).predict(X, prediction_type="RawFormulaVal")
    assert np.allclose(r, 0.25 * r1 + 0.75 * r2, atol=1e-3)


def test_fedprox_regularizes_all_layers(data):
    """Regression test: FedProx must regularize fc2 and fc3 despite BN buffers in state_dict."""
    import torch
    X, y = data
    torch.manual_seed(0)
    base_model = M.make_mlp(X.shape[1])
    ref_arrays = M.mlp_to_arrays(base_model)
    keys = M.mlp_keys(base_model)
    ref_map = dict(zip(keys, ref_arrays))

    m_free = M.make_mlp(X.shape[1]); M.mlp_from_arrays(m_free, ref_arrays)
    m_prox = M.make_mlp(X.shape[1]); M.mlp_from_arrays(m_prox, ref_arrays)
    M.train_mlp(m_free, X, y, epochs=5, lr=1e-2, prox_mu=0.0, ref_arrays=None, seed=1)
    M.train_mlp(m_prox, X, y, epochs=5, lr=1e-2, prox_mu=50.0, ref_arrays=ref_arrays, seed=1)

    free_map = dict(zip(keys, M.mlp_to_arrays(m_free)))
    prox_map = dict(zip(keys, M.mlp_to_arrays(m_prox)))
    for layer in ("fc1.weight", "fc2.weight", "fc3.weight"):
        d_free = np.linalg.norm(free_map[layer] - ref_map[layer])
        d_prox = np.linalg.norm(prox_map[layer] - ref_map[layer])
        assert d_prox < d_free * 0.5, f"FedProx did not pull {layer}: {d_prox} vs {d_free}"


def test_fedbn_persists_across_instances(tmp_path):
    """FedBN local BN statistics must persist across separate FraudClient instances (Ray/Flower)."""
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({f"V{i}": rng.normal(loc=3.0, scale=2.0, size=n) for i in range(1, 5)})
    df["Amount_log"] = rng.uniform(1, 5, n)
    df["hour"] = rng.integers(0, 24, n)
    df["Amount"] = np.expm1(df["Amount_log"])
    df["is_fraud"] = (rng.random(n) < 0.1).astype(np.int8)
    frames = {"train": df, "val": df, "test": df}

    bn_dir = str(tmp_path / "bn")
    cfg = ClientCfg(mlp_epochs=2, mlp_agg="fedbn", seed=42, dataset="ulb", bn_state_dir=bn_dir)
    c1 = FraudClient("silo_a", "mlp", frames, cfg)
    init = c1._initial_mlp()
    trained, _, _ = c1.fit(init, server_round=1)

    # Fresh client instance (as created in a separate Ray worker or driver after simulation)
    c2 = FraudClient("silo_a", "mlp", frames, cfg)
    zeroed = [np.zeros_like(a) for a in trained]
    restored_model = c2._mlp_model_from(zeroed)
    restored = M.mlp_to_arrays(restored_model)
    for idx in M.bn_indices(c2.keys):
        assert np.allclose(restored[idx], trained[idx])


def test_federated_scaler_matches_centralized(data):
    X, _ = data
    cent = MLPPreprocessor().fit(X)
    stats = [MLPPreprocessor.local_stats(chunk) for chunk in (X[:100], X[100:450], X[450:])]
    fed = MLPPreprocessor().fit_from_stats(stats)
    assert np.allclose(fed.mean_, cent.mean_, atol=1e-5)
    assert np.allclose(fed.std_, cent.std_, atol=1e-5)
    assert np.allclose(fed.transform(X), cent.transform(X), atol=1e-5)


def test_centralized_early_stopping(data):
    X, y = data
    Xtr, ytr = X[:400], y[:400]
    Xva, yva = X[400:], y[400:]

    bst = M.train_xgb(Xtr, ytr, 200, seed=0, eval_set=(Xva, yva), early_stopping_rounds=10)
    assert M.xgb_num_trees(M.xgb_to_bytes(bst)) <= 200
    assert M.xgb_num_trees(M.xgb_to_bytes(bst)) < 200

    cat = M.train_cat(Xtr, ytr, 200, seed=0, eval_set=(Xva, yva), early_stopping_rounds=10)
    assert 1 <= cat.tree_count_ < 200

    import torch
    torch.manual_seed(0)
    mlp = M.make_mlp(X.shape[1])
    mlp = M.train_mlp(mlp, Xtr, ytr, epochs=25, lr=1e-2, seed=0,
                      eval_set=(Xva, yva), patience=3)
    assert 1 <= mlp.best_epoch_ <= 25


def test_fedbn_restores_best_round_bn_not_last_round(tmp_path):
    """Regression test: when best_round < final round, FedBN must pair best_round
    linear weights with best_round BN statistics (both in-memory and from disk)."""
    import pandas as pd
    rng = np.random.default_rng(19)
    n = 160
    df = pd.DataFrame({f"V{i}": rng.normal(loc=2.0, scale=1.5, size=n) for i in range(1, 5)})
    df["Amount_log"] = rng.uniform(1, 5, n)
    df["hour"] = rng.integers(0, 24, n)
    df["Amount"] = np.expm1(df["Amount_log"])
    df["is_fraud"] = (rng.random(n) < 0.15).astype(np.int8)
    frames = {"train": df, "val": df, "test": df}

    bn_dir = str(tmp_path / "bn_rounds")
    cfg = ClientCfg(mlp_epochs=2, mlp_agg="fedbn", seed=7, dataset="ulb", bn_state_dir=bn_dir)
    c = FraudClient("silo_x", "mlp", frames, cfg)
    w = c._initial_mlp()

    w1, _, _ = c.fit(w, server_round=1)
    pred_r1 = c.predict(w1, "val", bn_round=1)
    bn_r1 = {i: w1[i].copy() for i in M.bn_indices(c.keys)}

    w2, _, _ = c.fit(w1, server_round=2)
    w3, _, _ = c.fit(w2, server_round=3)
    bn_r3 = {i: w3[i].copy() for i in M.bn_indices(c.keys)}
    assert any(not np.allclose(bn_r1[i], bn_r3[i]) for i in bn_r1), "BN stats should evolve across rounds"

    # Pin back to best_round=1 on both existing instance and fresh instance (Ray -> driver)
    c.restore_bn_round(1)
    assert np.allclose(c.predict(w1, "val"), pred_r1)

    c_fresh = FraudClient("silo_x", "mlp", frames, cfg)
    c_fresh.restore_bn_round(1)
    restored = M.mlp_to_arrays(c_fresh._mlp_model_from(w1))
    for i in M.bn_indices(c_fresh.keys):
        assert np.allclose(restored[i], bn_r1[i])
    assert np.allclose(c_fresh.predict(w1, "val"), pred_r1)


def test_budget_plan_and_magnitude_pruning_10_rounds(data):
    """Regression test: 10-round budgeted plan avoids residual eviction, and
    magnitude pruning retains higher-impact prior trees when budget is exceeded."""
    # default.yaml dimensions: k=10, rounds=10, t0=300, local_iters=20, budget=800
    r1_t0 = compute_r1_t0(300, k=10, rounds=10, local_iters=20, budget=800)
    eff_li = compute_round_local_iters(300, 20, k=10, rounds=10, budget=800)
    assert r1_t0 == 40
    assert eff_li == 4
    assert 10 * r1_t0 + 9 * 10 * eff_li <= 800

    # Unit check: magnitude pruning drops near-zero trees before large-residual trees
    toy_trees = [
        {"leaf_values": [0.001, -0.001]},   # idx 0: tiny
        {"leaf_values": [0.80, -0.75]},     # idx 1: large residual
        {"leaf_values": [0.002, 0.000]},    # idx 2: tiny
        {"leaf_values": [-0.60, 0.55]},     # idx 3: large residual
    ]
    kept = _prune_by_magnitude(toy_trees, 2, _xgb_tree_leaf_rms)
    assert kept == [toy_trees[1], toy_trees[3]]

    # Multi-round XGBoost & CatBoost forced budget cap must stay bounded and finite
    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 10, seed=1))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 10, seed=2))
    g = merge_xgb([b1, b2], normalize=True, budget=20)
    n_prev = tree_count("xgboost", g)
    foundation = n_prev
    for r in range(2, 6):
        gb = M.bytes_to_xgb(g)
        c1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 6, xgb_model=gb, seed=r))
        c2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 6, xgb_model=gb, seed=r + 10))
        g = merge_xgb([c1, c2], global_payload=g, normalize=True, n_prev=n_prev,
                      budget=36, foundation=foundation)
        n_prev = tree_count("xgboost", g)
        assert n_prev <= 36
    preds = M.xgb_predict(g, X)
    assert np.isfinite(preds).all() and (preds >= 0).all() and (preds <= 1).all()


def test_local_only_baseline_isolated_scaler():
    """Regression test: local_only MLP must fit a strictly local scaler from raw
    clients_df rather than reusing a federated global scaler stored on FraudClient."""
    import pandas as pd
    from fl_experiments.baselines import run_baselines

    rng = np.random.default_rng(11)
    def _mk_df(n, loc):
        df = pd.DataFrame({f"V{i}": rng.normal(loc=loc, scale=1.0, size=n) for i in range(1, 5)})
        df["Amount_log"] = rng.uniform(1, 5, n)
        df["hour"] = rng.integers(0, 24, n)
        df["Amount"] = np.expm1(df["Amount_log"])
        df["is_fraud"] = ((df["V1"] - loc + df["V2"] - loc) > 1.2).astype(np.int8)
        return df

    s1 = {"train": _mk_df(200, 0.0), "val": _mk_df(80, 0.0), "test": _mk_df(80, 0.0)}
    s2 = {"train": _mk_df(200, 50.0), "val": _mk_df(80, 50.0), "test": _mk_df(80, 50.0)}
    clients_df = {"s1": s1, "s2": s2}
    pooled = {sp: pd.concat([s1[sp], s2[sp]], ignore_index=True) for sp in ("train", "val", "test")}

    # Deliberately poison FraudClient's pre-scaled matrix with extreme shift
    bad_pre = MLPPreprocessor()
    bad_pre.mean_ = np.full(6, 1e6, dtype=np.float64)
    bad_pre.std_ = np.ones(6, dtype=np.float64)
    cfg = ClientCfg(dataset="ulb", seed=42)
    poisoned_clients = {
        "mlp": {name: FraudClient(name, "mlp", fr, cfg, pre=bad_pre)
                for name, fr in clients_df.items()}
    }
    clean_clients = {
        "mlp": {name: FraudClient(name, "mlp", fr, cfg, pre=None)
                for name, fr in clients_df.items()}
    }

    r_from_poisoned = run_baselines(
        pooled, poisoned_clients, ["mlp"], dataset="ulb",
        cent_epochs=3, c_fp=5.0, seed=42, clients_df=clients_df,
    )
    r_from_clean = run_baselines(
        pooled, clean_clients, ["mlp"], dataset="ulb",
        cent_epochs=3, c_fp=5.0, seed=42, clients_df=clients_df,
    )
    # Identical local_only results prove FraudClient's federated/poisoned scaler was NOT used
    assert np.isclose(
        r_from_poisoned["local_only"]["mlp"]["macro_ap"],
        r_from_clean["local_only"]["mlp"]["macro_ap"],
    )
    assert np.isfinite(r_from_poisoned["local_only"]["mlp"]["pooled_ap"])


def test_delta_tree_upload_matches_full_merge_and_single_class_guard(data):
    """Regression test: delta-tree uploads must be smaller than full models,
    produce identical merged predictions, and tolerate single-class CatBoost clients."""
    import os, tempfile
    import pandas as pd

    X, y = data
    # --- XGBoost delta upload ---
    b1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 20, seed=1))
    b2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 20, seed=2))
    g_xgb = merge_xgb([b1, b2], normalize=True)
    n_prev = tree_count("xgboost", g_xgb)
    gb = M.bytes_to_xgb(g_xgb)
    c1 = M.xgb_to_bytes(M.train_xgb(X[:300], y[:300], 5, xgb_model=gb, seed=3))
    c2 = M.xgb_to_bytes(M.train_xgb(X[300:], y[300:], 5, xgb_model=gb, seed=4))
    d1 = extract_xgb_delta(c1, n_prev)
    d2 = extract_xgb_delta(c2, n_prev)
    assert len(d1) < len(c1) and len(d2) < len(c2)
    assert tree_count("xgboost", d1) == 5 and tree_count("xgboost", d2) == 5
    full_merged = merge_xgb([c1, c2], global_payload=g_xgb, normalize=True, n_prev=n_prev)
    delta_merged = merge_xgb([d1, d2], global_payload=g_xgb, normalize=True,
                             n_prev=n_prev, is_delta=True)
    assert np.allclose(_margins(full_merged, X), _margins(delta_merged, X), atol=1e-5)

    # --- CatBoost delta upload ---
    m1 = M.cat_to_bytes(M.train_cat(X[:300], y[:300], 20, seed=1))
    m2 = M.cat_to_bytes(M.train_cat(X[300:], y[300:], 20, seed=2))
    g_cat = merge_cat([m1, m2], normalize=True)
    n_prev_c = tree_count("catboost", g_cat)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "g.cbm")
        with open(p, "wb") as f:
            f.write(g_cat)
        cm1 = M.cat_to_bytes(M.train_cat(X[:300], y[:300], 5, init_model_path=p, seed=3))
        cm2 = M.cat_to_bytes(M.train_cat(X[300:], y[300:], 5, init_model_path=p, seed=4))
    dc1 = extract_cat_delta(cm1, n_prev_c)
    dc2 = extract_cat_delta(cm2, n_prev_c)
    assert len(dc1) < len(cm1) and len(dc2) < len(cm2)
    assert tree_count("catboost", dc1) == 5 and tree_count("catboost", dc2) == 5
    full_cat = merge_cat([cm1, cm2], global_payload=g_cat, normalize=True, n_prev=n_prev_c)
    delta_cat = merge_cat([dc1, dc2], global_payload=g_cat, normalize=True,
                          n_prev=n_prev_c, is_delta=True)
    r_full = M.bytes_to_cat(full_cat).predict(X, prediction_type="RawFormulaVal")
    r_delta = M.bytes_to_cat(delta_cat).predict(X, prediction_type="RawFormulaVal")
    assert np.allclose(r_full, r_delta, atol=1e-4)

    # --- Single-class CatBoost client returns empty delta without crashing ---
    df_zero = pd.DataFrame({f"V{i}": X[:60, i % X.shape[1]] for i in range(1, 5)})
    df_zero["Amount_log"] = 2.0
    df_zero["hour"] = 12
    df_zero["Amount"] = 10.0
    df_zero["is_fraud"] = np.zeros(60, dtype=np.int8)
    c_zero = FraudClient("empty_pos", "catboost",
                         {"train": df_zero, "val": df_zero, "test": df_zero},
                         ClientCfg(t0=10, local_iters=5, dataset="ulb"))
    out_empty, n_rows, met = c_zero.fit(None, server_round=1)
    assert out_empty == b"" and n_rows == 60 and met["payload_bytes"] == 0
    # Server merge skips empty payload cleanly
    merged_skip = merge_cat([m1, out_empty], normalize=True)
    assert tree_count("catboost", merged_skip) == tree_count("catboost", m1)
    # Evaluation on all-negative val reports has_pos=0.0 cleanly (no NaN/warning)
    _, _, ev_met = c_zero.evaluate(merged_skip, server_round=1)
    assert ev_met["has_pos"] == 0.0


def test_federated_early_stopping_patience():
    """Federated patience stops both local and Flower engines when val AP plateaus."""
    import pandas as pd
    from fl_experiments.run_fl import run_local_engine, run_flower_engine

    rng = np.random.default_rng(123)
    n = 300
    def _mk_df(seed_off):
        r = np.random.default_rng(seed_off)
        df = pd.DataFrame({f"V{i}": r.normal(size=n) for i in range(1, 5)})
        df["Amount_log"] = r.uniform(1, 5, n)
        df["hour"] = r.integers(0, 24, n)
        df["Amount"] = np.expm1(df["Amount_log"])
        df["is_fraud"] = (df["V1"] > 1.5).astype(np.int8)
        return df

    # Perfectly separable threshold -> Round 1 already hits AP=1.0, so patience=1 stops at Round 2
    frames = {"train": _mk_df(1), "val": _mk_df(2), "test": _mk_df(3)}
    cfg = ClientCfg(t0=15, local_iters=5, dataset="ulb", seed=42)
    cores_loc = [FraudClient(f"s{i}", "xgboost", frames, cfg) for i in range(2)]
    _, info_loc = run_local_engine(cores_loc, "xgboost", rounds=6, normalize=False,
                                   budget=None, patience=1, verbose=False)
    assert info_loc["stopped_early"] is True
    assert len(info_loc["rounds"]) < 6

    cores_fl = [FraudClient(f"s{i}", "xgboost", frames, cfg) for i in range(2)]
    _, info_fl = run_flower_engine(cores_fl, "xgboost", rounds=6, normalize=False,
                                   budget=None, patience=1, verbose=False)
    assert info_fl["stopped_early"] is True
    assert len(info_fl["rounds"]) < 6
    assert info_loc["final_round"] == len(info_loc["rounds"])
    assert info_fl["final_round"] == len(info_fl["rounds"])
    assert np.isfinite(info_loc["final_val_ap"]) and np.isfinite(info_fl["final_val_ap"])


def test_mlp_prior_correction_and_xgb_predict_fast():
    """Verify analytical pos_weight logit shift improves MLP probability calibration
    and xgb_predict_fast matches xgb_predict_loaded."""
    import torch
    from fl_experiments.metrics import ece

    rng = np.random.default_rng(42)
    n = 3000
    X = rng.normal(size=(n, 6)).astype(np.float32)
    logits = 2.0 * X[:, 0] + 1.5 * X[:, 1] - 5.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(np.int8)
    assert 10 <= y.sum() <= 150

    torch.manual_seed(0)
    mlp = M.make_mlp(6)
    mlp = M.train_mlp(mlp, X, y, epochs=8, lr=3e-3, seed=0)
    assert getattr(mlp, "prior_logit_shift_", 0.0) > 1.0
    p_corr = M.mlp_predict(mlp, X)
    p_raw = M.mlp_predict(mlp, X, logit_shift=0.0)
    # Analytical prior correction must substantially reduce ECE on imbalanced data
    assert ece(y, p_corr) < ece(y, p_raw)

    bst = M.train_xgb(X, y, 15, seed=0)
    assert np.allclose(M.xgb_predict_fast(bst, X[:50]),
                       M.xgb_predict_loaded(bst, X[:50]), atol=1e-5)


def test_flower_strategy_aggregate_fit_is_cid_order_invariant(data):
    """TreeBaggingStrategy.aggregate_fit must produce identical merged models
    regardless of the order in which Ray returns client fit results."""
    from types import SimpleNamespace
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters
    from fl_experiments.strategy import TreeBaggingStrategy

    X, y = data
    b1 = M.xgb_to_bytes(M.train_xgb(X[:200], y[:200], 15, seed=1))
    b2 = M.xgb_to_bytes(M.train_xgb(X[200:400], y[200:400], 15, seed=2))
    b3 = M.xgb_to_bytes(M.train_xgb(X[400:], y[400:], 15, seed=3))

    def _item(cid, payload, n):
        proxy = SimpleNamespace(cid=str(cid))
        res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.frombuffer(payload, dtype=np.uint8).copy()]),
            num_examples=n,
            metrics={},
        )
        return (proxy, res)

    items_fwd = [_item(0, b1, 100), _item(1, b2, 200), _item(2, b3, 300)]
    items_rev = list(reversed(items_fwd))

    # Budget=20 forces uneven remainder split (7, 7, 6) across 3 clients
    s1 = TreeBaggingStrategy("xgboost", normalize=True, budget=20, min_clients=3, size_weighted=True)
    s2 = TreeBaggingStrategy("xgboost", normalize=True, budget=20, min_clients=3, size_weighted=True)
    s1.aggregate_fit(1, items_fwd, [])
    s2.aggregate_fit(1, items_rev, [])
    assert np.allclose(_margins(s1.current, X), _margins(s2.current, X), atol=1e-6)
