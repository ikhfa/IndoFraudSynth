import numpy as np
import pandas as pd
import pytest

from indofraud_synth.config import SynthConfig
from indofraud_synth.generator import generate
from indofraud_synth.validation import calibration_report


@pytest.fixture(scope="module")
def tiny():
    cfg = SynthConfig(seed=7, n_customers=600, n_merchants=120, horizon_days=45,
                      start_date="2025-10-20")   # covers 11.11
    data = generate(cfg)
    return cfg, data


def test_columns_and_no_leakage_fields(tiny):
    cfg, data = tiny
    tx = data["transactions"]
    for col in ["ts", "silo_id", "rail", "amount", "is_fraud", "typology",
                "velocity_1h", "velocity_24h", "amount_z", "payee_indegree_24h",
                "merchant_velocity_24h", "gap_prev_sec"]:
        assert col in tx.columns
    assert set(tx.is_fraud.unique()) <= {0, 1}
    assert (tx.amount > 0).all()
    # window features must be finite
    for col in ["velocity_1h", "velocity_24h", "amount_z", "payee_indegree_24h"]:
        assert np.isfinite(tx[col].to_numpy()).all()


def test_determinism():
    cfg = SynthConfig(seed=11, n_customers=200, n_merchants=50, horizon_days=20,
                      start_date="2025-11-01")
    a = generate(cfg)["transactions"]
    b = generate(cfg)["transactions"]
    pd.testing.assert_frame_equal(a, b)


def test_fraud_present_and_typologies(tiny):
    cfg, data = tiny
    tx = data["transactions"]
    assert tx.is_fraud.sum() > 0
    typs = set(tx.typology.unique())
    assert "legit" in typs
    assert len(typs & {"ATO", "APP", "MULE", "QRIS_ABUSE"}) >= 2
    # ATO mostly triggers device_change; APP must NOT (social engineering!)
    ato = tx[tx.typology == "ATO"]
    app = tx[tx.typology == "APP"]
    if len(ato):
        assert ato.device_change.mean() >= 0.5
    if len(app):
        assert (app.device_change == 0).all()


def test_no_deterministic_shortcuts():
    cfg = SynthConfig(seed=19, n_customers=1500, n_merchants=200, horizon_days=60,
                      start_date="2025-10-15")
    data = generate(cfg)
    tx = data["transactions"]
    agents = data["agents"]
    rail_names = tx.attrs["rail_names"]
    ridx = {r: i for i, r in enumerate(rail_names)}

    # 1. Legit QRIS/VA/CARD must have realistic non-zero new_payee rate (no 0% shortcut)
    for r in ("QRIS", "VA", "CARD"):
        legit_r = tx[(tx.rail == ridx[r]) & (tx.is_fraud == 0)]
        assert legit_r.new_payee.mean() > 0.10, f"legit {r} new_payee too low"

    # 2. VA/CARD fraud must route through real merchant categories (no merchant_cat == -1 leak)
    for r in ("VA", "CARD"):
        fr = tx[(tx.rail == ridx[r]) & (tx.is_fraud == 1)]
        if len(fr):
            assert (fr.merchant_cat >= 0).all(), f"{r} fraud leaked merchant_cat == -1"

    # 3. Mules must have legit background transactions and causal post-intake fan-out
    mule_ids = set(agents.loc[agents.is_mule == 1, "customer_id"].tolist())
    mule_legit = tx[(tx.customer_id.isin(mule_ids)) & (tx.is_fraud == 0)]
    assert len(mule_legit) > 0, "mules should have legitimate background activity"

    mule_out = tx[tx.typology == "MULE"]
    if len(mule_out):
        intake = tx[(tx.is_fraud == 1) & (tx.payee_id.isin(mule_ids))]
        first_in = intake.groupby("payee_id")["ts"].min().to_dict()
        for m_id, grp in mule_out.groupby("customer_id"):
            assert m_id in first_in
            assert grp["ts"].min() >= first_in[m_id], "MULE fan-out occurred before intake!"


def test_stratified_make_clients_preserves_prior(tiny):
    from fl_experiments.data import temporal_split, make_clients
    _, data = tiny
    df, _ = temporal_split(data["transactions"])
    clients, pooled = make_clients(df, mode="profile", k=4, train_cap=5000, seed=42)
    # Pooled train must match the exact union of client training splits
    assert len(pooled["train"]) == sum(len(c["train"]) for c in clients.values())
    assert int(pooled["train"].is_fraud.sum()) == sum(
        int(c["train"].is_fraud.sum()) for c in clients.values()
    )


def test_calibration_and_skew(tiny):
    cfg, data = tiny
    report = calibration_report(data["transactions"], cfg)
    failed = [c["check"] for c in report["checks"] if not c["ok"]]
    # tiny population -> generous: allow at most 2 soft failures, none structural
    assert len(failed) <= 2, failed
    sizes = data["transactions"].groupby("silo_id").size()
    assert sizes.max() / max(sizes.min(), 1) >= 3.0


def test_temporal_ordering(tiny):
    _, data = tiny
    tx = data["transactions"]
    assert (tx.ts.diff().dropna() >= 0).all()


def test_no_end_of_horizon_fraud_pileup():
    cfg = SynthConfig(seed=31, n_customers=1500, n_merchants=200, horizon_days=60,
                      start_date="2025-10-15")
    data = generate(cfg)
    tx = data["transactions"]
    start_epoch = int(pd.Timestamp(cfg.start_date).timestamp())
    horizon_end = start_epoch + cfg.horizon_days * 86400
    fraud = tx[tx.is_fraud == 1]
    assert len(fraud) > 0
    # Fraud legs must stay inside horizon and never pile up on the exact last second
    assert (fraud.ts < horizon_end).all()
    assert int((fraud.ts == horizon_end - 1).sum()) <= 1


def test_profile_partition_exact_k_including_k20(tiny):
    from indofraud_synth.silos import profile_partition
    _, data = tiny
    tx = data["transactions"]
    n_raw = tx.silo_id.nunique()

    p5 = profile_partition(tx, k=5)
    assert len(p5) == 5 and "silo_other" in p5
    assert sum(len(v) for v in p5.values()) == len(tx)

    p_exact = profile_partition(tx, k=n_raw)
    assert len(p_exact) == n_raw and "silo_other" not in p_exact
    assert sum(len(v) for v in p_exact.values()) == len(tx)

    p20 = profile_partition(tx, k=20, seed=42)
    assert len(p20) == 20
    assert sum(len(v) for v in p20.values()) == len(tx)
    # Customers within a split institution must not span multiple branch sub-silos
    b1_branches = [v for nm, v in p20.items() if nm.startswith("bank_big_1_b")]
    assert len(b1_branches) >= 2
    cust_sets = [set(df.customer_id.unique()) for df in b1_branches]
    for i in range(len(cust_sets)):
        for j in range(i + 1, len(cust_sets)):
            assert cust_sets[i].isdisjoint(cust_sets[j])


def test_v1_behavioral_realism_invariants():
    """v1.0 invariants: regional locality, graph-consistent new_payee,
    no <30d mule-only tenure leak, and split typology audit."""
    cfg = SynthConfig(seed=42, n_customers=2000, n_merchants=300, horizon_days=90,
                      start_date="2025-09-15")
    data = generate(cfg)
    tx = data["transactions"]
    agents = data["agents"]
    rail_names = tx.attrs["rail_names"]
    ridx = {r: i for i, r in enumerate(rail_names)}

    # 1. Regional locality: legit QRIS & P2P have much lower region_mismatch than fraud
    legit_qris_rm = tx[(tx.rail == ridx["QRIS"]) & (tx.is_fraud == 0)].region_mismatch.mean()
    legit_p2p_rm = tx[tx.rail.isin([ridx["EWALLET"], ridx["BIFAST"]]) & (tx.is_fraud == 0)].region_mismatch.mean()
    fraud_rm = tx[tx.is_fraud == 1].region_mismatch.mean()
    assert legit_qris_rm < 0.30, f"legit QRIS region_mismatch too high: {legit_qris_rm}"
    assert legit_p2p_rm < 0.45, f"legit P2P region_mismatch too high: {legit_p2p_rm}"
    assert fraud_rm - tx[tx.is_fraud == 0].region_mismatch.mean() >= 0.15

    # 2. Chronological graph consistency: repeat (silo_id, customer_id, payee_id) MUST have new_payee == 0
    seen_dup = tx.duplicated(subset=["silo_id", "customer_id", "payee_id"], keep="first")
    assert seen_dup.any()
    assert (tx.loc[seen_dup, "new_payee"] == 0).all(), "repeat payee flagged as new_payee=1!"

    # 3. No cold-start collapse in new_payee across time (first 15% vs last 15% of legit txns)
    legit = tx[tx.is_fraud == 0]
    n_l = len(legit)
    early_np = legit.iloc[: int(0.15 * n_l)].new_payee.mean()
    late_np = legit.iloc[int(0.85 * n_l):].new_payee.mean()
    assert abs(early_np - late_np) < 0.15, f"cold-start drift in new_payee: early={early_np:.3f}, late={late_np:.3f}"

    # 4. Newly opened (<30d) non-mule accounts exist (no hard tenure < 30 mule shortcut)
    regular_fresh = agents[(agents.is_mule == 0) & (agents.tenure0_days < 30)]
    assert len(regular_fresh) > 0

    # 5. Calibration report passes and audits 70/15/15 split typology & per-silo fraud counts
    rep = calibration_report(tx, cfg)
    assert rep["ok"], [c for c in rep["checks"] if not c["ok"]]
    assert set(rep["split_typology_counts"].keys()) == {"train", "val", "test"}
    assert isinstance(rep["split_typology_coverage_ok"], bool)
    assert len(rep["per_silo_split_fraud_counts"]) == tx.silo_id.nunique()
    assert sum(sum(d.values()) for d in rep["per_silo_split_fraud_counts"].values()) == int(tx.is_fraud.sum())

    # 6. MULE fan-outs are NOT artificially pushed to the horizon tail (per-batch laundering)
    mule_tr = rep["split_typology_counts"]["train"].get("MULE", 0)
    mule_tot = rep["typology_counts"].get("MULE", 0)
    if mule_tot >= 20:
        assert mule_tr / mule_tot >= 0.50, (
            f"MULE fan-out skewed into val/test tail: train={mule_tr}/{mule_tot}"
        )

    # 7. No amount-rounding shortcut: every transaction matches rail currency step (500 QRIS, 1000 other)
    qris_mask = tx.rail == ridx["QRIS"]
    assert (tx.loc[qris_mask, "amount"] % 500 == 0).all(), "QRIS amount rounding mismatch"
    assert (tx.loc[~qris_mask, "amount"] % 1000 == 0).all(), "Non-QRIS amount rounding mismatch"

    # 8. Diurnal hour realism: daytime typologies (QRIS_ABUSE, APP) are not uniform-24h night-heavy
    qris_ab = tx[tx.typology == "QRIS_ABUSE"]
    app_tx = tx[tx.typology == "APP"]
    ato_tx = tx[tx.typology == "ATO"]
    if len(qris_ab) >= 20:
        assert qris_ab.is_night.mean() < 0.15, f"QRIS_ABUSE night rate too high: {qris_ab.is_night.mean():.3f}"
    if len(app_tx) >= 20:
        assert app_tx.is_night.mean() < 0.20, f"APP night rate too high: {app_tx.is_night.mean():.3f}"
    if len(ato_tx) >= 20:
        assert ato_tx.is_night.mean() > 0.25, f"ATO should retain elevated night bias: {ato_tx.is_night.mean():.3f}"


def test_proportional_train_cap_preserves_skew_and_config_deepcopy(tiny, tmp_path):
    """Regression tests:
    1. `make_clients` with `train_cap` scales all silos proportionally rather than
       flattening the largest silos to a uniform cap, and `train_cap=0` leaves rows uncapped.
    2. `SynthConfig.from_yaml` must not mutate module-level `DEFAULT_RAILS`.
    """
    from indofraud_synth.config import DEFAULT_RAILS
    from fl_experiments.data import temporal_split, make_clients

    _, data = tiny
    df, _ = temporal_split(data["transactions"])
    c_full, _ = make_clients(df, mode="profile", k=5, train_cap=0, seed=42)
    full_sizes = {k: len(v["train"]) for k, v in c_full.items()}
    max_full = max(full_sizes.values())
    min_full = min(full_sizes.values())
    orig_ratio = max_full / min_full

    cap = max(max_full // 2, 100)
    c_cap, pooled_cap = make_clients(df, mode="profile", k=5, train_cap=cap, seed=42)
    cap_sizes = {k: len(v["train"]) for k, v in c_cap.items()}
    assert max(cap_sizes.values()) <= cap
    assert len(pooled_cap["train"]) == sum(cap_sizes.values())
    # Max/min quantity skew ratio must be preserved within ~5%
    cap_ratio = max(cap_sizes.values()) / min(cap_sizes.values())
    assert abs(cap_ratio - orig_ratio) / orig_ratio < 0.05

    # Config deepcopy guard
    orig_qris = DEFAULT_RAILS["QRIS"]["share"]
    yfile = tmp_path / "custom.yaml"
    yfile.write_text("synth:\n  seed: 99\n")
    cfg = SynthConfig.from_yaml(str(yfile))
    cfg.rails["QRIS"]["share"] = 0.12345
    assert DEFAULT_RAILS["QRIS"]["share"] == orig_qris
