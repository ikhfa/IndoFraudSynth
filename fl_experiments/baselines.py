"""Centralized (individual + ensemble) and local-only baselines under the SAME protocol/metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import models as M
from .ensemble import fit_calibrators, search_weights
from .features import build_matrix, MLPPreprocessor
from .metrics import (
    ap, recall_at_fpr, cost_threshold, fraud_cost, savings, f1_at, bayes_savings,
    bayes_cost_predict, ece, brier, alert_rate, fpr_flags,
)


def _prep(df, dataset, pre=None):
    X, _ = build_matrix(df, dataset)
    if pre is not None:
        X = pre.transform(X)
    return X


def _amount_arr(df: pd.DataFrame) -> np.ndarray:
    if "amount" in df:
        return df["amount"].to_numpy(float)
    if "Amount" in df:
        return df["Amount"].to_numpy(float)
    return np.zeros(len(df), dtype=float)


def _per_silo_ap_summary(y_by_silo: dict[str, np.ndarray],
                         s_by_silo: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    per_silo = {}
    for name, y in y_by_silo.items():
        a = ap(y, s_by_silo[name])
        if not np.isnan(a):
            per_silo[name] = float(a)
    macro = float(np.mean(list(per_silo.values()))) if per_silo else float("nan")
    return macro, per_silo


def _eval_scores(y, s, amt, yv, sv, amv, c_fp, y_by_silo=None, s_by_silo=None):
    thr, _ = cost_threshold(yv, sv, amv, c_fp)
    flags = (np.asarray(s) >= thr).astype(int)
    b_flags = bayes_cost_predict(s, amt, c_fp=c_fp)
    out = {
        "pooled_ap": ap(y, s), "recall_at_fpr_001": recall_at_fpr(y, s, 0.01),
        "f1": f1_at(y, s, thr), "savings": savings(y, s, thr, amt, c_fp),
        "alert_rate": alert_rate(flags), "fpr": fpr_flags(y, flags),
        "bayes_savings": bayes_savings(y, s, amt, c_fp),
        "bayes_alert_rate": alert_rate(b_flags),
        "bayes_savings_at_1pct": bayes_savings(y, s, amt, c_fp, max_alert_rate=0.01),
        "bayes_savings_at_01pct": bayes_savings(y, s, amt, c_fp, max_alert_rate=0.001),
        "ece": ece(y, s), "brier": brier(y, s),
        "threshold": float(thr),
    }
    if y_by_silo is not None and s_by_silo is not None:
        macro, per_silo = _per_silo_ap_summary(y_by_silo, s_by_silo)
        out["macro_ap"] = macro
        out["per_silo_ap"] = per_silo
    return out


def _rules_scores(df: pd.DataFrame) -> np.ndarray:
    """Simple transparent heuristic fraud rules score in [0, 1] for IndoFraud-Synth.

    Combines standard bank/PJP rule triggers (device change, new payee, burst
    velocity, amount z-score, static-QR merchant velocity spike).
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    score = np.zeros(n, dtype=np.float64)
    if "device_change" in df:
        score += 2.5 * df["device_change"].to_numpy(np.float64)
    if "new_payee" in df:
        score += 1.0 * df["new_payee"].to_numpy(np.float64)
    if "velocity_1h" in df:
        score += 1.2 * (df["velocity_1h"].to_numpy(np.float64) >= 3)
    if "amount_z" in df:
        score += 1.0 * np.clip((df["amount_z"].to_numpy(np.float64) - 2.0) / 3.0, 0.0, 1.5)
    if "merchant_vel_z" in df:
        score += 1.2 * np.clip((df["merchant_vel_z"].to_numpy(np.float64) - 2.5) / 3.0, 0.0, 1.5)
    if "is_night" in df:
        score += 0.3 * df["is_night"].to_numpy(np.float64)
    # Map rule points to a monotonic pseudo-probability in (0, 1)
    return 1.0 / (1.0 + np.exp(-(score - 3.5)))


def run_baselines(pooled, clients, model_types, dataset="indofraud",
                  cent_iters=300, cent_epochs=6, local_t0=120, c_fp=7500.0, seed=42,
                  tree_patience=80, mlp_patience=3, clients_df=None):
    Xtr = _prep(pooled["train"], dataset)
    Xva = _prep(pooled["val"], dataset)
    Xte = _prep(pooled["test"], dataset)
    ytr = pooled["train"].is_fraud.to_numpy()
    yva = pooled["val"].is_fraud.to_numpy()
    yte = pooled["test"].is_fraud.to_numpy()
    ava = _amount_arr(pooled["val"])
    ate = _amount_arr(pooled["test"])

    first_val = next(iter(clients.values()))
    is_nested = isinstance(first_val, dict)
    ref_clients = first_val if is_nested else clients
    silo_names = list(clients_df.keys()) if clients_df is not None else list(ref_clients.keys())

    # Raw (unscaled) per-silo test matrices for evaluating centralized models per-silo
    silo_Xte_raw: dict[str, np.ndarray] = {}
    silo_yte: dict[str, np.ndarray] = {}
    for name in silo_names:
        if clients_df is not None and name in clients_df:
            silo_Xte_raw[name] = _prep(clients_df[name]["test"], dataset)
            silo_yte[name] = clients_df[name]["test"].is_fraud.to_numpy(np.int8)
        elif name in ref_clients:
            silo_yte[name] = ref_clients[name].y["test"]
            if ref_clients[name].model_type != "mlp":
                silo_Xte_raw[name] = ref_clients[name].X["test"]

    results = {"centralized": {}, "local_only": {}}

    # Heuristic rules baseline on IndoFraud-Synth (for IDR cost savings anchor)
    if dataset == "indofraud":
        r_va = _rules_scores(pooled["val"])
        r_te = _rules_scores(pooled["test"])
        r_silo = (
            {nm: _rules_scores(clients_df[nm]["test"]) for nm in silo_names}
            if clients_df is not None else None
        )
        results["rules"] = _eval_scores(
            yte, r_te, ate, yva, r_va, ava, c_fp,
            y_by_silo=(silo_yte if r_silo is not None else None),
            s_by_silo=r_silo,
        )

    cent_val_scores: dict[str, np.ndarray] = {}
    cent_test_scores: dict[str, np.ndarray] = {}
    cent_silo_scores: dict[str, dict[str, np.ndarray]] = {}
    local_val_scores: dict[str, dict[str, np.ndarray]] = {}
    local_test_scores: dict[str, dict[str, np.ndarray]] = {}
    local_val_y: dict[str, np.ndarray] = {}
    local_test_y: dict[str, np.ndarray] = {}
    local_val_amt: dict[str, np.ndarray] = {}
    local_test_amt: dict[str, np.ndarray] = {}

    for mt in model_types:
        mt_clients = clients[mt] if is_nested and mt in clients else clients
        silo_pred = None
        if mt == "xgboost":
            bst = M.train_xgb(Xtr, ytr, cent_iters, seed=seed,
                              eval_set=(Xva, yva), early_stopping_rounds=tree_patience)
            payload = M.xgb_to_bytes(bst)
            sv, st = M.xgb_predict(payload, Xva), M.xgb_predict(payload, Xte)
            if silo_Xte_raw:
                bst_loaded = M.bytes_to_xgb(payload)
                silo_pred = {nm: M.xgb_predict_loaded(bst_loaded, Xm)
                             for nm, Xm in silo_Xte_raw.items()}
            results["centralized"][mt] = _eval_scores(
                yte, st, ate, yva, sv, ava, c_fp,
                y_by_silo=(silo_yte if silo_pred is not None else None),
                s_by_silo=silo_pred,
            )
            results["centralized"][mt]["n_trees"] = M.xgb_num_trees(payload)
        elif mt == "catboost":
            model = M.train_cat(Xtr, ytr, cent_iters, seed=seed,
                                eval_set=(Xva, yva), early_stopping_rounds=tree_patience)
            payload = M.cat_to_bytes(model)
            sv, st = M.cat_predict(payload, Xva), M.cat_predict(payload, Xte)
            if silo_Xte_raw:
                silo_pred = {nm: M.cat_predict_loaded(model, Xm)
                             for nm, Xm in silo_Xte_raw.items()}
            results["centralized"][mt] = _eval_scores(
                yte, st, ate, yva, sv, ava, c_fp,
                y_by_silo=(silo_yte if silo_pred is not None else None),
                s_by_silo=silo_pred,
            )
            results["centralized"][mt]["n_trees"] = int(model.tree_count_)
        elif mt == "mlp":
            pre = MLPPreprocessor().fit(Xtr)
            Xtr_s, Xva_s, Xte_s = pre.transform(Xtr), pre.transform(Xva), pre.transform(Xte)
            import torch
            torch.manual_seed(seed)
            model = M.make_mlp(Xtr.shape[1])
            model = M.train_mlp(model, Xtr_s, ytr, epochs=cent_epochs, seed=seed,
                                eval_set=(Xva_s, yva), patience=mlp_patience)
            sv = M.mlp_predict(model, Xva_s)
            st = M.mlp_predict(model, Xte_s)
            if silo_Xte_raw:
                silo_pred = {nm: M.mlp_predict(model, pre.transform(Xm))
                             for nm, Xm in silo_Xte_raw.items()}
            results["centralized"][mt] = _eval_scores(
                yte, st, ate, yva, sv, ava, c_fp,
                y_by_silo=(silo_yte if silo_pred is not None else None),
                s_by_silo=silo_pred,
            )
            results["centralized"][mt]["best_epoch"] = int(getattr(model, "best_epoch_", cent_epochs))

        cent_val_scores[mt] = sv
        cent_test_scores[mt] = st
        if silo_pred is not None:
            cent_silo_scores[mt] = silo_pred

        # local-only: each silo trains completely isolated (local scaler, local early stopping,
        # local cost threshold, matching centralized iteration budget — zero cross-silo info).
        per_silo = {}
        ys_loc, ss_loc, amts_loc, flags_loc = [], [], [], []
        total_cost, total_loss = 0.0, 0.0
        local_iters_budget = max(int(cent_iters), int(local_t0))
        for name, c in mt_clients.items():
            if clients_df is not None and name in clients_df:
                fr = clients_df[name]
                lx_tr = _prep(fr["train"], dataset)
                lx_va = _prep(fr["val"], dataset)
                lx_te = _prep(fr["test"], dataset)
                ly_tr = fr["train"].is_fraud.to_numpy(np.int8)
                ly_va = fr["val"].is_fraud.to_numpy(np.int8)
                ly_te = fr["test"].is_fraud.to_numpy(np.int8)
                la_va = _amount_arr(fr["val"])
                la_te = _amount_arr(fr["test"])
            else:
                lx_tr, lx_va, lx_te = c.X["train"], c.X["val"], c.X["test"]
                ly_tr, ly_va, ly_te = c.y["train"], c.y["val"], c.y["test"]
                la_va, la_te = c.amt["val"], c.amt["test"]

            if len(ly_tr) == 0 or len(np.unique(ly_tr)) < 2:
                prior = float(ly_tr.mean()) if len(ly_tr) else 0.0
                sv_l = np.full(len(ly_va), prior, dtype=np.float64)
                st_l = np.full(len(ly_te), prior, dtype=np.float64)
            elif mt == "mlp":
                # Fit strictly silo-local standardization (never reuse federated fed_pre!)
                lpre = MLPPreprocessor().fit(lx_tr)
                lx_tr_s = lpre.transform(lx_tr)
                lx_va_s = lpre.transform(lx_va)
                lx_te_s = lpre.transform(lx_te)
                import torch
                torch.manual_seed(seed)
                model = M.make_mlp(lx_tr_s.shape[1])
                model = M.train_mlp(model, lx_tr_s, ly_tr, epochs=cent_epochs, seed=seed,
                                    eval_set=(lx_va_s, ly_va), patience=mlp_patience)
                sv_l = M.mlp_predict(model, lx_va_s); st_l = M.mlp_predict(model, lx_te_s)
            elif mt == "xgboost":
                bst = M.train_xgb(lx_tr, ly_tr, local_iters_budget, seed=seed,
                                  eval_set=(lx_va, ly_va), early_stopping_rounds=tree_patience)
                p = M.xgb_to_bytes(bst)
                sv_l, st_l = M.xgb_predict(p, lx_va), M.xgb_predict(p, lx_te)
            else:
                model = M.train_cat(lx_tr, ly_tr, local_iters_budget, seed=seed,
                                    eval_set=(lx_va, ly_va), early_stopping_rounds=tree_patience)
                p = M.cat_to_bytes(model)
                sv_l, st_l = M.cat_predict(p, lx_va), M.cat_predict(p, lx_te)

            thr_l, _ = cost_threshold(ly_va, sv_l, la_va, c_fp)
            fl_l = (np.asarray(st_l) >= thr_l).astype(int)
            total_cost += fraud_cost(ly_te, st_l, thr_l, la_te, c_fp)
            total_loss += float(la_te[ly_te == 1].sum())
            a = ap(ly_te, st_l)
            per_silo[name] = float(a) if not np.isnan(a) else None
            ys_loc.append(ly_te); ss_loc.append(st_l)
            amts_loc.append(la_te); flags_loc.append(fl_l)
            local_val_scores.setdefault(mt, {})[name] = np.asarray(sv_l, dtype=np.float64)
            local_test_scores.setdefault(mt, {})[name] = np.asarray(st_l, dtype=np.float64)
            local_val_y[name] = ly_va
            local_test_y[name] = ly_te
            local_val_amt[name] = la_va
            local_test_amt[name] = la_te

        vals = [v for v in per_silo.values() if v is not None]
        y_loc_all = np.concatenate(ys_loc) if ys_loc else np.zeros(0, dtype=np.int8)
        s_loc_all = np.concatenate(ss_loc) if ss_loc else np.zeros(0, dtype=np.float64)
        a_loc_all = np.concatenate(amts_loc) if amts_loc else np.zeros(0, dtype=np.float64)
        f_loc_all = np.concatenate(flags_loc) if flags_loc else np.zeros(0, dtype=int)
        b_loc_all = bayes_cost_predict(s_loc_all, a_loc_all, c_fp=c_fp)
        results["local_only"][mt] = {
            "macro_ap": float(np.mean(vals)) if vals else float("nan"),
            "pooled_ap": ap(y_loc_all, s_loc_all) if len(y_loc_all) else float("nan"),
            "recall_at_fpr_001": recall_at_fpr(y_loc_all, s_loc_all, 0.01) if len(y_loc_all) else float("nan"),
            "f1": float(f1_score(y_loc_all, f_loc_all, zero_division=0)) if len(y_loc_all) else float("nan"),
            "alert_rate": alert_rate(f_loc_all),
            "fpr": fpr_flags(y_loc_all, f_loc_all),
            "savings": float(1.0 - total_cost / total_loss) if total_loss > 0 else float("nan"),
            "bayes_savings": bayes_savings(y_loc_all, s_loc_all, a_loc_all, c_fp),
            "bayes_alert_rate": alert_rate(b_loc_all),
            "bayes_savings_at_1pct": bayes_savings(y_loc_all, s_loc_all, a_loc_all, c_fp, max_alert_rate=0.01),
            "bayes_savings_at_01pct": bayes_savings(y_loc_all, s_loc_all, a_loc_all, c_fp, max_alert_rate=0.001),
            "ece": ece(y_loc_all, s_loc_all),
            "brier": brier(y_loc_all, s_loc_all),
            "per_silo_ap": per_silo,
        }

    # Centralized calibrated ensemble upper bound (apples-to-apples vs federated ensemble)
    if len(cent_val_scores) >= 2:
        models_present = list(cent_val_scores.keys())
        cals, cal_methods, cal_val_oof = fit_calibrators(
            {m: {"pooled": cent_val_scores[m]} for m in models_present},
            {"pooled": yva},
            seed=seed,
        )
        cal_w, cal_val_obj = search_weights(cal_val_oof, {"pooled": yva}, n_draws=300, seed=seed)
        ens_va = sum(cal_w[m] * cal_val_oof[m]["pooled"] for m in models_present)
        ens_te = sum(
            cal_w[m] * (cals[m]["pooled"].predict(cent_test_scores[m])
                        if cals[m]["pooled"] is not None else cent_test_scores[m])
            for m in models_present
        )
        ens_silo = None
        if all(m in cent_silo_scores for m in models_present):
            ens_silo = {
                nm: sum(
                    cal_w[m] * (cals[m]["pooled"].predict(cent_silo_scores[m][nm])
                                if cals[m]["pooled"] is not None else cent_silo_scores[m][nm])
                    for m in models_present
                )
                for nm in silo_Xte_raw
            }
        ens_metrics = _eval_scores(
            yte, ens_te, ate, yva, ens_va, ava, c_fp,
            y_by_silo=(silo_yte if ens_silo is not None else None),
            s_by_silo=ens_silo,
        )
        ens_metrics["weights"] = cal_w
        ens_metrics["val_objective_ap"] = float(cal_val_obj)
        ens_metrics["calibration_methods"] = {m: cal_methods[m]["pooled"] for m in models_present}
        results["centralized"]["ensemble"] = ens_metrics

    # Local-only calibrated ensemble (strictly silo-local calibration, weights, and thresholds)
    if len(local_val_scores) >= 2:
        loc_models = list(local_val_scores.keys())
        loc_silos = list(local_test_y.keys())
        loc_per_silo_ap = {}
        loc_weights = {}
        ys_le, ss_le, amts_le, flags_le = [], [], [], []
        tot_cost_le, tot_loss_le = 0.0, 0.0
        for s_idx, s in enumerate(loc_silos):
            sv_s = {m: {s: local_val_scores[m][s]} for m in loc_models}
            yv_s = {s: local_val_y[s]}
            cals_s, _, oof_s = fit_calibrators(sv_s, yv_s, seed=seed + s_idx)
            w_s, _ = search_weights(oof_s, yv_s, n_draws=300, seed=seed + s_idx)
            loc_weights[s] = w_s
            ens_va_s = sum(w_s[m] * oof_s[m][s] for m in loc_models)
            ens_te_s = sum(
                w_s[m] * (cals_s[m][s].predict(local_test_scores[m][s])
                          if cals_s[m][s] is not None else local_test_scores[m][s])
                for m in loc_models
            )
            thr_s, _ = cost_threshold(local_val_y[s], ens_va_s, local_val_amt[s], c_fp)
            fl_s = (ens_te_s >= thr_s).astype(int)
            tot_cost_le += fraud_cost(local_test_y[s], ens_te_s, thr_s, local_test_amt[s], c_fp)
            tot_loss_le += float(local_test_amt[s][local_test_y[s] == 1].sum())
            a_s = ap(local_test_y[s], ens_te_s)
            loc_per_silo_ap[s] = float(a_s) if not np.isnan(a_s) else None
            ys_le.append(local_test_y[s])
            ss_le.append(ens_te_s)
            amts_le.append(local_test_amt[s])
            flags_le.append(fl_s)

        vals_le = [v for v in loc_per_silo_ap.values() if v is not None]
        y_le_all = np.concatenate(ys_le) if ys_le else np.zeros(0, dtype=np.int8)
        s_le_all = np.concatenate(ss_le) if ss_le else np.zeros(0, dtype=np.float64)
        a_le_all = np.concatenate(amts_le) if amts_le else np.zeros(0, dtype=np.float64)
        f_le_all = np.concatenate(flags_le) if flags_le else np.zeros(0, dtype=int)
        b_le_all = bayes_cost_predict(s_le_all, a_le_all, c_fp=c_fp)
        results["local_only"]["ensemble"] = {
            "macro_ap": float(np.mean(vals_le)) if vals_le else float("nan"),
            "pooled_ap": ap(y_le_all, s_le_all) if len(y_le_all) else float("nan"),
            "recall_at_fpr_001": recall_at_fpr(y_le_all, s_le_all, 0.01) if len(y_le_all) else float("nan"),
            "f1": float(f1_score(y_le_all, f_le_all, zero_division=0)) if len(y_le_all) else float("nan"),
            "alert_rate": alert_rate(f_le_all),
            "fpr": fpr_flags(y_le_all, f_le_all),
            "savings": float(1.0 - tot_cost_le / tot_loss_le) if tot_loss_le > 0 else float("nan"),
            "bayes_savings": bayes_savings(y_le_all, s_le_all, a_le_all, c_fp),
            "bayes_alert_rate": alert_rate(b_le_all),
            "bayes_savings_at_1pct": bayes_savings(y_le_all, s_le_all, a_le_all, c_fp, max_alert_rate=0.01),
            "bayes_savings_at_01pct": bayes_savings(y_le_all, s_le_all, a_le_all, c_fp, max_alert_rate=0.001),
            "ece": ece(y_le_all, s_le_all),
            "brier": brier(y_le_all, s_le_all),
            "per_silo_ap": loc_per_silo_ap,
            "per_silo_weights": loc_weights,
        }

    return results
