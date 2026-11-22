"""Engine-agnostic FL client + thin Flower NumPyClient adapter.

The same FraudClient core runs under (a) the real Flower simulation engine and
(b) a lightweight in-process loop (sandbox/CI friendly). Aggregation semantics
are identical because both call fl_experiments.aggregation.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from . import models as M
from .aggregation import extract_xgb_delta, extract_cat_delta
from .features import build_matrix, MLPPreprocessor
from .metrics import ap, recall_at_fpr


@dataclass
class ClientCfg:
    t0: int = 120              # round-1 local boosting iterations
    local_iters: int = 15      # new trees per round (t > 1)
    mlp_epochs: int = 2
    mlp_lr: float = 1e-3
    mlp_batch_size: int = 1024
    mlp_pos_weight: float | None = None  # FL-safe global sqrt(neg/pos) weight
    prox_mu: float = 0.0       # >0 enables FedProx
    mlp_agg: str = "fedavg"    # fedavg | fedprox | fedbn
    seed: int = 42
    dataset: str = "indofraud"
    bn_state_dir: str | None = None   # persistent FedBN dir (required for Flower/Ray)


class FraudClient:
    def __init__(self, name, model_type, frames, cfg: ClientCfg, pre: MLPPreprocessor | None = None,
                 shared_raw: tuple[dict, dict, dict] | None = None):
        self.name = name
        self.model_type = model_type
        self.cfg = cfg
        if shared_raw is not None:
            raw_X, raw_y, raw_amt = shared_raw
            self.X = {s: raw_X[s] for s in ("train", "val", "test")}
            self.y = raw_y
            self.amt = raw_amt
        else:
            self.X = {}; self.y = {}; self.amt = {}
            for split in ("train", "val", "test"):
                df = frames[split]
                X, _ = build_matrix(df, cfg.dataset)
                self.X[split] = X
                self.y[split] = df.is_fraud.to_numpy(np.int8)
                if "amount" in df:
                    self.amt[split] = df["amount"].to_numpy(np.float64)
                elif "Amount" in df:
                    self.amt[split] = df["Amount"].to_numpy(np.float64)
                else:
                    self.amt[split] = np.zeros(len(df))
        self.pre = None
        if model_type == "mlp":
            self.pre = pre if pre is not None else MLPPreprocessor().fit(self.X["train"])
            self.X = {s: self.pre.transform(self.X[s]) for s in ("train", "val", "test")}
            self.in_dim = self.X["train"].shape[1]
        self.local_bn = None    # FedBN: latest local BatchNorm slots
        self.bn_history: dict[int, dict[int, np.ndarray]] = {}  # round -> BN slots
        self.active_bn_round: int | None = None  # pinned BN round for best_round eval
        self.keys = M.mlp_keys(M.make_mlp(self.in_dim)) if model_type == "mlp" else None
        self._xgb_dtrain = None  # lazily cached xgboost.DMatrix (omitted on pickle)

    def release_train_buffers(self):
        """Free training feature matrices & cached DMatrix once FL training finishes."""
        self._xgb_dtrain = None
        if "train" in self.X:
            self.X["train"] = np.empty((0, 0), dtype=np.float32)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_xgb_dtrain"] = None
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._xgb_dtrain = None

    # ---------- FedBN state persistence (survives Ray worker serialization) ----------
    def _bn_safe_name(self) -> str:
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in self.name)

    def _bn_path(self, server_round: int | None = None) -> str | None:
        if not self.cfg.bn_state_dir:
            return None
        safe = self._bn_safe_name()
        if server_round is None:
            return os.path.join(self.cfg.bn_state_dir, f"{safe}_bn.npz")
        return os.path.join(self.cfg.bn_state_dir, "rounds", f"{safe}_r{int(server_round)}.npz")

    def _load_local_bn(self, server_round: int | None = None):
        if server_round is not None and server_round in self.bn_history:
            return self.bn_history[server_round]
        p = self._bn_path(server_round)
        if p and os.path.exists(p):
            with np.load(p) as data:
                bn = {int(k): data[k].copy() for k in data.files}
            if server_round is not None:
                self.bn_history[server_round] = bn
            else:
                self.local_bn = bn
            return bn
        return self.local_bn

    def _save_local_bn(self, server_round: int | None = None):
        if self.local_bn is None:
            return
        if server_round is not None:
            self.bn_history[int(server_round)] = {
                k: v.copy() for k, v in self.local_bn.items()
            }
        p = self._bn_path(None)
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, **{str(k): v for k, v in self.local_bn.items()})
        if server_round is not None:
            pr = self._bn_path(server_round)
            if pr:
                os.makedirs(os.path.dirname(pr), exist_ok=True)
                np.savez(pr, **{str(k): v for k, v in self.local_bn.items()})

    def restore_bn_round(self, server_round: int | None):
        """Pin this client's FedBN statistics to `server_round` (e.g., `best_round`)
        so post-training evaluation never pairs best-round linear weights with
        last-round BatchNorm statistics."""
        self.active_bn_round = int(server_round) if server_round is not None else None
        if self.cfg.mlp_agg == "fedbn" and self.active_bn_round is not None:
            bn = self._load_local_bn(self.active_bn_round)
            if bn is not None:
                self.local_bn = {k: v.copy() for k, v in bn.items()}

    # ---------- payload helpers ----------
    def _empty_payload(self):
        return None if self.model_type != "mlp" else None

    def _initial_mlp(self):
        import torch
        torch.manual_seed(self.cfg.seed)
        return M.mlp_to_arrays(M.make_mlp(self.in_dim))

    def _mlp_model_from(self, arrays, bn_round: int | None = None, use_active_bn: bool = True):
        model = M.make_mlp(self.in_dim)
        M.mlp_from_arrays(model, arrays)
        if self.cfg.mlp_agg == "fedbn":
            eff_round = bn_round if bn_round is not None else (
                self.active_bn_round if use_active_bn else None
            )
            if eff_round is not None:
                bn = self._load_local_bn(eff_round)
            else:
                bn = self._load_local_bn(None) if self.cfg.bn_state_dir else self.local_bn
            if bn is not None:
                cur = M.mlp_to_arrays(model)
                for i in M.bn_indices(self.keys):
                    if i in bn:
                        cur[i] = bn[i]
                M.mlp_from_arrays(model, cur)
        return model

    # ---------- core FL API ----------
    def fit(self, payload, server_round: int, t0_override: int | None = None,
            local_iters_override: int | None = None, n_prev: int = 0,
            init_model_path: str | None = None, loaded_global=None):
        cfg = self.cfg
        r1_iters = t0_override if t0_override is not None else cfg.t0
        round_iters = (local_iters_override if local_iters_override is not None
                       else cfg.local_iters)
        if self.model_type == "xgboost":
            if len(self.y["train"]) == 0:
                return b"", 0, {"n_trees": 0, "payload_bytes": 0}
            if self._xgb_dtrain is None:
                import xgboost as xgb
                self._xgb_dtrain = xgb.DMatrix(self.X["train"], label=self.y["train"])
            if server_round == 1 or payload is None:
                bst = M.train_xgb(self.X["train"], self.y["train"], r1_iters,
                                  seed=cfg.seed, dtrain=self._xgb_dtrain)
                out = M.xgb_to_bytes(bst)
            else:
                g = M.bytes_to_xgb(payload)
                bst = M.train_xgb(self.X["train"], self.y["train"], round_iters,
                                  xgb_model=g, seed=cfg.seed + server_round,
                                  dtrain=self._xgb_dtrain)
                full_out = M.xgb_to_bytes(bst)
                out = extract_xgb_delta(full_out, n_prev) if n_prev > 0 else full_out
            n_up = M.xgb_num_trees(out) if out else 0
            return out, len(self.y["train"]), {"n_trees": n_up,
                                               "payload_bytes": len(out)}

        if self.model_type == "catboost":
            y_tr = self.y["train"]
            # CatBoost requires >= 2 classes in training targets; under extreme
            # Dirichlet skew a client with 0 positives contributes 0 new trees.
            if len(y_tr) == 0 or len(np.unique(y_tr)) < 2:
                return b"", len(y_tr), {"payload_bytes": 0}
            if server_round == 1 or payload is None:
                model = M.train_cat(self.X["train"], y_tr, r1_iters, seed=cfg.seed)
                out = M.cat_to_bytes(model)
            elif n_prev > 0:
                # Train delta trees directly on the global model's raw margin baseline:
                # mathematically identical to init_model + extract_cat_delta, faster,
                # and avoids passing unioned multi-silo float borders back into CatBoost.
                g_cat = loaded_global if loaded_global is not None else M.bytes_to_cat(payload)
                base_margin = M.cat_predict_raw_loaded(g_cat, self.X["train"])
                model = M.train_cat(self.X["train"], y_tr, round_iters,
                                    baseline=base_margin, seed=cfg.seed + server_round)
                out = M.cat_to_bytes(model)
            elif init_model_path is not None and os.path.exists(init_model_path):
                model = M.train_cat(self.X["train"], y_tr, round_iters,
                                    init_model_path=init_model_path,
                                    seed=cfg.seed + server_round)
                out = M.cat_to_bytes(model)
            else:
                with tempfile.TemporaryDirectory() as td:
                    p = os.path.join(td, "global.cbm")
                    with open(p, "wb") as f:
                        f.write(payload)
                    model = M.train_cat(self.X["train"], y_tr, round_iters,
                                        init_model_path=p, seed=cfg.seed + server_round)
                out = M.cat_to_bytes(model)
            return out, len(y_tr), {"payload_bytes": len(out)}

        if self.model_type == "mlp":
            import torch
            torch.manual_seed(cfg.seed + server_round)
            if payload is None:
                model = M.make_mlp(self.in_dim)
                ref = None
            else:
                model = self._mlp_model_from(payload, use_active_bn=False)
                ref = payload if cfg.mlp_agg == "fedprox" else None
            mu = cfg.prox_mu if cfg.mlp_agg == "fedprox" else 0.0
            if len(self.y["train"]) >= 2:
                model = M.train_mlp(model, self.X["train"], self.y["train"],
                                    epochs=cfg.mlp_epochs, lr=cfg.mlp_lr,
                                    batch_size=cfg.mlp_batch_size, prox_mu=mu,
                                    ref_arrays=ref, seed=cfg.seed + server_round,
                                    pos_weight=cfg.mlp_pos_weight)
            arrays = M.mlp_to_arrays(model)
            if cfg.mlp_agg == "fedbn":
                self.local_bn = {i: arrays[i].copy() for i in M.bn_indices(self.keys)}
                self._save_local_bn(server_round=server_round)
            nbytes = int(sum(a.nbytes for a in arrays))
            return arrays, len(self.y["train"]), {"payload_bytes": nbytes}

        raise ValueError(self.model_type)

    def _mlp_logit_shift(self) -> float:
        if self.cfg.mlp_pos_weight is not None:
            pw = max(float(self.cfg.mlp_pos_weight), 1.0)
        else:
            y_tr = self.y["train"]
            pw = M.compute_mlp_pos_weight(float(y_tr.sum()), float(len(y_tr))) if len(y_tr) else 1.0
        return float(np.log(pw))

    def predict(self, payload, split="val", bn_round: int | None = None, loaded_model=None):
        if self.model_type == "xgboost":
            if loaded_model is not None:
                return M.xgb_predict_fast(loaded_model, self.X[split])
            return M.xgb_predict(payload, self.X[split])
        if self.model_type == "catboost":
            if loaded_model is not None:
                return M.cat_predict_loaded(loaded_model, self.X[split])
            return M.cat_predict(payload, self.X[split])
        model = self._mlp_model_from(payload, bn_round=bn_round)
        return M.mlp_predict(model, self.X[split], logit_shift=self._mlp_logit_shift())

    def evaluate(self, payload, server_round: int | None = None, loaded_model=None):
        y_val = self.y["val"]
        has_pos = bool(len(y_val) > 0 and 0 < y_val.sum() < len(y_val))
        if not has_pos:
            return 1.0, len(y_val), {
                "ap": -1.0,
                "has_pos": 0.0,
                "recall_at_fpr_001": -1.0,
            }
        s = self.predict(payload, "val", bn_round=server_round, loaded_model=loaded_model)
        a = ap(y_val, s)
        if np.isnan(a):
            return 1.0, len(y_val), {
                "ap": -1.0,
                "has_pos": 0.0,
                "recall_at_fpr_001": -1.0,
            }
        return float(1.0 - a), len(y_val), {
            "ap": float(a),
            "has_pos": 1.0,
            "recall_at_fpr_001": float(recall_at_fpr(y_val, s, 0.01)),
        }


# ---------------- Flower adapter ----------------

def params_to_payload(parameters, model_type):
    if model_type == "mlp":
        return [np.asarray(a) for a in parameters] if len(parameters) else None
    arr = np.asarray(parameters[0])
    return arr.tobytes() if arr.size else None


def payload_to_params(payload, model_type):
    if model_type == "mlp":
        return payload
    if not payload:
        return [np.zeros(0, dtype=np.uint8)]
    return [np.frombuffer(payload, dtype=np.uint8).copy()]


def make_flower_client(core: FraudClient):
    import flwr as fl

    class FlowerFraudClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            if core.model_type == "mlp":
                return core._initial_mlp()
            return [np.zeros(0, dtype=np.uint8)]

        def fit(self, parameters, config):
            payload = params_to_payload(parameters, core.model_type)
            t0_ov = int(config["t0"]) if "t0" in config else None
            li_ov = int(config["local_iters"]) if "local_iters" in config else None
            n_prev = int(config.get("n_prev", 0))
            out, n, metrics = core.fit(
                payload, int(config.get("server_round", 1)),
                t0_override=t0_ov, local_iters_override=li_ov, n_prev=n_prev,
            )
            metrics = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                       for k, v in metrics.items()}
            return payload_to_params(out, core.model_type), int(n), metrics

        def evaluate(self, parameters, config):
            payload = params_to_payload(parameters, core.model_type)
            sr = int(config["server_round"]) if "server_round" in config else None
            loss, n, metrics = core.evaluate(payload, server_round=sr)
            return float(loss), int(n), {k: float(v) for k, v in metrics.items()}

    return FlowerFraudClient()
