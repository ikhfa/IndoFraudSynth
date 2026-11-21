"""Flower strategies with model-specific aggregation + FEDERATED VALIDATION.

The server never sees raw data: early stopping / model selection is driven by the
macro-averaged validation AUC-PR reported as SCALARS by clients (fixes the
server-side-validation inconsistency of the source paper).
"""
from __future__ import annotations

import numpy as np

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .aggregation import merge_trees, tree_count, merge_mlp, compute_r1_t0


class _BestTracker:
    def __init__(self):
        self.best_ap = -1.0
        self.best_payload = None
        self.best_round = 0
        self.round_logs = []
        self._pending_fit = {}
        self.total_client_bytes = 0
        self.total_broadcast_bytes = 0

    def note_fit(self, server_round, size, upload_bytes, broadcast_bytes):
        self._pending_fit[server_round] = {
            "round": server_round,
            "size": int(size),
            "upload_bytes": int(upload_bytes),
            "broadcast_bytes": int(broadcast_bytes),
        }

    def note_eval(self, server_round, results, payload):
        aps = []
        valid_aps = []
        for _, res in results:
            a = float(res.metrics.get("ap", 0.0))
            has_pos = float(res.metrics.get("has_pos", 1.0)) > 0.5
            if has_pos and np.isfinite(a) and a >= 0.0:
                aps.append(a)
                valid_aps.append(a)
            else:
                aps.append(None)
        macro = float(np.mean(valid_aps)) if valid_aps else 0.0
        if macro > self.best_ap:
            self.best_ap = macro
            self.best_payload = ([a.copy() for a in payload]
                                 if isinstance(payload, list) else payload)
            self.best_round = server_round
        entry = self._pending_fit.pop(server_round, {"round": server_round})
        entry["macro_val_ap"] = macro
        entry["per_client_ap"] = aps
        self.round_logs.append(entry)
        losses = [float(res.loss) for _, res in results]
        ns = [int(res.num_examples) for _, res in results]
        agg_loss = float(np.average(losses, weights=ns)) if sum(ns) > 0 else 0.0
        return agg_loss, {"macro_val_ap": macro, "best_val_ap": self.best_ap}


class TreeBaggingStrategy(FedAvg, _BestTracker):
    """Federated XGBoost/CatBoost via tree merging (baseline or normalized+budget)."""

    def __init__(self, model_kind: str, normalize: bool = False, budget=None,
                 min_clients: int = 3, r1_t0: int | None = None,
                 eff_local_iters: int | None = None,
                 size_weighted: bool = False, **kwargs):
        FedAvg.__init__(
            self, fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=min_clients, min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            initial_parameters=ndarrays_to_parameters([np.zeros(0, dtype=np.uint8)]),
            accept_failures=False, **kwargs)
        _BestTracker.__init__(self)
        self.model_kind = model_kind
        self.normalize = normalize
        self.budget = budget
        self.r1_t0 = r1_t0
        self.eff_local_iters = eff_local_iters
        self.size_weighted = size_weighted
        self.n_prev = 0
        self.foundation = 0
        self.current = None

    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in ins:
            fit_ins.config["server_round"] = server_round
            fit_ins.config["n_prev"] = self.n_prev
            if server_round == 1 and self.r1_t0 is not None:
                fit_ins.config["t0"] = int(self.r1_t0)
            elif server_round > 1 and self.eff_local_iters is not None:
                fit_ins.config["local_iters"] = int(self.eff_local_iters)
        return ins

    def aggregate_fit(self, server_round, results, failures):
        if failures or not results:
            return None, {}
        payloads = [parameters_to_ndarrays(res.parameters)[0].tobytes()
                    for _, res in results]
        c_weights = ([res.num_examples for _, res in results]
                     if (self.normalize and self.size_weighted) else None)
        merged = merge_trees(self.model_kind, payloads, global_payload=self.current,
                             normalize=self.normalize, n_prev=self.n_prev,
                             budget=self.budget, foundation=self.foundation,
                             client_weights=c_weights, is_delta=(self.n_prev > 0))
        self.current = merged
        self.n_prev = tree_count(self.model_kind, merged)
        if server_round == 1:
            self.foundation = self.n_prev
        up = sum(len(p) for p in payloads)
        down = len(merged) * len(results)
        self.total_client_bytes += up
        self.total_broadcast_bytes += down
        metrics = {"global_n_trees": int(self.n_prev),
                   "client_upload_bytes": int(up), "broadcast_bytes": int(down)}
        self.note_fit(server_round, self.n_prev, up, down)
        return ndarrays_to_parameters([np.frombuffer(merged, dtype=np.uint8).copy()]), metrics

    def configure_evaluate(self, server_round, parameters, client_manager):
        ins = super().configure_evaluate(server_round, parameters, client_manager)
        for _, ev_ins in ins:
            ev_ins.config["server_round"] = server_round
        return ins

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        return self.note_eval(server_round, results, self.current)

    def evaluate(self, server_round, parameters):
        return None  # no centralized evaluation: server holds no data


class MLPStrategy(FedAvg, _BestTracker):
    """FedAvg server-side; FedProx/FedBN semantics are client-side."""

    def __init__(self, initial_arrays, min_clients: int = 3, **kwargs):
        FedAvg.__init__(
            self, fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=min_clients, min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            initial_parameters=ndarrays_to_parameters(initial_arrays),
            accept_failures=False, **kwargs)
        _BestTracker.__init__(self)
        self.current = initial_arrays

    def configure_fit(self, server_round, parameters, client_manager):
        ins = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in ins:
            fit_ins.config["server_round"] = server_round
        return ins

    def aggregate_fit(self, server_round, results, failures):
        if failures or not results:
            return None, {}
        lists = [parameters_to_ndarrays(res.parameters) for _, res in results]
        weights = [res.num_examples for _, res in results]
        merged = merge_mlp(lists, weights)
        self.current = merged
        up = sum(int(sum(a.nbytes for a in parameters_to_ndarrays(res.parameters)))
                 for _, res in results)
        size_bytes = int(sum(a.nbytes for a in merged))
        down = size_bytes * len(results)
        self.total_client_bytes += up
        self.total_broadcast_bytes += down
        self.note_fit(server_round, size_bytes, up, down)
        return ndarrays_to_parameters(merged), {"client_upload_bytes": int(up)}

    def configure_evaluate(self, server_round, parameters, client_manager):
        ins = super().configure_evaluate(server_round, parameters, client_manager)
        for _, ev_ins in ins:
            ev_ins.config["server_round"] = server_round
        return ins

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        return self.note_eval(server_round, results, self.current)

    def evaluate(self, server_round, parameters):
        return None
