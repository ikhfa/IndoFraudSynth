"""Model-specific federated aggregation.

BASELINE (replicates Ghourabi & Khaldi 2026): concatenate client tree ensembles
without renormalization; append new trees each round (unbounded growth).

PROPOSED: shrinkage-normalized merging (leaf values scaled by 1/K, i.e., margin
averaging) + cohort-window tree budget (foundation round kept, rolling window of
later cohorts) -> bounded model size, calibrated margins, bounded latency.

MLP: FedAvg (FedProx prox term lives client-side; FedBN keeps BN local client-side).
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from .models import xgb_num_trees


# ---------------- XGBoost ----------------

def _xgb_model(j):
    return j["learner"]["gradient_booster"]["model"]


def _scale_xgb_trees(trees, factor):
    for t in trees:
        if "leaf_values" in t:                     # xgboost 1.x format
            t["leaf_values"] = [v * factor for v in t["leaf_values"]]
        if "base_weights" in t:                    # xgboost 2.x format
            t["base_weights"] = [v * factor for v in t["base_weights"]]
        # leaf nodes duplicate the leaf value in split_conditions; scale leaves only
        if "left_children" in t and "split_conditions" in t:
            sc = t["split_conditions"]
            for i, lc in enumerate(t["left_children"]):
                if lc == -1:
                    sc[i] = sc[i] * factor


def _xgb_tree_leaf_rms(tree: dict) -> float:
    """RMS magnitude of a tree's leaf values (works for XGBoost 1.x and 2.x JSON)."""
    if "leaf_values" in tree and tree["leaf_values"]:
        vals = np.asarray(tree["leaf_values"], dtype=np.float64)
        return float(np.sqrt(np.mean(vals * vals)))
    if "left_children" in tree and "split_conditions" in tree:
        sc = tree["split_conditions"]
        vals = [sc[i] for i, lc in enumerate(tree["left_children"]) if lc == -1]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            return float(np.sqrt(np.mean(arr * arr)))
    return 0.0


def _cat_tree_leaf_rms(tree: dict) -> float:
    """RMS magnitude of an oblivious CatBoost tree's leaf values."""
    vals = tree.get("leaf_values") or []
    if not vals:
        return 0.0
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def compute_r1_t0(t0: int, k: int, rounds: int, local_iters: int, budget: int | None) -> int:
    """Round-1 per-client tree count that respects `budget` while leaving room for
    subsequent rounds so round >= 2 updates are never silently discarded."""
    if budget is None or k <= 0:
        return t0
    if rounds <= 1:
        max_r1_total = budget
    else:
        later_total = (rounds - 1) * k * local_iters
        # Leave space for later rounds; keep at least half the budget for round 1
        # when total unbudgeted rounds would exceed `budget`.
        max_r1_total = max(budget - later_total, budget // 2, k)
    return min(t0, max(max_r1_total // k, 1))


def compute_round_local_iters(t0: int, local_iters: int, k: int, rounds: int,
                              budget: int | None) -> int:
    """Per-client tree count for rounds >= 2 so total trees across all rounds fit
    within `budget` without evicting intermediate boosting cohorts whenever possible."""
    if budget is None or k <= 0 or rounds <= 1:
        return local_iters
    r1_total = compute_r1_t0(t0, k, rounds, local_iters, budget) * k
    remaining = max(budget - r1_total, 0)
    later_slots = (rounds - 1) * k
    if later_slots <= 0:
        return local_iters
    return min(local_iters, max(remaining // later_slots, 1))


def _norm_weights(K: int, client_weights=None) -> list[float]:
    if client_weights is None:
        return [1.0 / K] * K
    w = np.asarray(client_weights, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        return [1.0 / K] * K
    return (w / s).tolist()


def _interleave_client_trees(per_client_trees: list[list]) -> list:
    """Interleave client trees by boosting step so any prefix is client-symmetric.

    GBDT prediction is an exact sum over trees (commutative), so interleaving
    preserves predictions bit-for-bit while guaranteeing prefix truncation never
    drops the last client(s).
    """
    out = []
    max_len = max((len(ts) for ts in per_client_trees), default=0)
    for step in range(max_len):
        for ts in per_client_trees:
            if step < len(ts):
                out.append(ts[step])
    return out


def _truncate_round1_prefixes(per_client_trees: list[list], budget: int) -> list[list]:
    """Take a valid boosting prefix from each client so total trees <= budget."""
    K = len(per_client_trees)
    base_n = max(budget // K, 1)
    rem = max(budget - base_n * K, 0)
    out = []
    for i, ts in enumerate(per_client_trees):
        take = base_n + (1 if i < rem else 0)
        out.append(ts[:take])
    # If budget < K, cap total strictly at budget
    total = sum(len(x) for x in out)
    if total > budget:
        trimmed = []
        left = budget
        for ts in out:
            if left <= 0:
                break
            trimmed.append(ts[:left])
            left -= len(trimmed[-1])
        return trimmed
    return out


def _prune_by_magnitude(trees: list, target_len: int, score_fn) -> list:
    """Keep the `target_len` trees with largest leaf RMS magnitude, preserving order."""
    if len(trees) <= target_len:
        return trees
    if target_len <= 0:
        return []
    mags = [score_fn(t) for t in trees]
    # Indices of the smallest-magnitude trees to drop (stable tie-break: drop older first)
    drop_order = sorted(range(len(trees)), key=lambda i: (mags[i], i))
    drop_set = set(drop_order[: len(trees) - target_len])
    return [t for i, t in enumerate(trees) if i not in drop_set]


def _apply_budget_rolling(trees: list, new_count: int, budget: int, foundation: int,
                          score_fn=None) -> list:
    """Cap total trees at `budget` via residual-preserving magnitude pruning.

    Instead of blindly slicing out entire intermediate boosting rounds (which
    causes margin drift) or dropping trailing clients from the foundation, this:
      1. Keeps the client-symmetric Round-1 foundation prefix (shrinking it only
         if `foundation >= budget` so newly trained round trees are never dropped).
      2. Preserves newly trained trees from the current round whenever possible,
         pruning the smallest-leaf-magnitude trees from prior post-foundation
         cohorts first (and from `new` only if `len(new) > budget - eff_foundation`).
    """
    if len(trees) <= budget:
        return trees
    if foundation >= budget:
        eff_foundation = max(budget - min(new_count, max(budget // 2, 1)), 0)
    else:
        eff_foundation = foundation
    keep_tail = max(budget - eff_foundation, 0)
    found_trees = trees[:eff_foundation]
    if keep_tail == 0:
        return found_trees

    # Split remaining trees into prior post-foundation cohorts vs newly added round trees.
    prior_end = max(len(trees) - new_count, eff_foundation)
    prior_tail = trees[eff_foundation:prior_end]
    new_trees = trees[prior_end:]

    if score_fn is None:
        return found_trees + trees[-keep_tail:]

    if len(new_trees) >= keep_tail:
        # Must prune within new_trees (prior_tail fully evicted)
        kept_new = _prune_by_magnitude(new_trees, keep_tail, score_fn)
        return found_trees + kept_new

    # Keep all new_trees and magnitude-prune prior_tail to fit remaining slots
    keep_prior = keep_tail - len(new_trees)
    kept_prior = _prune_by_magnitude(prior_tail, keep_prior, score_fn)
    return found_trees + kept_prior + new_trees


def extract_xgb_delta(full_payload: bytes, n_prev: int) -> bytes:
    """Slice only newly trained trees `trees[n_prev:]` from an XGBoost JSON payload
    so clients upload delta trees rather than re-transmitting the full global model."""
    if n_prev <= 0:
        return full_payload
    j = json.loads(full_payload)
    m = _xgb_model(j)
    delta = m["trees"][n_prev:]
    if not delta:
        return b""
    for i, t in enumerate(delta):
        t["id"] = i
    m["trees"] = delta
    m["tree_info"] = [0] * len(delta)
    m["iteration_indptr"] = list(range(len(delta) + 1))
    m["gbtree_model_param"]["num_trees"] = str(len(delta))
    return json.dumps(j).encode()


def merge_xgb(client_payloads, global_payload=None, normalize=False, n_prev=0,
              budget=None, foundation=0, client_weights=None, is_delta=False) -> bytes:
    """Round 1: global_payload=None, merge client models (prefix-capped if > budget).
    Round t: clients continued from the global model (n_prev trees); extract and
    append only their NEW trees (either pre-sliced when `is_delta=True` or via `trees[n_prev:]`)."""
    valid_pairs = []
    for idx, p in enumerate(client_payloads):
        if p:
            valid_pairs.append((idx, json.loads(p)))
    if not valid_pairs:
        if global_payload is not None:
            return global_payload
        raise ValueError("merge_xgb received no non-empty client payloads")
    js = [j for _, j in valid_pairs]
    K = len(js)
    sub_weights = ([client_weights[idx] for idx, _ in valid_pairs]
                   if client_weights is not None else None)
    weights = _norm_weights(K, sub_weights)
    if global_payload is None:
        base = js[0]
        per_client = [_xgb_model(j)["trees"] for j in js]
        if budget is not None and sum(len(ts) for ts in per_client) > budget:
            per_client = _truncate_round1_prefixes(per_client, budget)
        scaled = []
        for ts, w in zip(per_client, weights):
            if normalize:
                _scale_xgb_trees(ts, w)
            scaled.append(ts)
        trees = _interleave_client_trees(scaled)
    else:
        base = json.loads(global_payload)
        trees = _xgb_model(base)["trees"]
        per_client_new = []
        for j, w in zip(js, weights):
            ct = _xgb_model(j)["trees"] if is_delta else _xgb_model(j)["trees"][n_prev:]
            if normalize:
                _scale_xgb_trees(ct, w)
            per_client_new.append(ct)
        new = _interleave_client_trees(per_client_new)
        trees = trees + new
        if budget is not None and len(trees) > budget:
            trees = _apply_budget_rolling(
                trees, len(new), budget, foundation, score_fn=_xgb_tree_leaf_rms
            )

    for i, t in enumerate(trees):
        t["id"] = i
    m = _xgb_model(base)
    m["trees"] = trees
    m["tree_info"] = [0] * len(trees)
    m["iteration_indptr"] = list(range(len(trees) + 1))
    m["gbtree_model_param"]["num_trees"] = str(len(trees))
    return json.dumps(base).encode()


# ---------------- CatBoost ----------------

def _cat_native_to_json(payload: bytes) -> dict:
    from catboost import CatBoostClassifier
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.cbm")
        with open(p, "wb") as f:
            f.write(payload)
        m = CatBoostClassifier()
        m.load_model(p)
        jp = os.path.join(td, "m.json")
        m.save_model(jp, format="json")
        with open(jp) as f:
            return json.load(f)


def _cat_json_to_native(j: dict) -> bytes:
    from catboost import CatBoostClassifier
    with tempfile.TemporaryDirectory() as td:
        jp = os.path.join(td, "m.json")
        with open(jp, "w") as f:
            json.dump(j, f)
        m = CatBoostClassifier()
        m.load_model(jp, format="json")
        p = os.path.join(td, "m.cbm")
        m.save_model(p)
        with open(p, "rb") as f:
            return f.read()


def _scale_cat_trees(trees, factor):
    for t in trees:
        t["leaf_values"] = [v * factor for v in t["leaf_values"]]


def _union_cat_borders(base, others):
    """Trees embed their own border values; union metadata borders for safety."""
    fi = base.get("features_info", {})
    floats = fi.get("float_features", [])
    by_idx = {f.get("flat_feature_index", i): f for i, f in enumerate(floats)}
    for j in others:
        for i, f in enumerate(j.get("features_info", {}).get("float_features", [])):
            idx = f.get("flat_feature_index", i)
            if idx in by_idx:
                merged = sorted(set(by_idx[idx].get("borders", [])) |
                                set(f.get("borders", [])))
                by_idx[idx]["borders"] = merged
                by_idx[idx]["has_nans"] = bool(by_idx[idx].get("has_nans")) or bool(f.get("has_nans"))
            else:
                by_idx[idx] = f
    if floats:
        fi["float_features"] = [by_idx[k] for k in sorted(by_idx)]


def _remap_cat_split_indices(model_json: dict):
    """Update every tree split's flat `split_index` to match unioned float borders."""
    floats = model_json.get("features_info", {}).get("float_features", [])
    bmap = {}
    off = 0
    for i, f in enumerate(floats):
        fidx = f.get("flat_feature_index", f.get("feature_index", i))
        for k, b in enumerate(f.get("borders", [])):
            bmap[(fidx, b)] = off + k
        off += len(f.get("borders", []))
    for t in model_json.get("oblivious_trees", []):
        for s in t.get("splits", []):
            if s.get("split_type") == "FloatFeature":
                key = (s["float_feature_index"], s["border"])
                if key in bmap:
                    s["split_index"] = bmap[key]


def extract_cat_delta(full_payload: bytes, n_prev: int) -> bytes:
    """Slice only newly trained oblivious trees `oblivious_trees[n_prev:]` from a
    CatBoost native payload so clients upload delta trees rather than full ensembles."""
    if n_prev <= 0:
        return full_payload
    j = _cat_native_to_json(full_payload)
    delta = j.get("oblivious_trees", [])[n_prev:]
    if not delta:
        return b""
    j["oblivious_trees"] = delta
    mi = j.get("model_info", {})
    if "iteration_count" in mi:
        mi["iteration_count"] = len(delta)
    params = mi.get("params", {})
    if isinstance(params, dict) and "iteration_count" in params:
        params["iteration_count"] = str(len(delta))
    if "model_scale_and_bias" in mi:
        mi["model_scale_and_bias"] = [1.0, 0.0]
    return _cat_json_to_native(j)


def cat_num_trees(payload: bytes) -> int:
    if not payload:
        return 0
    return len(_cat_native_to_json(payload).get("oblivious_trees", []))


def merge_cat(client_payloads, global_payload=None, normalize=False, n_prev=0,
              budget=None, foundation=0, client_weights=None, is_delta=False) -> bytes:
    valid_pairs = []
    for idx, p in enumerate(client_payloads):
        if p:
            valid_pairs.append((idx, _cat_native_to_json(p)))
    if not valid_pairs:
        if global_payload is not None:
            return global_payload
        raise ValueError("merge_cat received no non-empty client payloads")
    js = [j for _, j in valid_pairs]
    K = len(js)
    sub_weights = ([client_weights[idx] for idx, _ in valid_pairs]
                   if client_weights is not None else None)
    weights = _norm_weights(K, sub_weights)
    if global_payload is None:
        base = js[0]
        per_client = [j.get("oblivious_trees", []) for j in js]
        if budget is not None and sum(len(ts) for ts in per_client) > budget:
            per_client = _truncate_round1_prefixes(per_client, budget)
        scaled = []
        for ts, w in zip(per_client, weights):
            if normalize:
                _scale_cat_trees(ts, w)
            scaled.append(ts)
        trees = _interleave_client_trees(scaled)
        _union_cat_borders(base, js[1:])
    else:
        base = _cat_native_to_json(global_payload)
        trees = base.get("oblivious_trees", [])
        per_client_new = []
        for j, w in zip(js, weights):
            raw_t = j.get("oblivious_trees", [])
            ct = raw_t if is_delta else raw_t[n_prev:]
            if normalize:
                _scale_cat_trees(ct, w)
            per_client_new.append(ct)
        new = _interleave_client_trees(per_client_new)
        trees = trees + new
        if budget is not None and len(trees) > budget:
            trees = _apply_budget_rolling(
                trees, len(new), budget, foundation, score_fn=_cat_tree_leaf_rms
            )
        _union_cat_borders(base, js)

    base["oblivious_trees"] = trees
    _remap_cat_split_indices(base)
    mi = base.get("model_info", {})
    if "iteration_count" in mi:
        mi["iteration_count"] = len(trees)
    params = mi.get("params", {})
    if isinstance(params, dict) and "iteration_count" in params:
        params["iteration_count"] = str(len(trees))
    # Logloss models use scale=1, bias=0; enforce if the key exists.
    if "model_scale_and_bias" in mi:
        mi["model_scale_and_bias"] = [1.0, 0.0]
    return _cat_json_to_native(base)


# ---------------- MLP ----------------

def fedavg(param_lists, weights=None):
    """Weighted elementwise average of state-array lists (dtype-preserving)."""
    K = len(param_lists)
    if weights is None:
        weights = np.ones(K) / K
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    out = []
    for i in range(len(param_lists[0])):
        dtype = param_lists[0][i].dtype
        acc = np.zeros_like(param_lists[0][i], dtype=np.float64)
        for plist, w in zip(param_lists, weights):
            acc += plist[i].astype(np.float64) * w
        out.append(acc.astype(dtype))
    return out


def merge_mlp(client_param_lists, weights=None, bn_idx=None):
    """FedAvg over all slots. With FedBN, clients restore their LOCAL BN slots after
    loading the global payload, so averaged BN values are simply ignored."""
    return fedavg(client_param_lists, weights)


# ---------------- dispatch ----------------

def merge_trees(model_kind, client_payloads, global_payload=None, normalize=False,
                n_prev=0, budget=None, foundation=0, client_weights=None, is_delta=False):
    if model_kind == "xgboost":
        return merge_xgb(client_payloads, global_payload, normalize, n_prev,
                         budget, foundation, client_weights=client_weights,
                         is_delta=is_delta)
    if model_kind == "catboost":
        return merge_cat(client_payloads, global_payload, normalize, n_prev,
                         budget, foundation, client_weights=client_weights,
                         is_delta=is_delta)
    raise ValueError(model_kind)


def tree_count(model_kind, payload):
    return xgb_num_trees(payload) if model_kind == "xgboost" else cat_num_trees(payload)
