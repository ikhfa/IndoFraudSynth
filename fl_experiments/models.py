"""Model wrappers: XGBoost / CatBoost (native APIs, mergeable payloads) + torch MLP."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np

XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "base_score": 0.5,          # logit(0.5)=0 -> tree margins are additive across clients
    "eta": 0.1,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "nthread": 2,
    "eval_metric": "aucpr",
    "seed": 42,
}


def train_xgb(X, y, num_rounds, xgb_model=None, params=None, seed=42,
              eval_set=None, early_stopping_rounds=None):
    import xgboost as xgb
    p = dict(XGB_BASE_PARAMS); p["seed"] = seed
    if params:
        p.update(params)
    dtrain = xgb.DMatrix(X, label=y)
    if eval_set is not None and early_stopping_rounds:
        Xva, yva = eval_set
        yva_arr = np.asarray(yva)
        if 0 < yva_arr.sum() < len(yva_arr):
            dval = xgb.DMatrix(Xva, label=yva_arr)
            bst = xgb.train(
                p, dtrain, num_boost_round=num_rounds, xgb_model=xgb_model,
                evals=[(dval, "val")], early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
            # Slice to best_iteration + 1 so serialized raw JSON keeps only optimal trees.
            best_it = getattr(bst, "best_iteration", None)
            if best_it is not None and best_it >= 0:
                bst = bst[: best_it + 1]
            return bst
    return xgb.train(p, dtrain, num_boost_round=num_rounds, xgb_model=xgb_model)


def xgb_to_bytes(bst) -> bytes:
    return bytes(bst.save_raw(raw_format="json"))


def bytes_to_xgb(payload: bytes):
    import xgboost as xgb
    bst = xgb.Booster()
    bst.load_model(bytearray(payload))
    return bst


def xgb_num_trees(payload: bytes) -> int:
    j = json.loads(payload)
    return len(j["learner"]["gradient_booster"]["model"]["trees"])


def xgb_predict_loaded(bst, X) -> np.ndarray:
    import xgboost as xgb
    return bst.predict(xgb.DMatrix(X))


def xgb_predict(payload: bytes, X) -> np.ndarray:
    return xgb_predict_loaded(bytes_to_xgb(payload), X)


def cat_params(iterations, seed=42):
    return dict(
        loss_function="Logloss", iterations=iterations, depth=6, learning_rate=0.1,
        l2_leaf_reg=5, bootstrap_type="Bernoulli", subsample=0.8,
        allow_writing_files=False, thread_count=2, verbose=False, random_seed=seed,
    )


def train_cat(X, y, iterations, init_model_path=None, seed=42,
              eval_set=None, early_stopping_rounds=None):
    from catboost import CatBoostClassifier
    kw = cat_params(iterations, seed)
    use_es = False
    if eval_set is not None and early_stopping_rounds:
        Xva, yva = eval_set
        yva_arr = np.asarray(yva)
        if 0 < yva_arr.sum() < len(yva_arr):
            kw["eval_metric"] = "PRAUC"
            use_es = True
    model = CatBoostClassifier(**kw)
    if use_es:
        model.fit(
            X, y, init_model=init_model_path, eval_set=eval_set,
            use_best_model=True, early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    else:
        model.fit(X, y, init_model=init_model_path, verbose=False)
    return model


def cat_to_bytes(model) -> bytes:
    """Native .cbm bytes (usable as init_model for continuation)."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.cbm")
        model.save_model(p)
        with open(p, "rb") as f:
            return f.read()


def bytes_to_cat(payload: bytes):
    from catboost import CatBoostClassifier
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "model.cbm")
        with open(p, "wb") as f:
            f.write(payload)
        m = CatBoostClassifier()
        m.load_model(p)
        return m


def cat_predict_loaded(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def cat_predict(payload: bytes, X) -> np.ndarray:
    return cat_predict_loaded(bytes_to_cat(payload), X)


# ---------------- MLP ----------------

def make_mlp(in_dim: int):
    import torch.nn as nn

    class FraudMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            import torch.nn.functional as F
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.drop(x)
            x = F.relu(self.fc2(x))
            return self.fc3(x).squeeze(-1)

    return FraudMLP()


def mlp_keys(model):
    return list(model.state_dict().keys())


def bn_indices(keys):
    return [i for i, k in enumerate(keys) if k.startswith("bn1.")]


def mlp_to_arrays(model):
    return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]


def mlp_from_arrays(model, arrays):
    import torch
    keys = mlp_keys(model)
    sd = {k: torch.as_tensor(a) for k, a in zip(keys, arrays)}
    model.load_state_dict(sd)
    return model


def train_mlp(model, X, y, epochs=2, lr=1e-3, batch_size=1024, prox_mu=0.0,
              ref_arrays=None, seed=42, eval_set=None, patience=None):
    import torch
    from .metrics import ap as _ap
    torch.manual_seed(seed)
    device = "cpu"
    model = model.to(device).train()
    Xt = torch.as_tensor(np.array(X, dtype=np.float32, copy=True))
    yt = torch.as_tensor(np.array(y, dtype=np.float32, copy=True))
    pos = float(yt.sum()); neg = float(len(yt) - pos)
    pos_weight = torch.tensor(max(neg / max(pos, 1.0), 1.0) ** 0.5)  # mild sqrt reweight
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Map reference arrays by state_dict key so BatchNorm buffers never misalign
    # with model.named_parameters() (fixes silent skipping of fc2/fc3 in FedProx).
    ref_map = None
    if prox_mu > 0 and ref_arrays is not None:
        keys = mlp_keys(model)
        ref_map = {k: torch.as_tensor(np.array(a, dtype=np.float32, copy=True), device=device)
                   for k, a in zip(keys, ref_arrays)}
    use_es = False
    Xva = yva_arr = None
    if eval_set is not None:
        Xva, yva = eval_set
        yva_arr = np.asarray(yva)
        use_es = bool(0 < yva_arr.sum() < len(yva_arr))
    best_ap, best_arrays, best_epoch = -1.0, None, epochs
    stale = 0
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:      # BatchNorm needs >=2 samples
                continue
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            if ref_map is not None:
                prox = torch.zeros((), dtype=torch.float32, device=device)
                for name, p in model.named_parameters():
                    r = ref_map.get(name)
                    if r is not None and p.shape == r.shape:
                        prox = prox + ((p - r) ** 2).sum()
                loss = loss + (prox_mu / 2.0) * prox
            loss.backward()
            opt.step()
        if use_es:
            val_score = _ap(yva_arr, mlp_predict(model, Xva))
            if not np.isnan(val_score) and val_score > best_ap:
                best_ap = val_score
                best_arrays = mlp_to_arrays(model)
                best_epoch = ep
                stale = 0
            else:
                stale += 1
                if patience is not None and stale >= patience:
                    break
    if use_es and best_arrays is not None:
        mlp_from_arrays(model, best_arrays)
    model.best_epoch_ = best_epoch
    return model


def mlp_predict(model, X, batch_size=4096):
    import torch
    model = model.eval()
    outs = []
    with torch.no_grad():
        Xt = torch.as_tensor(X, dtype=torch.float32)
        for i in range(0, len(Xt), batch_size):
            outs.append(torch.sigmoid(model(Xt[i:i + batch_size])).numpy())
    return np.concatenate(outs) if outs else np.zeros(0)
