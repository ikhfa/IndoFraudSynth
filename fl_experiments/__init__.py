"""Federated ensemble fraud-detection experiments (Flower scaffolding).

Baseline replication target: Ghourabi & Khaldi (2026), Appl. Sci. 16, 4508
(federated XGBoost + CatBoost + MLP, weighted-voting ensemble).
Proposed extensions: shrinkage-normalized budget-capped tree aggregation,
FedProx/FedBN for the MLP, federated validation, per-silo cost-calibrated thresholds.
"""

__version__ = "0.1.0"
