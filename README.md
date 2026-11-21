# IndoFraud-Synth & Skew-Aware Federated Ensemble Fraud Detection

An open, aggregate-calibrated synthetic benchmark (**IndoFraud-Synth**) and federated ensemble fraud-detection framework tailored to Indonesia's heterogeneous multi-rail digital payment ecosystem (**QRIS, BI-FAST, e-wallet, virtual accounts, cards**), extending Ghourabi & Khaldi (*Applied Sciences* 2026, 16(9), 4508).

> **Synthetic Data Disclosure:** IndoFraud-Synth generates **100% synthetic transactions**. No real customer data are used. Public Bank Indonesia (BI), OJK/Satgas PASTI (IASC), and PPATK aggregates calibrate ticket sizes, typologies, and rail caps; assumptions (per-rail fraud rates, market shares) are explicitly tagged in `indofraud_synth/calibration.py` and in every generated `DATASET_CARD.md`.

---

## Key Contributions & Features

1. **IndoFraud-Synth (`indofraud_synth/`):**
   - Multi-rail Indonesian payment simulator (QRIS, EWALLET, BIFAST, VA, CARD) with payday (*gajian*), weekend, Harbolnas (11.11 / 12.12), and optional THR seasonality.
   - Four agent-level fraud campaign typologies informed by OJK IASC / PPATK patterns: **ATO** (account takeover), **APP** (social-engineering authorized push payment), **MULE** (causal post-intake fan-out laundering with realistic background activity), and **QRIS_ABUSE** (static-QR tampering), plus configurable out-of-time concept drift (`drift_strength`).
   - Realistic cross-silo institutional partitioning (`profile` KBMI-style skew, `equal`, and `dirichlet(α)`) with strictly **silo-local, leakage-safe** temporal window features.
   - Automated aggregate calibration validation (`calibration_report.md` / `.json`) and dataset card generation.

2. **Skew-Aware Federated Ensemble (`fl_experiments/`):**
   - Heterogeneous federated trio: **XGBoost + CatBoost + PyTorch MLP** running on both Flower (`--engine flower`, Ray simulation or in-process Flower `Server`) and an identical-semantics local loop (`--engine local`).
   - **Budget-capped, shrinkage-normalized tree merging** (`--tree-agg normalized`, uniform $1/K$ or size-weighted `--tree-weight size`) — preventing unnormalized margin explosion, Hessian collapse, and unbounded tree growth.
   - **FedProx** and persistent **FedBN** for the federated MLP with FL-safe global standardization from per-client sufficient statistics `(count, sum, sum_sq)`.
   - **Federated validation** (scalar per-silo AUC-PR early stopping; server holds no raw data), **per-silo out-of-fold probability calibration** (Platt scaling for small silos, isotonic regression for large silos) *before* ensemble weight search, and **IDR cost-sensitive / Bayes minimum-risk decisions** ($p_i \cdot A_i \ge C_{\text{FP}}$).
   - **Public benchmark comparability:** temporal-split loaders for **ULB**, **IEEE-CIS**, and **Sparkov**.
   - **Publication statistics (`fl_experiments/stats.py`, `run_suite.py`):** multi-seed runner with Student-$t$ 95% CIs, Friedman test + Kendall's $W$, Nemenyi Critical Difference diagrams, Holm–Bonferroni Wilcoxon tests, Markdown summaries, and LaTeX tables.

---

## Repository Structure

```text
IndoFraudSynth/
├── configs/
│   ├── quick.yaml              # Fast sandbox/iteration config (4k customers, 90 days)
│   └── default.yaml            # Full workstation publication config (50k customers, 180 days)
├── indofraud_synth/            # Synthetic Indonesian multi-rail transaction generator
│   ├── config.py               # Rail specs, fraud rates, silo profiles, drift & seasonality
│   ├── agents.py               # Customers, mules, aggregators, merchants & acquirers
│   ├── behavior.py             # Vectorized legitimate transaction generation
│   ├── fraud.py                # Campaign-level ATO, APP, MULE, QRIS_ABUSE injection
│   ├── silos.py                # Institution silo assignment + profile/equal/Dirichlet splits
│   ├── features.py             # Leakage-safe, silo-local chronological window features
│   ├── calibration.py          # Documented public calibration sources & assumptions
│   ├── validation.py           # Aggregate calibration report checker
│   └── generator.py / cli.py   # Orchestrator, DATASET_CARD export, CLI entry point
├── fl_experiments/             # Federated learning, baselines, ensembles & statistics
│   ├── data.py                 # Loaders (IndoFraud, ULB, IEEE-CIS, Sparkov) + temporal splits
│   ├── features.py             # Feature matrices & FL-safe federated MLP standardization
│   ├── models.py               # XGBoost, CatBoost, PyTorch MLP (+ early stopping, FedProx)
│   ├── aggregation.py          # Baseline & normalized/budgeted tree merging + FedAvg
│   ├── client.py               # FraudClient + Flower NumPyClient + persistent FedBN
│   ├── strategy.py             # Flower TreeBaggingStrategy & MLPStrategy (federated val)
│   ├── ensemble.py             # OOF Platt/Isotonic calibration, weight search, Bayes decisions
│   ├── baselines.py            # Early-stopped centralized + local-only baselines
│   ├── metrics.py              # AUC-PR, recall@FPR, ECE, Brier, IDR cost/Bayes savings, p95 latency
│   ├── run_fl.py               # Single-experiment CLI (local or Flower engine)
│   ├── run_suite.py            # Multi-seed × condition × partition suite runner
│   └── stats.py                # Mean ± 95% CI, Friedman-Nemenyi, Holm-Wilcoxon, CD plot, LaTeX
└── tests/                      # Unit & end-to-end tests (pytest)
```

---

## Setup

Python 3.11+ is recommended. Using [`uv`](https://docs.astral.sh/uv/) is the fastest way to create and populate the environment:

```bash
# Install uv if needed:
#   Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows:     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv venv .venv --python 3.11
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

uv pip install -r requirements.txt
# Optional: install pyarrow for automatic fast Parquet dataset caching
uv pip install pyarrow
# Optional: install Ray for multi-process Flower simulation
# uv pip install "flwr[simulation]"
```

Run the test suite to verify the installation:

```bash
python -m pytest -v
```

---

## Fast Execution (GPU, Multi-Core CPU, Google Colab & Kaggle)

The runner automatically accelerates experiments without altering methodology:

- **Auto GPU detection (`--device auto`):** Uses CUDA automatically for PyTorch MLP and XGBoost (`tree_method=hist, device=cuda`) whenever a GPU is available, and uses all CPU cores (`--threads 0`) for CatBoost and CPU runs.
- **Cross-condition model caching:** `run_suite.py` automatically caches trained per-model FL payloads in `<partition>/_model_cache/` so identical XGBoost/CatBoost/MLP runs shared across ablation conditions (`normalized`, `normalized_fedprox`, `normalized_fedbn`, `normalized_size_fedbn`) are trained once per seed (~40% fewer FL trainings with bit-for-bit identical outputs). Pass `--no-model-cache` to disable.
- **Automatic Parquet caching:** Large CSVs (`transactions.csv.gz`, IEEE-CIS, Sparkov, ULB) are cached to `.parquet` on first load when `pyarrow` is installed (~2s reload vs. ~35s CSV parse). Set `INDOFRAUD_DATA_CACHE=/writable/dir` when inputs live on read-only mounts (e.g. `/kaggle/input`).
- **Parallel seed workers (multi-core CPU):** Pass `--workers 2` to `run_suite.py` to run seeds concurrently (threads per worker are auto-divided). Keep `--workers 1` on a single GPU.

### Cloud Notebooks (Ready-to-Run with `uv`)

- **Google Colab:** [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) — installs dependencies via `uv pip install --system` while preserving Colab's CUDA PyTorch, caches Parquet data on Google Drive, and writes results directly to Drive with `--skip-existing` for disconnect-safe resumption.
- **Kaggle (Free 30 GB RAM + `GPU T4 x2` / `P100`):** [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb) — auto-discovers attached Kaggle public datasets (ULB, IEEE-CIS, Sparkov), supports resuming from previous notebook outputs, and includes a **Dual-GPU (`T4 x2`) parallel sharding cell** that splits seeds across GPU 0 and GPU 1 simultaneously.

---

## Usage

### 1. Generate `IndoFraud-Synth`

```bash
# Quick dataset (~240k transactions, 90 days)
python -m indofraud_synth --config configs/quick.yaml

# Full publication-scale dataset (~6M transactions, 180 days)
python -m indofraud_synth --config configs/default.yaml
```

Outputs `transactions.csv.gz`, `agents.csv.gz`, `merchants.csv.gz`, `meta.json`, `DATASET_CARD.md`, and `calibration_report.{md,json}`.

### 2. Run Single FL Experiments (Baseline vs. Proposed)

```bash
# Source-paper baseline (unnormalized tree concatenation, FedAvg)
python -m fl_experiments.run_fl --config configs/quick.yaml \
    --tree-agg baseline --mlp-agg fedavg --out results/quick_baseline

# Proposed skew-aware FL (normalized + budgeted trees, FedBN, per-silo calibration)
python -m fl_experiments.run_fl --config configs/quick.yaml \
    --tree-agg normalized --mlp-agg fedbn --out results/quick_normalized

# Using the Flower engine (automatically uses Ray if installed, else Flower Server in-process)
python -m fl_experiments.run_fl --config configs/quick.yaml \
    --engine flower --tree-agg normalized --mlp-agg fedbn --out results/quick_flower
```

### 3. Run Public Benchmarks (Temporal Splits)

Download datasets manually from Kaggle (not redistributed in this repo):
- **ULB:** `creditcard.csv` → `--dataset ulb --data path/to/creditcard.csv`
- **IEEE-CIS:** directory with `train_transaction.csv` (+ `train_identity.csv`) → `--dataset ieee_cis --data path/to/ieee_dir`
- **Sparkov:** directory with `fraudTrain.csv` (+ `fraudTest.csv`) → `--dataset sparkov --data path/to/sparkov_dir`

```bash
python -m fl_experiments.run_fl --dataset ieee_cis --data data/ieee_cis \
    --partition dirichlet --alpha 0.5 --k 5 --rounds 10 \
    --tree-agg normalized --mlp-agg fedbn --out results/ieee_normalized
```

### 4. Multi-Seed Suite & Friedman–Nemenyi Statistics

```bash
# Run 5 seeds across baseline, normalized, and normalized+FedBN and generate
# summary.json, summary.md, table.tex, and cd_diagram.png
python -m fl_experiments.run_suite --config configs/default.yaml \
    --seeds 1,2,3,4,5 \
    --conditions baseline,normalized,normalized_fedprox,normalized_fedbn,normalized_size_fedbn \
    --partitions equal,profile,dirichlet:0.1,dirichlet:0.5,dirichlet:5.0 \
    --out results/suite_full --skip-existing

# Re-run statistical analysis on an existing suite directory with another metric
python -m fl_experiments.stats --suite-dir results/suite_full/profile --metric bayes_savings
```

---

## Reproducibility & Ethics

- All data generation, partitioning, training, calibration, and weight search steps are seed-controlled (`--seed`).
- Temporal out-of-time splits (`70% / 15% / 15%`) are enforced across all datasets to prevent future leakage.
- When citing or extending this benchmark, retain the synthetic-data disclosure and report sensitivity analyses over fraud rates and skew parameters (`--alpha`).
