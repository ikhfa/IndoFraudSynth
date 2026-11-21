# Research Extension & Publication Plan
## From "A Federated Ensemble Learning Framework for Distributed Fraud Detection" to an Indonesia-Contextualized, SINTA-2-Ready Study

**Source paper:** Ghourabi, A.; Khaldi, K. *A Federated Ensemble Learning Framework for Distributed Fraud Detection.* Applied Sciences 2026, 16(9), 4508. https://doi.org/10.3390/app16094508 (open access, CC BY)
**Prepared:** 16 November 2026
**Goal:** A defensible, novel extension with genuine Indonesian context, publishable in a SINTA-2 (or higher) journal.

---

## 0. Executive Summary

**Verdict: yes — this paper is very extendable, and Indonesia is not just "addable context," it is the strongest available source of scientific novelty.**

The source paper's own stated limitations (equal-sized clients, mild non-IID only, no formal privacy guarantees) map almost perfectly onto the *real* structure of Indonesia's payment ecosystem: extreme institution-size skew (KBMI-4 banks vs. BPD vs. e-wallet PJP), rail heterogeneity (QRIS, BI-FAST, e-wallet, virtual accounts — not credit cards), social-engineering-dominated fraud typologies, and a hard legal barrier to raw-data pooling (UU PDP 27/2022 + bank secrecy under POJK 44/2024). That means you can fix the paper's weaknesses *because* Indonesia demands it — which is exactly what reviewers at good SINTA-2 journals want to see: context-driven methodology, not a cosmetic dataset swap.

**Recommended package (Package A):**
1. Reproduce the federated ensemble (XGBoost + CatBoost + MLP, weighted voting) as the baseline.
2. Build **IndoFraud-Synth** — an open, calibrated synthetic generator of Indonesian multi-rail payment fraud (QRIS / BI-FAST / e-wallet / VA), following the PaySim precedent, partitioned into realistically skewed institutional silos.
3. Propose **skew-aware federated aggregation + per-silo calibrated decision thresholds** and show they recover centralized-level AUC-PR under realistic heterogeneity where the original framework degrades.
4. Evaluate with temporal (out-of-time) splits, ≥5 seeds, cost-sensitive metrics in IDR, communication/latency profiling, and proper statistics — fixing every major evaluation weakness of the source paper.

**Target journals (verify rank at submission):** Kinetik (UMM, S2 until 2028), JOIN (UIN Bandung, S2), Jurnal RESTI, JITEKI (UAD), JTIIK (UB) as S2 core; IJAIN, IJEECS, IJCCS, JISEBI as S1/Scopus stretch ("at least S2" is satisfied by S1 too).

**Effort:** ~14–16 weeks to submission with 1–2 researchers and a single GPU workstation; no real bank data required (partnership path optional).

---

## 1. Critical Appraisal of the Source Paper

### 1.1 What the paper does (and does well)

| Component | Implementation |
|---|---|
| Architecture | Cross-silo FL, central server, K ∈ {3, 5, 10} clients, Flower framework |
| Models | XGBoost + CatBoost (tree-concatenation aggregation), MLP (FedAvg, 128–64 hidden units, BatchNorm) |
| Ensemble | Weighted soft voting; weights via Bayesian optimization (100 iters, validation AUC-PR); tuned threshold τ |
| Data | IEEE-CIS (590,540 tx), Sparkov (1.85M tx), European/ULB (284,807 tx, 0.172% fraud) |
| Protocol | R = 10 rounds, E = 10 local epochs, early stopping (patience 5), equal-sized clients, seed 42 |
| Results | Ensemble ≈ or > centralized at 3–5 clients; e.g., ULB AUC-PR 0.8933 (5 clients) vs. 0.8860 centralized best |
| Statistics | Friedman (χ² = 21.4, p ≈ 8.7×10⁻⁵) + Wilcoxon (p = 0.0039) over 9 dataset×config observations |

**Strengths worth preserving in your extension:** heterogeneous model diversity (boosting + neural); model-specific aggregation instead of forcing everything into FedAvg; ensemble as a *stabilizer* against federation-induced degradation; no resampling (avoids SMOTE's synthetic-data artifacts); multi-dataset evaluation; attempt at statistical testing.

### 1.2 Weaknesses and gaps (your opportunity list)

**A. Federated methodology**
1. **Tree concatenation is unnormalized and grows without bound.** Round 1 concatenates K *full* local ensembles; later rounds append new trees from every client. With K = 10, R = 10, the global ensemble can be an order of magnitude larger than a centralized model. Summing K clients' trees without shrinkage renormalization effectively multiplies the learning rate by K, inflating margins and distorting probability calibration. The paper reports no model-size, communication-volume, or inference-latency analysis — a serious omission for a "practical framework."
2. **Only mild label skew, equal client sizes.** Acknowledged by the authors (Section 4.5). No quantity skew, no feature-distribution shift, no schema heterogeneity — i.e., none of the conditions that actually define cross-institutional federation.
3. **No client dropout / partial participation**, even though the paper cites Park et al. (2025), who found FL-XGBoost sensitive to exactly that.
4. **BatchNorm + FedAvg under non-IID is a known failure mode** (running statistics diverge across clients); no FedBN/FedProx/SCAFFOLD comparison or personalization.
5. **Server-side early stopping on a validation set** implicitly assumes the aggregator holds labeled transaction data — architecturally inconsistent with the privacy premise. In a real deployment (e.g., OJK as coordinator), the server cannot hold raw labeled transactions. Federated validation or a public proxy set is needed.

**B. Evaluation rigor**
6. **Random splits on temporal data.** IEEE-CIS is explicitly time-ordered (TransactionDT); a random 80/20 split leaks future information and overestimates performance. (Sparkov's provided split is temporal; ULB spans only two days but still has Time.) No concept-drift evaluation anywhere.
7. **Single seed (42), no confidence intervals.** The Friedman/Wilcoxon tests run on 9 *correlated* observations (3 datasets × 3 client counts); p = 0.0039 is exactly the two-sided floor of a Wilcoxon test at n = 9 (2/2⁹) — consistent, but fragile evidence. No per-dataset tests, no post-hoc analysis, no effect sizes.
8. **Metric inconsistency in the comparison table.** Table 8 compares the paper's ROC-AUC (97.66%) against other papers' ROC-AUCs obtained under different splits and protocols, while the rest of the paper (correctly) argues AUC-PR is the right metric under imbalance. Cross-paper numbers without re-implementation are not a controlled comparison.
9. **Pooled test metrics hide per-silo behavior.** With skewed silos, pooled AUC-PR can look fine while small institutions are poorly served (Simpson's-paradox risk). No per-client reporting.
10. **No cost-sensitive evaluation.** Fraud detection is a cost problem (missed fraud loss vs. false-alarm friction vs. review cost); no monetary or operational metric is reported.

**C. Ensemble design**
11. **Static, globally optimized weights; no ablations.** On Sparkov the MLP weight is 0.3/15.6 ≈ 2% — the "heterogeneous trio" is effectively a duo there. No 2-model ablations, no stacking/calibration alternative, no per-silo or adaptive weights, no probability-calibration analysis (weighted averaging of uncalibrated scores + a single global τ is questionable across heterogeneous silos).

**D. Privacy & security**
12. **No formal privacy mechanism** (acknowledged): no secure aggregation, no differential privacy, no leakage analysis. Tree ensembles leak: split thresholds reveal feature ranges, leaf values reveal label statistics; FL alone is not a privacy guarantee.
13. **No adversarial robustness.** Fraud is an adversarial domain; a compromised or malicious client can poison the global tree ensemble. No Byzantine/trust-aware aggregation, no attack evaluation.

**E. Context & related work**
14. **All datasets are US/EU card-centric** (ULB 2013 PCA features; Sparkov US simulation; IEEE-CIS Vesta e-commerce). None represent emerging-market, QRIS/e-wallet/real-time-transfer-dominated ecosystems.
15. **Related-work gaps:** misses federated meta-learning for fraud (Zheng et al., IJCAI 2020, 200+ citations), the PaySim synthetic-data lineage, horizontal federated GBDT work beyond Park et al. (e.g., SecureBoost-style histogram sharing), and the recent e-wallet FL paper (Fadel et al., ICBATS 2025). No explainability discussion despite financial-regulatory needs.

> **Reviewer's bottom line:** a solid, well-written incremental paper whose ensemble idea works, but whose federation is unrealistically benign, whose evaluation has temporal-leakage and statistical weaknesses, and whose privacy claim is informal. Every one of these is a publishable extension axis.

---

## 2. Why Indonesia Is a Scientific Driver, Not Decoration

### 2.1 The ecosystem facts (all verified against official sources, Nov 2026)

| Fact | Number | Source |
|---|---|---|
| QRIS volume, H1 2026 | **12.55 billion transactions** (~2× H1 2025), ≈ Rp600.7 trillion | BI press release No. 28/190/DKom, 17 Sep 2026 |
| QRIS reach | **>66 million users, ~45 million merchants** (mostly UMKM) | same BI release |
| BI-FAST volume, Q2 2026 | **1.529 billion transactions, Rp3,777 trillion**, +38% YoY | BI data (via PaymentBrief industry summary) |
| Digital payments total, Aug 2026 | 6.11 billion tx/month, +40.4% YoY | BI (via Kumparan/Infobank) |
| Financial literacy vs. inclusion (SNLIK 2026) | **69.57% vs. 93.61%** — a 24-point gap | BI release above |
| IASC scam reports (22 Nov 2024 – 31 Jul 2026) | **637,055 reports; 1,179,881 accounts reported; 607,210 blocked** | OJK/Satgas PASTI press release SP 16/STPASTI/VIII/2026, 31 Aug 2026 |
| IASC funds | **Rp724.1 billion blocked; Rp204.3 billion returned to victims** | same OJK release |
| Online-gambling turnover (mule-account driver), 2025 | **Rp286.84 trillion in 422.1M transactions**; H1 2026: Rp86.82T | PPATK press releases, Jan & Aug 2026 |

### 2.2 The regulatory driver

- **UU No. 27/2022 (PDP Law)** — fully enforceable since 17 October 2024; imposes purpose limitation, data minimization, and security duties on personal-data processing, with administrative fines up to 2% of annual revenue.
- **POJK No. 44/2024 on Bank Secrecy (Rahasia Bank)** — codifies when customer data may be shared; inter-bank exchange of customer information is tightly constrained (e.g., prohibited for marketing).
- **POJK 11/POJK.03/2022** (IT risk management for commercial banks) and BI's PJP regulations (PBI 23/6/PBI/2021) impose fraud-risk-management duties on banks and payment providers.
- **OJK's IASC** (launched 22 Nov 2024, coordinated with ASPI/industry) already operationalizes *cross-institution* scam response — but it is **reactive** (report → trace → block). A privacy-preserving *proactive* detection layer is the natural next step, and FL is the only architecture that fits the legal constraints.

**This gives you a motivation paragraph no reviewer can call generic:** "Indonesia has the institutional coordination (IASC) and the legal prohibition (PDP + bank secrecy) that together make federated fraud detection not merely attractive but necessary — yet no study evaluates federated fraud detection under Indonesian payment-rail and institution heterogeneity."

### 2.3 Context → technical-challenge mapping (the core novelty table of your paper)

| Indonesian reality | Technical challenge the source paper ignores | Your extension component |
|---|---|---|
| KBMI-4 banks ≫ BPD ≫ e-wallet PJP ≫ P2P lenders (orders-of-magnitude data asymmetry) | Quantity skew | Profile-based + Dirichlet silo partitioning; size-aware aggregation |
| Different rails → different feature supports & distributions (QRIS merchant activity vs. BI-FAST transfer bursts vs. e-wallet P2P) | Feature-distribution & rail-mix skew | Multi-rail feature space under a standardized canonical reporting schema (analogous to BI SNAP / ISO 20022) with rail-specific missing/absent fields |
| Fraud typology = social engineering / ATO / APP scams / mule fan-out (not card-not-present) | Different positive-class structure; label skew (fraud rates vary by rail) | Typology-specific fraud injection in the generator; per-silo label priors |
| QRIS micro-tickets (median ≈ Rp10–30k) vs. ULB's €88 mean | Amount-distribution shift | Rupiah-calibrated log-normal amounts per rail |
| Payday/gajian, Ramadan/THR, Harbolnas 11.11/12.12 seasonality; fraud campaigns (e.g., World Cup gambling spikes) | Concept drift & temporal validation | Out-of-time splits; seasonal + campaign dynamics in generator |
| BI-FAST is real-time | Inference latency budget | Latency profiling; tree-budget-constrained aggregation |
| UU PDP + bank secrecy; OJK as coordinator without raw data | Server cannot hold validation data; formal privacy | Federated validation / public proxy set; (Package B: SecAgg + DP) |
| IASC dispute handling & consumer protection | Explainability | (Package C: federated SHAP) |

---

## 3. Data Strategy (the make-or-break decision)

**No canonical public Indonesian fraud-transaction dataset exists** (verified: searches surface only ULB, PaySim, and generic Kaggle sets). Choose one of three paths:

### Path 1 — Real siloed data via partnership (highest value, highest friction)
MoU with a bank/PJP/fintech, or OJK regulatory-sandbox route. Requires NDA, ethics clearance, anonymization, and 6–12 months of lead time. **Recommended only if you already have the relationship.** Even then, publish the synthetic track alongside for reproducibility.

### Path 2 — IndoFraud-Synth (recommended; the publishable artifact)
An **open-source, calibrated synthetic generator** of Indonesian multi-rail payment transactions with injected fraud typologies. Methodological precedent: **PaySim** (Lopez-Rojas et al., 2016) — a synthetic mobile-money dataset calibrated on private real aggregates, now a standard benchmark with thousands of citations. You do for Indonesian QRIS/BI-FAST/e-wallet what PaySim did for mobile money.

**Generator specification (v1):**
- **Agents:** ~50k–200k customers across city tiers (Jabodetabek, provincial capitals, rural), income segments, device profiles; ~5k–20k merchants (warung/UMKM, marketplace, pulsa, transport, F&B) with MCC-like categories.
- **Rails:** QRIS (MPM/CPM), BI-FAST transfers, e-wallet top-up/transfer/P2P, virtual-account e-commerce, (optional) cards as a minority rail.
- **Legitimate behavior:** per-rail amount distributions (log-normal, calibrated to BI published average ticket sizes: QRIS ≈ Rp600.7T/12.55B ≈ **Rp48k mean**; BI-FAST ≈ Rp3,777T/1.529B ≈ **Rp2.47M mean**), diurnal/weekly cycles, gajian payday spikes, Ramadan/THR and Harbolnas seasonality, geographic mixing.
- **Fraud typologies (injected as agent-level campaigns, not i.i.d. label flips):**
  1. **ATO drain:** OTP-phishing/APK-malware takeover → device change → rapid balance drain via BI-FAST/wallet transfer bursts;
  2. **APP scam (authorized push payment):** social-engineered victim transfers to mule accounts (impersonation, "salah transfer", task-commission scams — the exact modus operandi listed by Satgas PASTI);
  3. **Mule fan-out networks:** first-party/mule accounts receiving and rapidly dispersing funds (gambling/scam laundering; PPATK-documented patterns);
  4. **Merchant-side QRIS abuse:** static-QR tampering (quishing), fake merchants, refund abuse.
- **Calibration targets (aggregate-level, all public):** rail volumes/tickets (BI), scam-report and blocked-account counts (OJK/IASC), typology shares (Satgas PASTI modus list), fraud rate per rail set to a plausible 0.05–1% range with sensitivity analysis. **Document every calibration constant and its source in the repo** — this is your defense against "synthetic data is arbitrary."
- **Silo partitioning:** institutions as silos with profiles — 2 large banks, 2 mid banks/BPD, 3 e-wallets, 1–2 P2P/fintech → natural quantity skew (e.g., 40/25/15/8/5/3/2/1/0.7/0.3%), rail-specific feature schemas, per-silo fraud priors. Also provide Dirichlet(α) partitions (α ∈ {0.1, 0.5, 5, ∞}) for controlled skew experiments.
- **Splits:** chronological train/val/test (e.g., 12 months: 7/2/3) with a fraud-campaign drift between periods.
- **Release:** GitHub + Zenodo DOI (citable artifact), dataset card, generator seed control. This alone strengthens any S2 submission and enables a later data-descriptor paper.

### Path 3 — Hybrid (always do this regardless)
Keep **ULB + IEEE-CIS (+ Sparkov)** for comparability with the source paper and the literature, and add IndoFraud-Synth as the contextual testbed. Reviewers trust conclusions that hold on public benchmarks *and* the new setting.

> **Honesty requirement:** state clearly in the paper that Indonesian data are synthetic-but-calibrated, cite PaySim as precedent, and include a limitations paragraph + sensitivity analysis over fraud-rate and skew parameters. Never imply the data are real.

---

## 4. Extension Packages (pick ONE core per paper)

### Package A — "Federated ensemble fraud detection under realistic Indonesian cross-silo heterogeneity" ⭐ RECOMMENDED

**Research questions**
- **RQ1:** How does the Ghourabi–Khaldi federated ensemble degrade under realistic quantity/label/feature skew and temporal drift (vs. the benign equal-size setting)?
- **RQ2:** Can skew-aware aggregation — shrinkage-normalized tree merging for XGBoost/CatBoost + FedProx/FedBN for the MLP — plus per-silo calibrated thresholds recover centralized-level performance?
- **RQ3:** Does the ensemble's stabilizing effect survive out-of-time evaluation, and at what communication/latency cost?

**Method components (concrete)**
1. **Normalized tree aggregation:** when merging K clients' new trees each round, scale leaf contributions by 1/K (or by a validation-tuned shrinkage λ_k), and cap the global tree budget with per-round pruning/distillation of redundant trees → controlled model size, calibrated margins, bounded latency. (Directly fixes weakness #1; simple, defensible, novel-in-combination.)
2. **MLP under skew:** replace FedAvg with FedProx (μ tuned) and FedBN (keep BatchNorm layers local) — ablate both.
3. **Per-silo decision layer:** per-client probability calibration (Platt for small silos / isotonic for large silos, out-of-fold on local validation) + per-silo threshold τ_k / Bayes minimum-risk decisions ($p_i \cdot A_i \ge C_{\text{FP}}$); ensemble weights optimized on the simplex via Dirichlet random search + coordinate refinement with skew-aware objective = macro-averaged AUC-PR across silos (executable via 1 round of vectorized scalar AP reports, no raw data at server).
4. **Federated validation:** early stopping and weight optimization driven by *aggregated local validation metrics* (clients report scalar AUC-PR; no raw data at server) — fixes the architectural inconsistency (#5).
5. **Cost model in IDR:** expected cost = C_FN·(missed fraud amount) + C_FP·(review/friction cost per alert) + fixed review capacity; report savings vs. no-model and vs. rules baseline; alert-rate-constrained metrics (recall@FPR 0.1%/1%).

**Experimental matrix**
- Datasets: ULB, IEEE-CIS, Sparkov (comparability) + IndoFraud-Synth (context).
- Clients: K ∈ {3, 5, 10, 20}; partitions: equal (replication), profile-skew, Dirichlet α ∈ {0.1, 0.5, 5}.
- Splits: random (replication only) vs. **out-of-time** (primary).
- Models: local-only, centralized, source-paper FL ensemble (replicated), proposed skew-aware FL ensemble, + ablations (no normalization; no FedBN; global vs. per-silo τ; 2-model ensembles; soft voting; stacking).
- Seeds: ≥5 per configuration; report mean ± 95% CI.
- Metrics: AUC-PR (primary), recall@FPR{0.1%,1%}, F1, per-silo macro metrics, IDR cost savings, MB/round, total rounds to converge, inference p95 latency.
- Statistics: Friedman + Nemenyi (or Holm-corrected pairwise Wilcoxon) **per dataset** across seeds/configs; effect sizes.

**Delta vs. source paper (put this table in the manuscript):**

| Dimension | Ghourabi & Khaldi 2026 | This work |
|---|---|---|
| Client heterogeneity | Equal size, mild label skew | Quantity + label + rail-mix / feature-distribution skew, institution profiles |
| Aggregation | Unnormalized tree concatenation; FedAvg | Shrinkage-normalized, budget-capped tree merging; FedProx/FedBN |
| Decision layer | Global weights + single τ | Per-silo calibration + cost-optimal τ_k; macro-objective weighting |
| Validation | Server-side validation set | Federated validation (scalar metrics only) |
| Evaluation | Random splits, 1 seed, pooled metrics | Out-of-time, ≥5 seeds, per-silo + cost metrics |
| Efficiency | Not reported | Communication, model size, latency budgets |
| Context | US/EU card datasets | + Indonesian multi-rail calibrated benchmark (open artifact) |

**Effort:** ~14–16 weeks. **Fit:** strong S2; with excellent execution, S1/Scopus-reachable.

### Package B — "Privacy-hardened federated fraud detection aligned with UU PDP" (second paper)
Secure aggregation (Flower SecAgg+) for MLP updates + DP variants (DP-SGD for MLP; leaf/histogram noise for GBDT); membership-inference attack AUC as the privacy metric; ε–AUC-PR–cost trade-off curves; explicit mapping to UU PDP data-minimization/security duties and POJK 44/2024 sharing constraints. This is the source authors' own future work — legitimate to do, but differentiate via the Indonesian regulatory analysis and the attack-based (not just mechanism-based) evaluation. **Do not merge A+B into one paper** — scope creep kills S2 submissions; two coherent papers beat one bloated one.

### Package C — Robustness & operations (third paper / workshop)
Poisoned-client simulation (label flipping, backdoored trees) + trust-filtered tree aggregation using a small public proxy validation set; client dropout/straggler resilience; federated SHAP aggregation for OJK-compliant dispute explanations.

---

## 5. Manuscript Blueprint (Package A)

**Title candidates**
1. *Federated Ensemble Fraud Detection for Indonesia's Heterogeneous Digital Payment Ecosystem: Skew-Aware Aggregation and a Calibrated Multi-Rail Benchmark*
2. *Beyond Equal Clients: Robust Federated Ensemble Learning for Cross-Institution Fraud Detection in Indonesian Digital Payments*
3. *Privacy-Constrained Collaborative Fraud Detection under Realistic Cross-Silo Heterogeneity: An Indonesian QRIS/BI-FAST Case Study*

**Draft abstract (~200 words)**
> Indonesia's digital payment ecosystem — 12.55 billion QRIS transactions in the first half of 2026 alone — faces escalating scam losses, with the Indonesia Anti-Scam Centre blocking Rp724.1 billion across 607,210 accounts between November 2024 and July 2026. Cross-institution detection is hindered by the Personal Data Protection Law (UU 27/2022) and bank-secrecy rules, making federated learning (FL) a natural fit. Existing federated ensemble frameworks, however, are evaluated under benign conditions: equal-sized clients, mild label skew, and random data splits. We show that a state-of-the-art federated ensemble (XGBoost, CatBoost, MLP with weighted voting) degrades substantially under realistic Indonesian cross-silo heterogeneity — orders-of-magnitude quantity skew, multi-rail feature-distribution skew, and temporal drift. We propose skew-aware aggregation (shrinkage-normalized, budget-capped tree merging; FedProx/FedBN for the neural member) with per-silo calibrated decision thresholds and federated validation. On three public benchmarks and IndoFraud-Synth — a new open, aggregate-calibrated generator of Indonesian QRIS, BI-FAST, and e-wallet transactions with social-engineering and mule-network fraud typologies — the proposed framework recovers [X]% of centralized AUC-PR under extreme skew (vs. [Y]% for the baseline), reduces communication cost by [Z]×, and improves cost-weighted savings at fixed alert budgets. Results are validated across five seeds with Friedman–Nemenyi analysis under out-of-time evaluation.

**Contributions (4 bullets)**
1. First systematic stress-test of federated ensemble fraud detection under emerging-market cross-silo heterogeneity (quantity/label/feature skew + drift), with a faithful replication of the state of the art as baseline.
2. A skew-aware federated ensemble: normalized budget-capped tree aggregation, FedProx/FedBN neural training, federated validation, and per-silo cost-calibrated thresholds.
3. **IndoFraud-Synth**: an open, calibrated multi-rail Indonesian payment-fraud benchmark generator with documented typologies (ATO, APP scams, mule fan-out, QRIS abuse) and institutional silo profiles.
4. A rigorous evaluation protocol (out-of-time splits, ≥5 seeds, per-silo and IDR cost metrics, communication/latency profiling, Friedman–Nemenyi) released as reusable code.

**Section outline**
1. Introduction (Indonesian stats + regulatory gap + RQs; ~1.5 pp)
2. Related work (fraud ML → FL fraud detection incl. Zheng et al. 2020, Fadel et al. 2025, Park et al. 2025, Ghourabi & Khaldi 2026 → non-IID FL → synthetic benchmarks/PaySim; comparison table)
3. Problem setting & threat/context model (cross-silo FL; silo profiles; legal constraints; what the server may/may not see)
4. IndoFraud-Synth (generator design, calibration table with sources, typologies, validation of aggregate statistics, silo profiles)
5. Proposed framework (algorithms for normalized tree aggregation, FedProx/FedBN, federated validation, per-silo calibration; complexity & communication analysis)
6. Experimental setup (datasets, matrix, metrics, cost model, seeds, hardware, Flower version)
7. Results (RQ1 degradation curves; RQ2 recovery + ablations; RQ3 temporal robustness + efficiency; per-silo analysis; statistics)
8. Discussion (regulatory implications for IASC/OJK; deployment blueprint; limitations incl. synthetic-data caveat)
9. Conclusion & future work (Packages B/C)
+ Reproducibility appendix, data card, ethics statement.

**Expected tables/figures (8–10):** context→challenge mapping; generator calibration table; silo profile table; degradation-vs-skew curves (AUC-PR vs. Dirichlet α); main results table (4 datasets × methods, mean±CI); ablation table; PR curves out-of-time; cost-savings vs. alert-budget curves; communication/latency table; Friedman/Nemenyi diagram.

**Reproducibility checklist:** code + generator on GitHub/Zenodo; all seeds, hyperparameters, search spaces; hardware + wall-clock; exact dataset versions; baseline replication details (document any deviation from the source paper).

---

## 6. Target Journals

> **Verify at submission time** on https://sinta.kemdiktisaintek.go.id (ranks change with each accreditation round; the lists below reflect late-2025/2026 information). SINTA 1 ≈ Scopus/WoS-indexed; "at least S2" therefore also allows S1 targets.

### Tier 1 — SINTA-2 core targets (English-friendly, CS/ML scope)

| Journal | Publisher | Why it fits | Notes |
|---|---|---|---|
| **Kinetik** | Universitas Muhammadiyah Malang | Pure informatics/ML scope; accepts FL/fraud papers | **S2 accredited until 2028** (verified on journal site); quarterly |
| **JOIN (Jurnal Online Informatika)** | UIN Sunan Gunung Djati Bandung | Strong ML/applied-AI track | **S2 confirmed** (journal site, 2026); English |
| **Jurnal RESTI** | Ikatan Ahli Informatika Indonesia (IAII) | Very broad applied-IT scope; fast, high volume | S2; English/Indonesian; check current APC & queue |
| **JITEKI** | Universitas Ahmad Dahlan | ML + applied computing | S2; English |
| **JTIIK** | Universitas Brawijaya | Applied ML/DS | S2; English/Indonesian |
| **Khazanah Informatika** | Universitas Muhammadiyah Surakarta | Informatics breadth | S2; English |
| **Scientific Journal of Informatics** | Universitas Negeri Semarang | ML applications | S2; English |
| **JANAPATI** | Undiksha | Informatics | S2; English |
| **JNTETI** | UGM | Elektro-informatika | S2; more EE-flavored |
| **INTENSIF / JSINBIS** | UNP Kediri / Undip | If you lean the framing toward information systems / business intelligence (fraud + regulation angle) | S2 |

### Tier 2 — S1/Scopus stretch (Indonesian-published; exceeds the S2 requirement)

| Journal | Publisher | Fit | Caveat |
|---|---|---|---|
| **IJAIN (Int. J. of Advances in Intelligent Informatics)** | UAD | Best scope match for FL/ensemble ML | Scopus; competitive |
| **IJEECS / IJECE / BEEI** | IAES | Applied ML/EECS breadth | Scopus; APC ~USD 300–600 range (verify) |
| **TELKOMNIKA** | UAD | Telecom/computing/control | Scopus; stricter novelty bar |
| **IJCCS** | UGM | Computing & cybernetics | Scopus |
| **JISEBI** | Universitas Airlangga | IS + business intelligence — ideal if you emphasize IASC/regulatory implications | Recently S1/Scopus (verify) |
| **CommIT** | Binus | IT applications | Recently S1/Scopus (verify) |

**Submission strategy:** execute Package A fully → if results are strong and writing polished, try **IJAIN or IJEECS first** (S1); on rejection, cascade to **Kinetik → JOIN → RESTI/JITEKI** (S2). If time-constrained, submit directly to Kinetik or JOIN. Budget for APC (S2 typically Rp 1.5–4 juta; S1 USD 250–600 — verify current fees) and 1–3 months (S2) / 2–6 months (S1) review time. Avoid predatory look-alikes: check the exact ISSN on SINTA, not just the name.

---

## 7. Timeline, Risks, Ethics

### 16-week plan (1–2 researchers, single GPU workstation or Colab Pro)
| Weeks | Milestone |
|---|---|
| 1–2 | Literature consolidation (incl. Zheng 2020, Fadel 2025, PaySim, FedProx/FedBN/SCAFFOLD, federated GBDT); environment setup (Flower, XGBoost, CatBoost, PyTorch) |
| 3–4 | Faithful replication of source paper on ULB + IEEE-CIS (equal clients, random split) — sanity anchor |
| 5–7 | IndoFraud-Synth v1 + calibration documentation + silo profiles; freeze generator version |
| 8–10 | Main experiments: skew matrix, proposed aggregation, ablations, 5 seeds, out-of-time |
| 11 | Efficiency profiling + statistics + per-silo analysis |
| 12–13 | Manuscript drafting (use blueprint §5); figures |
| 14 | Internal review; English editing; reproducibility package (GitHub + Zenodo) |
| 15–16 | Journal formatting + submission; arXiv preprint (optional, establishes priority) |

### Risks & mitigations
- **"Synthetic data isn't real fraud"** → PaySim precedent; public-aggregate calibration with source table; sensitivity analysis; parallel public-benchmark results; explicit limitations. If any industry contact exists, add even a small aggregate-level validation (e.g., expert review of typology shares).
- **Compute blowup** (20 clients × 5 seeds × 4 datasets) → subsample IEEE-CIS (e.g., 200k rows) with documented protocol; Flower simulation mode; cache centralized baselines.
- **Novelty overlap** → the delta table (§4) + faithful replication makes the contribution boundary explicit; cite Fadel et al. 2025 and differentiate (they: single-setting e-wallet FL, 7-page conference paper; you: heterogeneity + ensemble + benchmark + rigor).
- **Rank changes / queue length** → verify SINTA at submission; have the cascade list ready.
- **Scope creep** → one package per paper; resist adding DP/poisoning "just a little."

### Ethics & academic integrity
- The source paper is **CC BY**: you may adapt figures/algorithms **with attribution**; cite it prominently as the baseline you replicate and extend. Do not copy text. Replicate from the algorithm descriptions; document deviations.
- Disclose synthetic data honestly; no real customer data without ethics approval, NDA, and anonymization (and even then, prefer aggregate calibration).
- Authorship/contribution statements per journal policy; declare funding and conflicts; AI-assistance disclosure per journal policy.
- If pursuing Path 1 later: UU PDP compliance (lawful basis, minimization), OJK sandbox consultation.

---

## 8. Sources (verified 16 Nov 2026)

1. Ghourabi & Khaldi, Appl. Sci. 2026, 16, 4508 — https://doi.org/10.3390/app16094508 (source paper, uploaded PDF)
2. BI press release No. 28/190/DKom, 17 Sep 2026 (QRIS H1-2026 volume/value, users, merchants; SNLIK 2026 literacy/inclusion) — https://www.bi.go.id/id/publikasi/ruang-media/news-release/Pages/sp_2819026.aspx
3. OJK/Satgas PASTI press release SP 16/STPASTI/VIII/2026, 31 Aug 2026 (IASC reports, accounts, blocked/returned funds; scam modus operandi) — https://ojk.go.id/id/berita-dan-kegiatan/info-terkini/Pages/Satgas-PASTI-Terima-25-Ribu-Pengaduan-Aktivitas-Keuangan-Ilegal-dan-IASC-Berhasil-Blokir-Dana-Terkait-Scam-Rp-724-Miliar.aspx
4. PPATK, "Catatan Capaian Strategis PPATK Tahun 2025," 28 Jan 2026 (judol Rp286.84T / 422.1M tx) — https://www.ppatk.go.id/siaran_pers/read/1594/ ; PPATK, 5 Aug 2026 (H1-2026 Rp86.82T) — https://www.ppatk.go.id/siaran_pers/read/1642/
5. BI-FAST Q2-2026 volume (1.529B tx, Rp3,777T) — https://paymentbrief.com/articles/indonesia-payments-operator-guide/ (industry summary of BI data; cross-check against BI's own SPI statistics before citing in the manuscript)
6. UU No. 27/2022 (PDP) — https://peraturan.bpk.go.id/Details/229798/uu-no-27-tahun-2022 ; POJK 44/2024 (Rahasia Bank) — https://ojk.go.id/id/regulasi/Pages/POJK-44-Tahun-2024-Rahasia-Bank-.aspx
7. Kinetik (S2 until 2028) — https://kinetik.umm.ac.id/ ; JOIN (S2) — https://join.if.uinsgd.ac.id/ ; SINTA portal — https://sinta.kemdiktisaintek.go.id
8. UNESA compiled S1/S2 electro-informatics journal list (Mar 2025; ranks partially outdated — re-verify) — https://ft.unesa.ac.id/post/daftar-jurnal-terakreditasi-scopus-dan-sinta-bidang-eletro-dan-informatika
9. Fadel, Tarek & Zamzam, "Defending Digital Transactions: Federated Learning for E-Wallet Fraud Detection," ICBATS 2025 — https://doi.org/10.1109/ICBATS66542.2025.11258285 (novelty-check reference)
10. Zheng et al., "Federated Meta-Learning for Fraudulent Credit Card Detection," IJCAI 2020 — https://ijcai.org/proceedings/2020/0642.pdf (related work missing from source paper)
11. PaySim precedent: Lopez-Rojas, Elmir & Axelsson (2016), "PaySim: A financial mobile money simulator for fraud detection," EMSS 2016; dataset — https://www.kaggle.com/datasets/ealaxi/paysim1

*Note: journal ranks, APCs, and Scopus coverage change; treat §6 as a shortlist to re-verify, not a guarantee. BI-FAST figures in row 5 come from an industry source — confirm against Bank Indonesia's official payment-system statistics before manuscript submission.*
