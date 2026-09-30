# FedLoRA Health: Product Requirements Document

> Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model.
>
> **Concepts:** `Fine Tuning`, `LoRA`, `Quantization`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LLMs`

## 1. Overview

FedLoRA Health runs federated learning where only LoRA adapter updates, never data, leave each site. Clients fine-tune a 4-bit quantised base (QLoRA) on local notes, add differential-privacy noise and clip updates, and a secure aggregator combines them (FedAvg/FedProx over LoRA A/B factors). Heterogeneous sites are handled with per-site personal adapters plus a shared global adapter. A final distillation step merges knowledge into a compact deployable model. Overfitting and memorisation are monitored per site with local held-out sets and canary tests.

## 2. Problem Statement

Clinical LLMs need domain data that hospitals legally cannot share. Each hospital alone has too little data (and overfits), and centralising data is a privacy and regulatory non-starter. Hospital hardware is modest, so full fine-tuning on-site is impossible.

## 3. Target Users & Personas

| Persona | Needs |
|---|---|
| Hospital consortium ML lead | Better clinical models without data sharing. |
| Privacy/compliance officer | Formal privacy guarantees and audit. |
| Clinician | Accurate summarisation/coding assistance. |

## 4. Goals

- Federated model within 3 pts of a (hypothetical) centralised model on clinical tasks.
- Formal DP guarantee (ε ≤ 8) per site.
- Each client trains on a single 24GB GPU.
- No memorised patient text (canary exposure below threshold).

## 5. Non-Goals

- Real-time clinical decision making without clinician review.
- Sharing raw data under any circumstances.

## 6. User Stories

- As consortium lead, I launch round 12 and 7 hospitals train locally overnight; the global adapter improves ICD coding F1.
- As compliance officer, I see the cumulative privacy budget ε per site and the audit of every update.
- As a site with rare specialties, my personalised adapter keeps local performance high.

## 7. Functional Requirements

| ID | Priority | Feature | Description |
|---|---|---|---|
| FR-01 | P0 | Federated orchestration | Rounds, client selection, straggler handling (Flower). |
| FR-02 | P0 | QLoRA clients | 4-bit NF4 base + LoRA; runs on one 24GB GPU. |
| FR-03 | P0 | Secure aggregation | Server sees only the sum of client updates (SecAgg). |
| FR-04 | P0 | Differential privacy | DP-SGD or update clipping + Gaussian noise; privacy accountant (RDP). |
| FR-05 | P1 | Heterogeneity handling | FedProx; personal adapters + shared global adapter. |
| FR-06 | P1 | Overfitting & memorisation monitors | Local held-out metrics; canaries; extraction tests before release. |
| FR-07 | P1 | Global distillation | Distil global + personal adapters' knowledge into a compact student using synthetic prompts. |
| FR-08 | P2 | Federated embeddings | Clinical embedding model trained federatedly for retrieval. |

## 8. Non-Functional Requirements

- Data never leaves site (verified by network policy).
- All updates signed and audited.
- Reproducible rounds (seeds, configs).

## 9. Success Metrics

| Metric | Target |
|---|---|
| Gap to centralised baseline (simulated) | ≤ 3 pts |
| Per-site ε | ≤ 8 (δ=1e-5) |
| Canary exposure | < threshold |
| Sites improved vs local-only training | 100% |

## 10. Release Plan

| Phase | Scope |
|---|---|
| M1 | Simulation on public clinical data (MIMIC-style, split into synthetic 'hospitals'). |
| M2 | QLoRA clients + FedAvg + DP. |
| M3 | Secure aggregation + personal adapters + monitors. |
| M4 | Distillation + pilot with 2–3 real sites. |

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| DP noise destroys utility | Tune clipping/noise, larger cohorts per round, LoRA's small parameter count helps. |
| Non-IID data slows convergence | FedProx, personalisation layers, careful client sampling. |
| Gradient inversion attacks | Secure aggregation + DP + minimum cohort size. |

## 12. Open Questions

- Aggregate A and B separately (standard) or aggregate the product BA (more faithful but larger)?
