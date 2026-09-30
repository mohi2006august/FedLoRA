# FedLoRA Health

> Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model.

**Concepts:** `Fine Tuning`, `LoRA`, `Quantization`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LLMs`

Only LoRA adapter updates leave a site. This repository is a **numpy simulation** of the full pipeline on synthetic non-IID "hospitals" (see `BRAIN.md` decision log for why): frozen NF4-quantised base + LoRA (A/B) classifier, client-level DP (clip + Gaussian noise) with an RDP accountant, pairwise-masking secure aggregation with dropout recovery, FedAvg/FedProx, per-site personal adapters, canary/membership-inference monitors with a release gate, distillation into a compact student, federated embeddings, a signed hash-chained audit log and a model card.

## Quick start

```bash
pip install -e ".[dev]"     # or: uv sync
fed sim --sites 8           # simulate; writes runs/latest/ (summary, ledger, audit, model card)
fed privacy report          # epsilon ledger + audit-chain verification
fed distill --student 3B    # distil into a compact student (1B / 3B / 7B analogues)
fed coordinator start --config fed.yaml
pytest
```

`fed client join` validates site environment only; networked transport (Flower + mTLS) is not implemented.

## Status vs PRD

| FR | Feature | State |
|---|---|---|
| FR-01 | Orchestration, selection, stragglers | Done (in-process; no Flower) |
| FR-02 | QLoRA clients | Simulated NF4 base + LoRA (no GPU/bitsandbytes) |
| FR-03 | Secure aggregation | Pairwise masking + dropout recovery (no double masking) |
| FR-04 | Differential privacy | Client-level clip+noise, RDP accountant; no per-example DP-SGD |
| FR-05 | Heterogeneity | FedProx, personal + global adapters |
| FR-06 | Monitors | Held-out gap, canary exposure, membership inference, release gate |
| FR-07 | Distillation | Done on synthetic prompts |
| FR-08 | Federated embeddings | Done (linear, InfoNCE) |

**Known result:** with default settings the simulated federation does *not* yet meet the PRD's <= 3 pt gap to centralised under eps <= 8 (see `BRAIN.md` finding). No claim of improvement is made without running the sweeps.

## Docs

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, research, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, wireframes, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms, setup |
