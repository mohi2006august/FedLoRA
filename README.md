# FedLoRA Health

> Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model.

**Concepts:** `Fine Tuning`, `LoRA`, `Quantization`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LLMs`

Only LoRA adapter updates leave a site. This repository is a **numpy simulation** of the full pipeline on synthetic non-IID "hospitals" (see `BRAIN.md` decision log for why): frozen NF4-quantised base + LoRA (A/B) classifier, client-level DP (clip + Gaussian noise) with an RDP accountant, pairwise-masking secure aggregation with dropout recovery, FedAvg/FedProx, per-site personal adapters, canary/membership-inference monitors with a release gate, distillation into a compact student, federated embeddings, a signed hash-chained audit log and a model card.

## Quick start

```bash
pip install -e ".[dev]"     # or: uv sync
fed sim --sites 8           # simulate (try --sites 32 --clients-per-round 32); writes runs/latest/ (summary, ledger, audit, model card)
fed privacy report          # epsilon ledger + audit-chain verification
fed distill --student 3B    # distil into a compact student (1B / 3B / 7B analogues)
fed coordinator start --config fed.yaml
fed sweep --sites 32 --rounds 10 --server-lr-decay 0,0.5   # hyper-parameter grid
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

**Known results** (`fed sweep`, seeds averaged, eps <= 8; details in `BRAIN.md`): at 32 sites with DP the gap to centralised is ~1.8 pts (meets the <= 3 target) but only ~70% of sites beat local-only; at 8 sites the gap is ~11 pts because DP noise dominates a small cohort. The "100% of sites improved" target is not met.

## Docs

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, research, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, wireframes, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms, setup |
