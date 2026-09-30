# FedLoRA Health

> Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model.

**Concepts:** `Fine Tuning`, `LoRA`, `Quantization`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LLMs`

FedLoRA Health runs federated learning where only LoRA adapter updates, never data, leave each site. Clients fine-tune a 4-bit quantised base (QLoRA) on local notes, add differential-privacy noise and clip updates, and a secure aggregator combines them (FedAvg/FedProx over LoRA A/B factors). Heterogeneous sites are handled with per-site personal adapters plus a shared global adapter. A final distillation step merges knowledge into a compact deployable model. Overfitting and memorisation are monitored per site with local held-out sets and canary tests.

## Docs

| File | Contents |
|---|---|
| [BRAIN.md](BRAIN.md) | Concept primers, research, experiments, decision log |
| [CLAUDE.md](CLAUDE.md) | Instructions for AI coding assistants |
| [DESIGN.md](DESIGN.md) | UX flows, screens, wireframes, visual language |
| [PRD.md](PRD.md) | Problem, users, requirements, metrics, roadmap |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment, failure modes |
| [TECHNICAL.md](TECHNICAL.md) | Stack, data models, APIs, algorithms, setup |
