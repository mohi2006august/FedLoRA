# CLAUDE.md: FedLoRA Health

Guidance for Claude Code (and other AI assistants) working in this project.

## Project Summary

Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model. FedLoRA Health runs federated learning where only LoRA adapter updates, never data, leave each site. Clients fine-tune a 4-bit quantised base (QLoRA) on local notes, add differential-privacy noise and clip updates, and a secure aggregator combines them (FedAvg/FedProx over LoRA A/B factors). Heterogeneous sites are handled with per-site personal adapters plus a shared global adapter. A final distillation step merges knowledge into a compact deployable model. Overfitting and memorisation are monitored per site with local held-out sets and canary tests.

## Read First

- `PRD.md`: what we are building and why; requirement IDs (FR-xx) are referenced in commits.
- `SYSTEM_ARCHITECTURE.md`: components, data flow and failure modes.
- `TECHNICAL.md`: stack, data models, APIs and algorithms.
- `DESIGN.md`: UX flows, screens and visual tokens.
- `BRAIN.md`: concept primers, experiments and the decision log.

## Commands

| Command | What it does |
|---|---|
| `uv sync` | Install |
| `uv run fed sim --sites 8` | Simulate |
| `uv run pytest` | Tests |
| `uv run fed privacy report` | Privacy ledger |

## Code Conventions

- No code path may serialise raw text off-site; enforced by tests and network policy.
- Keep functions small and typed; no silent `except`. Log with structured fields.
- Every new feature references its FR-ID in the PR description and ships with tests.
- Secrets only come from environment variables. Never hard-code keys or commit `.env`.

## Gotchas

- Aggregating A and B separately is an approximation of averaging BA. Document the choice.
- Report only bucketed dataset sizes.

## When Unsure

- Prefer the simplest design that satisfies the PRD; record bigger trade-offs in `BRAIN.md` → Decision Log.
- Run the eval / benchmark suite before claiming a quality or performance improvement.
- Ask before adding a new paid dependency, external service or data store.
