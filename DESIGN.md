# FedLoRA Health: Design Document

> Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model.
>
> **Concepts:** `Fine Tuning`, `LoRA`, `Quantization`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LLMs`

## 1. Design Principles

- Privacy guarantees are shown as numbers, not promises.
- Clinicians review; the model assists.

## 2. Key User Flows

### Run a round

1. Coordinator schedules round.
2. Sites receive notification.
3. Local training overnight.
4. Morning: global metrics + ε update.

## 3. Screens & Surfaces

| Screen | What it shows / does |
|---|---|
| Federation dashboard | Rounds, participating sites, global metric curve. |
| Privacy ledger | ε per site over rounds vs budget. |
| Site console (on-prem) | Local metrics, data never shown to coordinator. |
| Release checklist | Memorisation tests, model card, approvals. |

## 4. Wireframe (Primary Screen)

```text
Round 12/30   sites 7/9 participating   global ICD-F1 0.781 (local-only avg 0.702)
Privacy ε (δ=1e-5)   A 3.1 ▇▇▇░░░  B 3.4 ▇▇▇░░░  C 5.8 ▇▇▇▇▇░ ⚠  budget 8.0
Overfitting: local held-out loss ↑ at site C (small data) → personal adapter only
Canary exposure (last release): max 4.2 bits ✔ (< 10)
```

## 5. Visual Language

| Token | Value |
|---|---|
| Primary accent | `#0284C7` (medical blue) |
| Neutral surface | `#0F1117` dark / `#FAFAF9` light |
| Success / Warning / Danger | `#22C55E` / `#F59E0B` / `#EF4444` |
| Type | Inter for UI, JetBrains Mono for code, tokens and numbers |
| Spacing scale | 4 / 8 / 12 / 16 / 24 / 32 px |
| Radius | 8px cards, 6px inputs, full for pills |

## 6. States

| State | Treatment |
|---|---|
| Budget nearly exhausted | Amber on site. |
| Straggler | Grey site with timeout countdown. |

## 7. Accessibility

- WCAG 2.2 AA contrast in both themes; never encode meaning with colour alone (pair with icon or label).
- Full keyboard navigation with visible focus rings; `/` focuses search, `?` opens shortcuts.
- Charts ship with a data-table fallback and `aria-describedby` summaries.
- Respect `prefers-reduced-motion`; streaming text is announced politely via `aria-live`.
