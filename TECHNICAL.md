# FedLoRA Health: Technical Specification

> Privacy-preserving federated fine-tuning in which hospitals collaboratively train LoRA adapters on local clinical notes without sharing data, using differential privacy, quantised clients and distillation into one global model.
>
> **Concepts:** `Fine Tuning`, `LoRA`, `Quantization`, `Overfitting`, `Embeddings`, `Knowledge Distillation`, `LLMs`

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| FL framework | Flower | Mature, framework-agnostic. |
| Training | PyTorch, PEFT, bitsandbytes | QLoRA. |
| Privacy | Opacus (DP-SGD), custom SecAgg | Guarantees. |
| Base models | Clinical-capable open LLM (e.g. Llama/Qwen 7–8B) | Quality. |
| Eval | Clinical NLP tasks (summarisation, ICD coding, NER) | Domain relevance. |

## 2. Repository Layout

```text
fedlora/
├── coordinator/    # flower server, strategies/, accountant.py
├── client/         # qlora_client.py, dp.py, local_eval.py, canaries.py
├── secagg/
├── personalize/
├── distill/
├── sim/            # partition public data into synthetic hospitals
└── compliance/     # audit export, model card
```

## 3. Data Models

```python
class RoundConfig(BaseModel):
    round: int
    clients_per_round: int
    local_epochs: int
    lr: float
    clip_norm: float
    noise_multiplier: float
    strategy: Literal["fedavg", "fedprox"]
    mu: float = 0.01                 # FedProx proximal term

class SiteReport(BaseModel):
    site_id: str
    round: int
    n_examples_bucket: str           # coarse bucket, not exact count
    heldout_loss: float
    task_f1: float
    epsilon_spent: float
    update_sha256: str
```

## 4. API Surface

| Method | Endpoint / Tool | Purpose |
|---|---|---|
| cli | fed sim --sites 8 --dataset clinical-notes-public | Simulated federation. |
| cli | fed coordinator start --config fed.yaml | Start coordinator. |
| cli | fed client join --site hospital-a | Site client. |
| cli | fed privacy report | ε ledger. |
| cli | fed distill --student 3B | Distil final model. |

## 5. Core Algorithms

### FedAvg on LoRA

`A_g ← Σ_k (n_k/n)·A_k`, `B_g ← Σ_k (n_k/n)·B_k`. The LoRA factors are small (MBs), so communication is cheap. Weighting uses noised or bucketed n_k to protect sizes.

### DP for updates

Clip each client's update Δ to norm C and add N(0, σ²C²I) (client-level DP), or use DP-SGD per example locally. An RDP accountant composes over rounds to report (ε, δ).

### FedProx

Local objective `L_k(w) + (μ/2)‖w − w_g‖²` limits client drift under non-IID data.

### Personalisation

Output = base + global adapter + personal adapter (the personal adapter is trained locally and never shared). This captures site-specific vocabulary and reduces local overfitting pressure on the global model.

## 6. Configuration

| Variable | Purpose |
|---|---|
| FED_COORDINATOR_URL | Coordinator |
| SITE_ID | Client identity |
| FED_CERT_PATH | mTLS certs |
| DP_TARGET_EPSILON | Budget |

## 7. Testing Strategy

- Simulation with known centralised upper bound.
- Privacy accountant unit tests vs reference.
- SecAgg correctness with dropouts.
- Membership inference + canary extraction tests on final model.

## 8. Performance Budgets

- Client round (8B QLoRA, 2k notes) < 2 h on 24GB GPU.
- Update size < 50 MB per round.

## 9. Local Setup

```bash
uv sync
uv run fed sim --sites 8
uv run pytest
uv run fed privacy report
```
