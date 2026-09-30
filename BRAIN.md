# FedLoRA Health: Brain

> Second brain for this project: concepts, mental models, research, ideas and decisions. Keep it messy and alive.

## 1. Why This Project

It combines parameter-efficient fine-tuning, quantisation for modest hardware, privacy engineering, overfitting control on small local datasets and distillation, applied to a high-stakes domain.

## 2. Concept Primers

### Fine Tuning

Continuing training of a pre-trained model on task- or domain-specific data to change its behaviour, style or knowledge.

- **SFT**: Supervised fine-tuning on prompt → ideal response pairs.
- **DPO / preference tuning**: Optimising directly on chosen-vs-rejected pairs without a separate reward model.
- **Catastrophic forgetting**: Losing general capabilities while specialising.
- **PEFT**: Parameter-efficient methods (LoRA, adapters, prefix tuning) that train a small fraction of weights.

> **Mental model:** Fine-tune for form (format, tone, task behaviour). Retrieve for facts. Data quality beats data quantity almost every time.

### LoRA

Low-Rank Adaptation: freeze the base weights W and learn a low-rank update ΔW = B·A (rank r ≪ d), training well under 1% of parameters with small, swappable adapters.

- **Rank r / alpha**: Capacity of the update and its scaling (alpha/r).
- **Target modules**: Which projections get adapters (q,k,v,o, MLP).
- **QLoRA**: LoRA on top of a 4-bit quantised base model; fine-tune big models on one GPU.
- **Multi-LoRA serving**: One base model in memory serving many adapters per request (S-LoRA, Punica).

> **Mental model:** The base model is the OS and adapters are apps. They are cheap to train, store and switch, so give each tenant, task or style its own adapter.

### Quantization

Representing weights (and sometimes activations/KV cache) with fewer bits (int8, int4, fp8) to cut memory and increase speed, with some accuracy loss.

- **PTQ vs QAT**: Post-training quantisation vs quantisation-aware training.
- **GPTQ / AWQ**: Weight-only int4 methods using second-order info or activation-aware scaling.
- **GGUF**: llama.cpp file format with k-quants (Q4_K_M, Q5_K_M, …) for CPU/edge inference.
- **Outliers**: A few large-magnitude activation channels that break naive quantisation.

> **Mental model:** Memory bandwidth, not FLOPs, bounds LLM decoding. Halving bytes per weight roughly doubles tokens/sec until you hit compute limits.

### Overfitting

When a model fits noise or specifics of the training set so closely that it performs worse on unseen data; train loss keeps dropping while validation loss rises.

- **Generalisation gap**: Difference between training and validation/test performance.
- **Regularisation**: Weight decay, dropout, data augmentation, early stopping, smaller capacity.
- **Memorisation**: Verbatim reproduction of training examples; measured with canaries or extraction probes.
- **Data leakage**: Test information leaking into training, which hides overfitting behind inflated scores.

> **Mental model:** A low training loss is a claim. The held-out set is the evidence. Always keep a test set you never tune against.

### Embeddings

Dense vector representations where geometric closeness reflects semantic similarity; the backbone of search, clustering, recommendation and RAG.

- **Contrastive training**: Pulling positive pairs together and pushing negatives apart (e.g. InfoNCE).
- **Matryoshka embeddings**: Vectors whose prefixes are themselves usable lower-dimensional embeddings.
- **Anisotropy**: Embeddings crowding into a narrow cone, which hurts cosine similarity spread.
- **Embedding drift**: Distribution shift in vectors over time or after a model upgrade.

> **Mental model:** An embedding model is an opinion about what 'similar' means. Benchmarks like MTEB are a starting point, so evaluate on your own queries.

### Knowledge Distillation

Training a small 'student' model to mimic a large 'teacher' model's outputs (soft labels, logits, rationales), keeping most of the quality at a fraction of the cost.

- **Soft targets**: Teacher probability distributions softened with temperature T carry 'dark knowledge'.
- **KD loss**: KL divergence between teacher and student distributions, mixed with hard-label cross-entropy.
- **Sequence-level KD**: Student trained on teacher-generated outputs (synthetic data distillation).
- **Capacity gap**: Very large teacher/student gaps can hurt; teacher assistants bridge them.

> **Mental model:** You are compressing a function, not a dataset. The teacher's uncertainty is signal, so preserve it.

### LLMs

Large Language Models: transformer networks trained on large text corpora to predict the next token, then aligned (SFT and RLHF/RLAIF) to follow instructions and use tools.

- **Prompting**: System prompts, few-shot examples, structured outputs, tool use.
- **Sampling**: Temperature, top-p and max tokens shape determinism and length.
- **Evals**: Task-specific test sets plus LLM-as-judge with calibrated rubrics.
- **Routing**: Sending each request to the cheapest model that meets its quality bar.

> **Mental model:** Treat the LLM as a probabilistic component. Wrap it with evals, guardrails, retries and observability, the same way you would a flaky network dependency.

## 3. What I Want to Learn

- [ ] Federated learning algorithms.
- [ ] Differential privacy accounting.
- [ ] Secure aggregation.
- [ ] Personalisation under non-IID data.

## 4. Reading & Research List

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg, 2017)
- Li et al., *FedProx* (2020)
- Abadi et al., *Deep Learning with Differential Privacy* (2016)
- Bonawitz et al., *Practical Secure Aggregation* (2017)
- Zhang et al., *FedPETuning / federated LoRA* (2023)

## 5. Experiments to Run

| # | Hypothesis | How to test |
|---|---|---|
| 1 | Personal + global adapters beat global-only on small sites | Per-site F1. |
| 2 | ε=8 costs < 3 pts vs non-DP | Sweep noise multipliers. |
| 3 | FedProx μ=0.01 speeds convergence under non-IID | Rounds to target. |

## 6. Ideas Backlog

- Federated evaluation benchmark service.
- Federated RLHF with clinician feedback.
- Cross-silo federated embeddings for clinical search.

## 7. Decision Log

| Date | Decision | Reason |
|---|---|---|
| YYYY-MM-DD | LoRA-only communication | Tiny updates make secure aggregation and DP practical. |
| 2026-09-30 | Simulation is pure numpy (linear softmax classifier with frozen NF4 base + LoRA), not PyTorch/Flower/Opacus | Keeps the privacy, SecAgg and aggregation logic testable on a laptop; the algorithms transfer to PEFT/Flower. Network transport (`fed client join`) is intentionally unimplemented. |
| 2026-09-30 | Aggregate LoRA A and B separately by default; `aggregation: product` averages BA and re-factorises via SVD | Answers the PRD open question: factors are SecAgg-compatible and cheap; product is more faithful but the server sees individual updates. Tested that product error <= factor error. |
| 2026-09-30 | Client-level DP: clip to C, each client adds N(0, sigma^2 C^2 / K) so the SecAgg sum carries N(0, sigma^2 C^2); uniform weights under DP | Sizes are not leaked and sensitivity of the mean is C/K. RDP accountant does not credit sampling amplification (conservative). |
| 2026-09-30 | Both accountings kept: `participation` (default, valid vs coordinator) and `amplified` (Poisson sampling credited, hidden-sampling adversary only) | Amplification requires the adversary not to know who was sampled; the coordinator does. Amplified mode is opt-in and always prints the coordinator-visible eps. |
| 2026-09-30 | Decaying server step under DP (1 / (1 + 0.5 (r-1))) | DP noise is a random walk; at 8 sites F1 went 0.58 -> 0.82 with decay. Hurts slightly at 32 sites (0.90 -> 0.82). |
| 2026-09-30 | Distillation prompts are mixups of a public pool; personal adapters label prompts locally | No real notes or personal adapters leave a site. Student rank must be >= ~6 for 8 classes (rank 3 could not separate them). |

## 8. Scratchpad

- **Sweep results (2026-10-01, `fed sweep`, 5 seeds (3 at 32 sites), eps <= 8, delta 1e-5, 400-note held-out sets, fair local-only baseline with the same rank/epochs/selection):**

  | Setup | Gap to centralised | Sites beating local-only |
  |---|---|---|
  | 8 sites, no DP | 5.7 pts | 85% |
  | 8 sites, client-level DP, decay 0.5 | 6.7 pts | 60% |
  | 8 sites, client-level DP, decay 0 | 17.3 pts | 42% |
  | 8 sites, Poisson + amplified accounting | 9.3 pts | 70% |
  | 32 sites, client-level DP, decay 0 | **2.1 pts** (meets <= 3) | 82% |
  | 32 sites, client-level DP, decay 0.5 | 4.9 pts | 70% |

  - Targets still missed: gap <= 3 pts at 8 sites; "100% of sites improved". A site with lots of data gains little from a noisy global adapter, so 100% is not a sensible target; ~60-85% is what this simulation gives.
  - Earlier single-seed numbers in this log (gap 1.8 at 32 sites, 2.8 amplified at 8 sites) came from a 60-note held-out set where per-site F1 comparisons were mostly noise; they are superseded by the table above.
  - Personalisation: each site chooses (global scale in {0, .25, .5, .75, 1}, epochs) on a local validation split (rank-2 personal adapter). Selection matters most under DP; the noisy global adapter is often down-weighted.
  - Rank 1 global adapter beats rank 2 under DP (noise grows with parameter count). Lr 1.0 / 3 local epochs beat lower lr / fewer epochs by a wide margin.
  - FedProx mu 0.01 vs 0 and mu 0.1: no meaningful difference on these settings.
- **Sampling amplification (added):** `--sampling poisson --accounting amplified` uses the sampled-Gaussian RDP (matches Opacus 1.6.0 to 1e-6). **Caveat:** the coordinator picks the cohort, so against the coordinator the honest number is the participation ledger; `fed privacy report` prints both. The gap with amplified accounting at 8 sites is in the table above (it did not reach the target).
- **Per-example DP-SGD (added, `--dp-mode sample`):** record-level guarantee via per-example clipping + Poisson batches; eps composes over all local steps (sampled-Gaussian RDP, no client-sampling amplification). 8 sites, 10 rounds, seed 0: F1 0.81 (client-level: 0.82), worst eps 7.7, one small site excluded at round 6 because its batch rate q is larger. The two guarantees differ: client-level hides a site's whole contribution; sample-level hides each note but not participation.
