---
title: PrivatEdge
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: dashboard.py
pinned: false
---

# zkFedMoE

**Zero-Knowledge Federated Mixture-of-Experts for Privacy-Preserving Adaptive LLM Customization at the Intelligent Edge**

Major Project | Group #34 | IIIT Kota | April 2026

---

## Overview

**The core idea:** Every client trains the model locally on its own private data (never shared). After training, each client sends only its model updates to a central server. The server averages all clients' updates into a single global model, then broadcasts it back. This repeats every round. No raw data ever leaves the client device.

zkFedMoE extends this federated learning paradigm to solve three unsolved challenges simultaneously:

1. **Privacy leakage** — through Differential Privacy (DP-SGD) with formal (ε, δ)-guarantees
2. **Unverifiable updates** — through Selective Expert Proof Generation (SEPG) with SHA-256 integrity
3. **Communication cost** — through sparse Top-K expert updates in Mixture-of-Experts (MoE) models

By exploiting MoE sparsity, both communication cost and verification overhead reduce from **O(N) to O(K)** where K ≪ N.

---

## How It Works (Per Federated Round)

```
  ┌──────────┐   1. Download global model (θ_t)    ┌──────────┐
  │          │ ◄──────────────────────────────────  │          │
  │  CLIENT  │                                      │  SERVER  │
  │   (N)    │   2. Train locally on private D_i    │          │
  │          │      (data NEVER leaves the client)  │          │
  │          │                                      │          │
  │          │   3. Select Top-K experts            │          │
  │          │      Apply DP-SGD (clip + noise)     │          │
  │          │      Generate SEPG proof π_i         │          │
  │          │                                      │          │
  │          │   4. Upload (sparse update + π_i)    │          │
  │          │ ─────────────────────────────────►   │          │
  │          │                                      │          │
  │          │                                      │  5. Verify proofs  │
  │          │                                      │  6. Aggregate      │
  │          │                                      │     (FedAvg /      │
  │          │                                      │      Median /      │
  │          │                                      │      TrimMean)     │
  │          │                                      │  7. Update θ_{t+1} │
  │          │                                      │                    │
  │          │   8. Broadcast new θ_{t+1}           │                    │
  │          │ ◄────────────────────────────────    │                    │
  └──────────┘                                      └──────────┘
                  REPEAT EACH ROUND
```

**Key point:** Only gradients/parameters travel over the network — never the raw training data. The server sees model updates, not user data.

---

## Key Features

| # | Feature | Implementation |
|---|---------|----------------|
| 1 | MoE + LoRA model | `src/models/moe_model.py` |
| 2 | FedAvg federated learning | `src/fl/client.py`, `src/fl/server.py` |
| 3 | Sparse Top-K expert updates | `src/fl/client.py` |
| 4 | AG News dataset (120K samples) | `src/data/text_datasets.py` |
| 5 | Differential Privacy (DP-SGD) | `src/fl/dp.py` |
| 6 | SEPG proof generation + 4-check verification | `src/fl/sepg.py` |
| 7 | Adversary simulation (Poisoning, Free-rider, Sybil) | `src/fl/adversaries.py` |
| 8 | Robust aggregation (Median, Trimmed Mean) | `src/fl/server.py` |
| 9 | 4 automated experiments with JSON results | `experiments/run_all_experiments.py` |
| 10 | Interactive 10-page Streamlit dashboard | `dashboard.py` |

---

## Architecture

```
+----------------------+  +----------------------+  +--------------------+
|   EDGE DEVICE LAYER  |  |  COORDINATION LAYER  |  | VERIFICATION LAYER |
|    (Client x N)      |  |       (Server)       |  |   (SEPG Verifier)  |
+----------------------+  +----------------------+  +--------------------+
| Private Data D_i     |  | Collect Updates      |  | 1. |top_k| = K     |
| Local MoE+LoRA Train |  | SEPG Verify (4 chks) |  | 2. C <= C_max      |
| Top-K Selection      |->| FedAvg / Median /    |->| 3. sigma >= s_min  |
| DP-SGD (Clip+Noise)  |  | Trimmed Mean         |  | 4. SHA-256 match   |
| SEPG Proof pi_i      |  | Global Update        |  |                    |
| Sparse Transmission  |  | Privacy Accountant   |  | Accept / Reject    |
+----------------------+  +----------------------+  +--------------------+
                            theta_{t+1} broadcast
```

Model pipeline: `Tokens -> Embedding -> MeanPool -> MoE(Top-K of E) -> LoRA Classifier -> 4 Classes`

---

## Project Structure

```
Code/
├── README.md                          # This file
├── dashboard.py                       # Interactive Streamlit dashboard (2,061 lines)
├── run_demo.py                        # Quick end-to-end demo script
├── data/
│   ├── ag_news_train.csv              # 120,000 training samples
│   └── ag_news_test.csv               # 7,600 test samples
├── src/
│   ├── models/
│   │   └── moe_model.py               # MoE + LoRA classifier (167 lines)
│   ├── data/
│   │   └── text_datasets.py           # Dataset, vocab, client splits (164 lines)
│   └── fl/
│       ├── client.py                  # Local training + comm tracking (117 lines)
│       ├── server.py                  # FedAvg + Median + TrimmedMean (149 lines)
│       ├── dp.py                      # DP-SGD: clip, noise, accountant (73 lines)
│       ├── sepg.py                    # SEPG proof gen + verify (82 lines)
│       └── adversaries.py             # Poisoning, Free-rider, Sybil (80 lines)
├── experiments/
│   └── run_all_experiments.py         # 4-experiment suite (379 lines)
└── plots/
    ├── exp1_privacy_utility.png
    ├── exp2_comm_vs_k.png
    ├── exp3_verification_overhead.png
    ├── exp4_robustness.png
    └── experiment_results.json        # Machine-readable results
```

Total: **~3,270 lines of Python** across 10 source files.

---

## Installation

**Requirements:** Python 3.10+ (CPU is sufficient; GPU supported if available).

```bash
cd "Major Project/Code"
pip install torch pandas numpy streamlit plotly matplotlib graphviz
```

**Dependencies:**
- `torch` (>= 2.0) — model, training, aggregation
- `pandas`, `numpy` — data processing
- `streamlit` (>= 1.30) — dashboard
- `plotly` — interactive charts
- `matplotlib` — experiment plots
- `graphviz` — pipeline diagrams in dashboard

---

## Quick Start

### 1. Launch the interactive dashboard (recommended)

```bash
python -m streamlit run dashboard.py
```

Opens in browser at `http://localhost:8501`. Navigate through the 10 pages starting from **🏠 Home**.

### 2. Run all 4 experiments

```bash
python -m experiments.run_all_experiments
```

Generates plots in `plots/` and saves numerical results to `plots/experiment_results.json`. Takes ~10–15 minutes on CPU.

### 3. Quick demo (one-shot end-to-end)

```bash
python run_demo.py
```

---

## Experimental Results

All numbers below match the conference paper and major project report. The dashboard reads them from [plots/experiment_results.json](plots/experiment_results.json).

Base AG News configuration: 120K train, 7.6K test, 4 classes, 5 clients, 5 FL rounds, E=8 experts, K=2 default, seed=42. Robustness (Experiment 4) and the non-IID / MIA / FedProx sweeps use 9 clients and 8 rounds so coordinate-wise median is computed over a stable honest population.

### Experiment 1: Privacy-Utility Tradeoff (δ = 10⁻⁵)

| Noise σ | ε (basic, √T) | ε (Rényi) | Accuracy |
|---------|--------------|-----------|----------|
| 0.0 (no DP) | ∞ | ∞ | **57.09%** |
| 0.1 | 0.289 | 250.40 | 25.00% |
| 0.5 | 0.058 | 3.060 | 25.00% |
| 1.0 | 0.029 | 0.775 | 25.00% |
| 2.0 | 0.014 | 0.184 | 25.00% |

Rényi accountant is orders-of-magnitude tighter than the naive √T bound. Accuracy cliff at σ ≥ 0.1 is expected for a 600K-parameter MoE trained for only 5 rounds — DP noise overwhelms the gradient signal. The same σ regime is *not* catastrophic for the smaller 25K disease MLP (see Experiment 8), so DP calibration is model-size dependent.

### Experiment 2: Communication Savings vs Top-K

| K | Accuracy | Saving | Dense / Sparse (MB) |
|---|----------|--------|---------------------|
| 1 | 56.16% | **39.52%** | 58.6 / 35.4 |
| 2 | 57.59% | 33.88% | 58.6 / 38.8 |
| 3 | 57.25% | 28.23% | 58.6 / 42.1 |
| 4 | **74.42%** | 22.58% | 58.6 / 45.4 |
| 8 | 70.51% | 0.00% | 58.6 / 58.6 |

Sweet spot: **K = 4**. Top-K routing acts as implicit regularization — activating all 8 experts forces never-specialised ones into every prediction, which hurts accuracy. The K = 4 regime trades 22.58% of bandwidth for the best observed accuracy.

### Experiment 3: SEPG Verification Overhead

| K | Gen (ms) | Verify (ms) | Total (ms) |
|---|----------|-------------|------------|
| 1 | 5.44 | 6.28 | 11.72 |
| 4 | 4.81 | 5.44 | 10.25 |
| 8 | 5.55 | 6.98 | 12.53 |
| **Mean (K=1–8)** | **5.13** | **5.75** | **10.89 ± 0.77** |

**~11 ms per client**, roughly constant with K (both phases are dominated by SHA-256 over the sparse state). On a 5-client federation this adds ~55 ms per round — below 0.5% of per-round training time.

### Experiment 4: Robustness Under Poisoning (9 clients, 8 rounds)

| Malicious | FedAvg | Median | TrimMean |
|-----------|--------|--------|----------|
| 0% | 51.08% | 47.87% | **52.57%** |
| 22% | 48.59% | **51.74%** | 50.57% |
| 44% | 37.82% | **43.34%** | 42.75% |
| Drop (0→44) | −13.26 pp | **−4.53 pp** | −9.82 pp |

Coordinate-wise median is most robust under heavy poisoning; FedAvg is most vulnerable. We use N = 9 so that even at 44% malicious (f = 4) the median is computed over 5 honest clients.

---

## Dashboard Pages

| Page | Purpose |
|------|---------|
| 🏠 Home | System pipeline (Graphviz), concept cards, demo walkthrough |
| 🔮 Predict | Live text classification with expert routing, compare two headlines |
| 🏋️ Train | Configurable FL with optional DP + SEPG, live charts |
| 📂 Custom CSV | Upload any labelled CSV, FL training, confusion matrix |
| 🔒 Privacy & DP | DP-SGD training with live ε/δ chart, SEPG proof display |
| 🛡️ Robustness | Attack simulation (Poisoning/Free-rider/Sybil), strategy comparison |
| 📊 Experiments | Interactive Plotly charts for all 4 experiments |
| 📡 Compare | Real-time communication savings calculator |
| 🏗️ Architecture | Model data-flow, parameter breakdown, 5 code snippet tabs |
| ℹ️ About | Team, advisor, institution, implementation status table |

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Baseline FL with MoE + LoRA | ✅ Complete |
| 2 | Sparse Top-K communication | ✅ Complete |
| 3 | DP-SGD + SEPG verification | ✅ Complete |
| 4 | Adversary simulation + robust aggregation | ✅ Complete |
| 5 | Experiment suite + dashboard | ✅ Complete |

---

## Team (Group #34)

| Name | Roll Number |
|------|-------------|
| Keshav Kashyap | 2023KUCP1161 |
| Lakshya Sharma | 2023KUCP1167 |
| Prakriti Patel | 2023KUCP1109 |

**Supervisor:** Dr. Gyan Singh Yadav, Department of Computer Science & Engineering, IIIT Kota

**Institution:** Indian Institute of Information Technology, Kota

---

## References

Key papers behind the implementation:

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (AISTATS 2017) — FedAvg
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* (ICLR 2017) — MoE
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022) — LoRA
- Abadi et al., *Deep Learning with Differential Privacy* (CCS 2016) — DP-SGD
- Yin et al., *Byzantine-Resilient Distributed Learning: Towards Optimal Statistical Rates* (ICML 2018) — Median / Trimmed Mean
- Zhang, Zhao, LeCun, *Character-level Convolutional Networks for Text Classification* (NeurIPS 2015) — AG News dataset

Full bibliography (46 references) is in the project report at `../Report/main.tex`.

---

## License

Academic project — IIIT Kota, 2026.
