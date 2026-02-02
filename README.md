# Beyond Patient Invariance: Learning Cardiac Dynamics via Action-Conditioned JEPAs

This repository contains the implementation of an **Action-Conditioned World Model** for cardiac monitoring. Traditional medical AI focuses on patient-invariance; this project shifts the paradigm toward modeling **disease as a dynamic transition vector** acting upon a patient's latent anatomical state.

This work is prepared for the **ICLR 2026 Workshop on World Models: Understanding, Modelling and Scaling**.

## 🌟 Key Concept
We adapt the **Joint-Embedding Predictive Architecture (JEPA)** to 12-lead ECG time-series. By predicting the future latent state $h_{t+1}$ given a current state $h_t$ and a pathology transition vector $a_t$, the model learns to disentangle stable anatomical features (Patient Identity) from dynamic pathological forces (Disease).

---

## 📂 Directory Structure

### `scripts/` (Execution Pipeline)
Command-line scripts for training, finetuning, and diagnostic:
*   **`dynamics_lejepa.py`**: Core pretraining scripts and implementation of the JEPA architecture.
*   **`probe_lejepa.py`**: Scripts for frozen-backbone evaluation, used to identify representation quality.
*   **`finetune_lejepa.py`**: Standard supervised adaptation of the pretrained world model.
*   **`baseline_supervised.py`**: Supervised learning from scratch for benchmarking.
*   **`eval.py`**: Comprehensive evaluation suite with 95% Confidence Intervals via bootstrap resampling.

### `src/` (Core Library)
The engine of the project, containing reusable modules and architectures:
*   **`dataset.py`**: Custom PyTorch Dataset for the **MIMIC-IV-ECG** longitudinal matched subset.
*   **`loss.py`**: Implementation of **SIGReg** (Sketched Isotropic Gaussian Regularization) and **Asymmetric Loss** for handling extreme class imbalance.

---

## 🚀 Quick Start

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Pretraining the World Model
To learn the cardiac dynamics via Action-Conditioned JEPA:
```bash
python scripts/dynamics_lejepa.py
python scripts/finetune_lejepa.py --checkpoint_path "checkpoints/lejepa_{}.pth"
python scripts/eval.py --checkpoint_path "checkpoints/lejepa_{}.pth"
```

### 3. Running the Linear Probe (1% Failure Mode Analysis)
To reproduce the finding that pretraining captures features even when finetuning collapses:
```bash
python scripts/probe_lejepa.py --checkpoint_path "checkpoints/lejepa_{}.pth" --data_fraction 0.01
```
