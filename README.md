# Defending Against Data Reconstruction Attacks in Federated Learning

[![Framework: Flower](https://img.shields.io/badge/Framework-Flower%201.14+-blue)](https://flower.ai)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org)
[![Security: DP](https://img.shields.io/badge/Security-Differential%20Privacy-green)](https://opacus.ai)

This project is a research-focused implementation of Federated Learning (FL) using **Flower** and **PyTorch**. The core goal is to study **Data Reconstruction Attacks (DRA)**, where an "honest-but-curious" server attempts to recover private training images from shared gradients and evaluate the effectiveness of different client-side defenses.
---

## 📂 Project Structure

This repository separates the production-grade FL simulation from research notebooks and adversarial scripts.

```text
.
├── src/                             # 📦 ALL IMPLEMENTATIONS
│   ├── pytorchexample/              # Current PyTorch implementation
│   |   ├── __init__.py
│   │   ├── client_app.py
│   │   ├── server_app.py            # Server logic + Strategy (FedAvg/FedAdagrad)
│   │   └── task.py                  # CNN Model, training loops, & data loading
├── notebooks/                       # 🧪 RESEARCH & VISUALIZATION
│   ├── 01_data_analysis.ipynb       # Visualizing CIFAR-10 partitions
│   └── 02_attack_results.ipynb      # DLG Attack vs. Defended Gradients
├── attacks/                         # ⚔️ ADVERSARIAL SCRIPTS
│   └── dlg_attack.py                # Implementation of Deep Leakage from Gradients
├── pyproject.toml                   # ⚙️ Experiment parameters & dependencies
└── README.md                        # Documentation
└── .gitignore                       # Ignore __pycache__, data/, and .venv