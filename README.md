# 🛡️ Defending Against Data Reconstruction Attacks in Federated Learning

[![Framework: Flower](https://img.shields.io/badge/Framework-Flower%201.14+-blue)](https://flower.ai)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org)
[![Security: DP](https://img.shields.io/badge/Security-Differential%20Privacy-green)](https://opacus.ai)

This project is a research-focused implementation of Federated Learning (FL) using **Flower** and **PyTorch**. It adapts the standard Flower "Quickstart" to study **Data Reconstruction Attacks (DRA)**—specifically Gradient Inversion—and evaluates the effectiveness of client-side defenses.

---

## 📂 Project Structure

This repository uses a **Source (src) Layout**. This modular structure keeps the core Federated Learning logic separate from research notebooks and adversarial scripts, allowing for multiple implementations (e.g., PyTorch vs. JAX) in the future.

```text
.
├── src/                    # 🧠 CORE FLOWER APPS
│   ├── flower/             # Primary implementation (PyTorch + Security)
│   │   ├── __init__.py
│   │   ├── client_app.py   # Client logic & Defense injections
│   │   ├── server_app.py   # Server logic & Strategy
│   │   └── task.py         # CNN Model & CIFAR-10 data loading
│   ├── pyproject.toml      # ⚙️ Project metadata & experiment configs
├── notebooks/              # 🧪 RESEARCH & VISUALIZATION
├── attacks/                # ⚔️ ADVERSARIAL SCRIPTS
│   └── dlg_attack.py       # Gradient Inversion (DLG) implementation
├── .gitignore              # Research-ready git exclusions
└── README.md               # You are here
```

## 🚀 Running the Flower Implementation

To run the federated learning simulation, you must navigate into the `src` directory where the configuration and source code reside.

### Navigate to the Source Folder
```bash
cd src
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `flower` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> This example runs faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in your Flower Configuration. Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
# Run with the default federation (CPU only)
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
```

---

## ⚔️ Running the DLG Attack Demo

The `attacks/dlg_attack.py` module implements the **Deep Leakage from Gradients (DLG)** attack ([Zhu et al., 2019 — arxiv:1906.08935](https://arxiv.org/abs/1906.08935)). Given only the gradient update that a client would send to the server, it attempts to reconstruct the client's private training images.

### How it works

```
Global model weights (sent by server)
         │
         ▼
Client trains on private image → computes gradients
         │
         ▼  ← adversary intercepts here
DLG attack: random dummy data → optimise until dummy gradients ≈ real gradients
         │
         ▼
Reconstructed private image
```

### Run the stand-alone demo

From the **repo root**:

```bash
# Install src package first (needed for the model + data loader)
pip install -e src/

# Run the attack demo
python -m attacks.dlg_attack
```

The demo will:
1. Load the CNN model and a single Fashion-MNIST training sample.
2. Capture the gradient update as an adversary would.
3. Run 300 optimisation iterations to reconstruct the image.
4. Print the reconstructed label and gradient loss at each 50-step checkpoint.

### Integration with `ClientApp`

`client_app.py` automatically imports `capture_gradients` from `dlg_attack.py` when the `attacks` directory is available. During each training round it captures the first-batch gradients **before** local training begins, storing them in `captured_gradients`. You can pass these directly to `run_dlg_attack()` for reconstruction experiments without modifying the FL training loop.