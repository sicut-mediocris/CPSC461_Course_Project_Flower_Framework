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