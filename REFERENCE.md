# Project Reference: DLG Attack on Flower Federated Learning

## Project Overview

**Framework:** Flower (flwr >= 1.26.0) with PyTorch
**Dataset:** Fashion MNIST (grayscale, 28×28, 10 classes)
**Model:** Simple CNN (`Net` in `src/flower/task.py`)
**Aggregation:** FedAvg
**Partitioning:** IID across clients

---

## What Was Added: DLG Attack

### Files changed

| File | Change |
|---|---|
| `src/flower/dlg_attack.py` | **New.** Core DLG attack module. |
| `src/flower/server_app.py` | **Modified.** Snapshots initial weights, runs DLG after training if enabled. |
| `pyproject.toml` | **Modified.** Added `dlg-attack` and `dlg-steps` config options. |

---

## Concept: Deep Leakage from Gradients (DLG)

**Paper:** Zhu et al., 2019 — "Deep Leakage from Gradients"

**Core idea:** Gradients (or weight updates) a client sends to the server are not private. A curious/malicious server can work backwards from those gradients to reconstruct the client's original training images.

**The attack loop:**
1. Server observes the client's weight update `ΔW = w_local - w_global`
2. Server initializes random `dummy_data` and `dummy_label`
3. Server computes what gradient `dummy_data` would produce on the model
4. Server minimizes `||grad(dummy_data) - observed_gradient||²` using L-BFGS
5. After enough steps, `dummy_data` converges toward the real training image

---

## Key Approximation Used

This codebase does **not** capture per-client per-round gradients directly (Flower's `strategy.start()` only returns the final aggregated model). Instead:

```
ΔW = w_initial - w_final   (net update across all rounds, all clients)

Under SGD:  w_after = w_before - lr * grad
         =>  grad ≈ ΔW / lr
```

This is a **gradient proxy**, not the exact gradient. It is noisier than ideal because:
- It aggregates across multiple rounds (FedAvg blends all clients)
- Clients run multiple local epochs (each epoch adds noise to the signal)

**To get cleaner reconstructions** (closer to the original DLG paper): set `num-server-rounds = 1` and `local-epochs = 1` in `pyproject.toml`. This minimizes averaging and makes `ΔW` a closer approximation to the true single-batch gradient.

---

## Implementation Details

### `dlg_attack.py`

**`weight_delta_to_grads(w_before, w_after, lr, model)`**
- Takes two state_dicts and computes `(w_before[name] - w_after[name]) / lr` per parameter
- Returns a list of gradient tensors in the same order as `model.parameters()`
- Only iterates over `model.named_parameters()` (trainable params only, not buffers)

**`run_dlg_attack(model, true_grads, num_steps, device, save_dir)`**
- Initializes `dummy_data` as `torch.randn(1, 1, 28, 28)` (normalized space, mean=0, std=1)
- Initializes `dummy_label` as `torch.randn(1, 10)` — a soft label optimized jointly
- Uses `torch.optim.LBFGS` (standard choice from the DLG paper)
- Loss = L2 distance between dummy gradients and `true_grads`
- Uses `dummy_label.softmax(dim=-1)` so label is a valid probability distribution for CrossEntropyLoss
- Saves images every 50 steps to `dlg_results/step_XXXX.png`
- Final image saved to `dlg_results/final.png`
- Denormalization before saving: `img = img * 0.5 + 0.5` (reverses client's `Normalize((0.5,), (0.5,))`)

### `server_app.py`

Changes to `main()`:
1. Snapshots `w_initial` as a deep copy of the global model state_dict before `strategy.start()`
2. After `strategy.start()`, captures `w_final = result.arrays.to_torch_state_dict()`
3. If `context.run_config["dlg-attack"]` is `True`:
   - Loads `attack_model = Net()` initialized to `w_initial` (the pre-training model)
   - Converts `ΔW` to gradient proxy via `weight_delta_to_grads`
   - Calls `run_dlg_attack()`

### `pyproject.toml`

```toml
dlg-attack = false   # Set to true to enable the DLG attack after training
dlg-steps = 300      # Number of L-BFGS steps (more = better reconstruction, slower)
```

---

## Why the Attack Sits on the Server

The attack window is **after the server receives client updates, before FedAvg aggregation** (conceptually). Once gradients from multiple clients are averaged together, individual client signal is diluted and reconstruction becomes much harder.

In this implementation, it runs post-training as a retrospective analysis using the net weight drift `ΔW`. A more precise integration would require hooking into each round's per-client update, which the current Flower `strategy.start()` API does not expose directly.

---

## Why This Model / Dataset is Vulnerable

| Factor | Effect |
|---|---|
| Small CNN | Few parameters → gradient encodes a lot about each input |
| 28×28 grayscale | Low-dimensional input space → easier to invert |
| Fashion MNIST | Simple image distribution → reconstruction converges faster |
| SGD (no noise) | No differential privacy → gradient is a clean signal |
| IID partitioning | Clients have similar distributions, less obfuscation |

---

## Defenses (not implemented, for reference)

| Defense | Mechanism |
|---|---|
| Differential Privacy | Add calibrated noise to gradients before sending; Flower has DP strategies built in |
| Gradient compression | Sparsify gradient updates — less info for the attacker |
| Secure Aggregation | Server aggregates without ever seeing individual client updates |
| More local epochs | Gradient reflects a longer training trajectory, harder to invert |
| Larger batch size | Gradient is averaged over more samples, harder to reconstruct any one image |
