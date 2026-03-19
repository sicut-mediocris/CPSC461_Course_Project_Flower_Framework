# Project Reference: DLG Attack on Flower Federated Learning

## Project Overview

**Framework:** Flower (flwr >= 1.26.0) with PyTorch
**Dataset:** Fashion MNIST (grayscale, 28×28, 10 classes)
**Model:** Simple CNN (`Net` in `src/flower/task.py`)
**Aggregation:** FedAvg
**Partitioning:** IID across clients

---

## What Was Added: DLG Attack

### Files changed (current / V2)

| File | Change |
|---|---|
| `src/flower/dlg_attack.py` | Core DLG module: `ndarrays_to_state_dict`, `weight_delta_to_grads`, `run_dlg_attack` |
| `src/flower/server_app.py` | `DLGFedAvg` subclass hooking into `configure_fit` + `aggregate_fit` for per-client per-round interception |
| `pyproject.toml` | `dlg-attack`, `dlg-steps`, `dlg-round` config options |

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

The true DLG paper uses the raw gradient from a single forward-backward pass on one batch. We don't have direct access to that — clients compute it internally and never send it explicitly. Instead we approximate:

```
ΔW_client = w_global - w_local_client   (per individual client, per round)

Under SGD:  w_after = w_before - lr * grad
         =>  grad ≈ ΔW_client / lr
```

This approximation degrades with:
- More local epochs (ΔW becomes a multi-step trajectory, not a single gradient)
- Larger batch sizes (gradient is averaged over more samples)

**For the cleanest reconstructions:** set `local-epochs = 1` and `batch-size = 1` in `pyproject.toml`. This makes ΔW a near-exact proxy for the true single-sample gradient, which is the ideal DLG scenario.

---

## Implementation Details (V2 — current)

### `dlg_attack.py`

**`ndarrays_to_state_dict(ndarrays, param_names)`**
- Converts Flower's flat list of numpy arrays into an `OrderedDict` keyed by parameter name
- `param_names` comes from `[name for name, _ in model.named_parameters()]`
- Needed because Flower strips parameter names when sending weights over the network

**`weight_delta_to_grads(w_before, w_after, lr, model)`**
- Takes two named state dicts and computes `(w_before[name] - w_after[name]) / lr` per parameter
- Returns a list of gradient tensors in the same order as `model.parameters()`
- Only iterates over `model.named_parameters()` (trainable params only, not buffers)

**`run_dlg_attack(model, true_grads, num_steps, device, save_dir)`**
- Initializes `dummy_data` as `torch.randn(1, 1, 28, 28)` (same normalized space as client data)
- Initializes `dummy_label` as `torch.randn(1, 10)` — a soft label optimized jointly with data
- Uses `torch.optim.LBFGS` (standard choice from the DLG paper)
- Loss = L2 distance between dummy gradients and `true_grads`
- Uses `dummy_label.softmax(dim=-1)` so label behaves as a probability distribution for CrossEntropyLoss
- Saves images every 50 steps; final image at `save_dir/final.png`
- Denormalization before saving: `img = img * 0.5 + 0.5` (reverses client's `Normalize((0.5,), (0.5,))`)

### `server_app.py`

**`DLGFedAvg(FedAvg)`** — subclasses the Flower FedAvg strategy with two hooks:

`configure_fit(server_round, parameters, client_manager)`:
- Flower calls this before each round, passing the current global `parameters`
- We snapshot them as `self._w_global_arrays` (flat numpy list)
- Calls `super().configure_fit()` so normal round setup continues

`aggregate_fit(server_round, results, failures)`:
- Flower calls this after all clients return, passing `results` as a list of `(ClientProxy, FitRes)`
- If `dlg_enabled` and this is the configured attack round:
  - For each `(_, fit_res)` in results: extract `w_local` via `parameters_to_ndarrays(fit_res.parameters)`
  - Convert both `w_global` and `w_local` to named dicts via `ndarrays_to_state_dict()`
  - Build `attack_model = Net()` initialized to `w_global`
  - Compute gradient proxy via `weight_delta_to_grads()`
  - Call `run_dlg_attack()` saving to `dlg_results/round_N_client_i/`
- Calls `super().aggregate_fit()` — FedAvg aggregation is unaffected

### `pyproject.toml`

```toml
dlg-attack = true    # Enable/disable the attack
dlg-steps = 300      # L-BFGS steps per client (more = better reconstruction, slower)
dlg-round = 1        # Which round to attack (0 = every round, N = only round N)
```

---

## Why the Attack Sits on the Server

The attack window is **after the server receives client updates, before FedAvg aggregation**. Once gradients from multiple clients are averaged together, individual client signal is diluted and reconstruction becomes much harder.

---

## V2: Fixing the DLG — From Bad to Correct (per-client, per-round)

### What was wrong with V1

The original implementation had a fundamental architectural flaw. It ran DLG **once, after all rounds finished**, using:

```
ΔW = w_initial - w_final
```

This is the net weight drift across **all clients and all rounds combined**. It is a terrible gradient signal because:

1. **FedAvg averaging destroys individual client information.** When 5 clients train and their updates are averaged, you can no longer tell which gradient came from which client. DLG needs a single client's signal, not a blend.
2. **Multiple rounds compound the blur.** After 10 rounds of training, `w_final` has drifted far from `w_initial` in a way that reflects the entire dataset across all clients, not any one person's private images.
3. **Multiple local epochs add trajectory noise.** Even within a single client, training for 3 epochs means `ΔW` represents a zig-zag optimization path, not a single clean gradient from one batch.

In short: V1 was running DLG on the noisiest, least informative signal possible.

---

### What the correct approach is

The teammate's feedback was exactly right:

> *"The DLG attack needs to be within the aggregation of the server, so from the gradients that the server gets from each individual client in each round it should try to reconstruct from that."*

The correct interception point is:

```
Round N:
  Server sends w_global → all clients
  Clients train locally → return w_local_client_0, w_local_client_1, ...
                                   ↑
                          [ATTACK HERE — before FedAvg mixes them]
  FedAvg aggregates → w_global for Round N+1
```

Per-client `ΔW_client = w_global - w_local_client_i` is a much cleaner signal because:
- It reflects **one client's** private data only
- It reflects **one round** of training (not accumulated across many)
- Before averaging, it hasn't been contaminated by other clients' gradients

---

### How V2 implements this: `DLGFedAvg`

Instead of running DLG as a one-shot post-training analysis, V2 subclasses `FedAvg` to hook into the aggregation lifecycle:

```
class DLGFedAvg(FedAvg):
    configure_fit()    ← called BEFORE each round (captures w_global)
    aggregate_fit()    ← called AFTER clients return, BEFORE averaging (runs DLG per client)
```

**`configure_fit(server_round, parameters, client_manager)`**
- Flower calls this before each round when it's about to send `w_global` to clients
- We intercept `parameters` (Flower's flat array format) and snapshot it as `self._w_global_arrays`
- Then call `super().configure_fit()` so normal behaviour continues

**`aggregate_fit(server_round, results, failures)`**
- Flower calls this after all clients have returned their updated weights
- `results` is a list of `(ClientProxy, FitRes)` tuples — **one per individual client**
- For each client:
  1. Extract `w_local` from `fit_res.parameters` using `parameters_to_ndarrays()`
  2. Convert both `w_global` and `w_local` from flat arrays to named state dicts via `ndarrays_to_state_dict()`
  3. Compute `ΔW_client / lr` as the gradient proxy via `weight_delta_to_grads()`
  4. Run `run_dlg_attack()` — saves to `dlg_results/round_N_client_i/`
- Then call `super().aggregate_fit()` so FedAvg averages normally — **DLG is purely observational**

---

### New helper: `ndarrays_to_state_dict`

Flower internally represents model parameters as a **flat list of numpy arrays** (one per layer), in the same order as `model.parameters()`. This has no key names. To compute per-layer deltas, we need to map these arrays back to named keys.

`ndarrays_to_state_dict(ndarrays, param_names)` does this using the parameter names from `model.named_parameters()`, producing an `OrderedDict` that `weight_delta_to_grads` can use.

---

### New config option: `dlg-round`

```toml
dlg-round = 1   # Attack only round 1 (0 = attack every round, N = attack round N)
```

Running DLG (300 steps) for every client in every round is prohibitively slow — e.g. 10 rounds × 5 clients × 300 steps = 15,000 L-BFGS steps. `dlg-round = 1` limits the attack to round 1 only, which is also the most informative round (model is freshest, gradients are least blurred by prior training).

---

### Files changed in V2

| File | Change |
|---|---|
| `src/flower/dlg_attack.py` | Added `ndarrays_to_state_dict()` helper to convert Flower's flat array format to named state dicts |
| `src/flower/server_app.py` | Replaced post-training DLG block with `DLGFedAvg` subclass hooking into `configure_fit` and `aggregate_fit` |
| `pyproject.toml` | Added `dlg-round = 1` config option |

---

### Output structure (V2)

```
dlg_results/
  round_1_client_0/
    step_0000.png
    step_0050.png
    step_0100.png
    ...
    final.png          ← reconstructed image for client 0
  round_1_client_1/
    ...
    final.png          ← reconstructed image for client 1
```

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
