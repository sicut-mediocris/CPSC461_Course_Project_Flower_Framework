"""Deep Leakage from Gradients (DLG) attack — Zhu et al., 2019.

The attack reconstructs private client training data from gradient updates.
In federated learning, the server intercepts each individual client's weight
update ΔW = w_local - w_global BEFORE FedAvg aggregates them, then runs DLG
to reconstruct what training images that client was using.

Key approximation:
    Under SGD with learning rate `lr`:
        w_after = w_before - lr * gradient
    =>  gradient ≈ (w_before - w_after) / lr = ΔW / lr
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.utils import save_image


def ndarrays_to_state_dict(ndarrays: list, param_names: list) -> OrderedDict:
    """Convert Flower's list of numpy arrays into a named state dict.

    Flower sends model parameters as a flat list of numpy arrays, in the same
    order as model.parameters(). This maps them back to named keys so we can
    compute per-layer deltas.

    Args:
        ndarrays:    List of numpy arrays from parameters_to_ndarrays().
        param_names: List of parameter names from model.named_parameters().

    Returns:
        OrderedDict mapping parameter name → torch.Tensor.
    """
    return OrderedDict(
        (name, torch.tensor(arr))
        for name, arr in zip(param_names, ndarrays)
    )


def weight_delta_to_grads(
    w_before: dict,
    w_after: dict,
    lr: float,
    model: nn.Module,
) -> list:
    """Convert a per-client weight delta into approximate gradient tensors.

    Args:
        w_before: State dict of the global model sent to the client.
        w_after:  State dict of the model weights received back from the client.
        lr:       The learning rate the client used.
        model:    The model (used to get parameter order via named_parameters).

    Returns:
        List of gradient tensors in the same order as model.parameters().
    """
    grads = []
    for name, _ in model.named_parameters():
        delta = w_before[name].float() - w_after[name].float()
        grads.append((delta / lr).detach())
    return grads


def _gradient_distance(dummy_grads: tuple, true_grads: list) -> torch.Tensor:
    """L2 distance between two lists of gradient tensors."""
    return sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, true_grads))


def run_dlg_attack(
    model: nn.Module,
    true_grads: list,
    num_steps: int = 300,
    device: str = "cpu",
    save_dir: str = "dlg_results",
) -> torch.Tensor:
    """Reconstruct a private image from a single client's observed gradient.

    The attacker starts from random noise and iteratively optimizes it so that
    the gradient it produces on the model matches the intercepted true_grads.
    When the gradient distance converges, dummy_data approximates the real image.

    Args:
        model:      Neural network with same architecture as the client.
        true_grads: List of gradient tensors — one per model parameter.
                    Computed from ΔW = w_global - w_local of ONE client.
        num_steps:  Number of L-BFGS optimization steps.
        device:     Torch device string.
        save_dir:   Directory to save intermediate and final reconstructions.

    Returns:
        Reconstructed dummy input tensor (detached, on CPU).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Start from random noise in the same normalized space as client data
    # Client uses Normalize((0.5,), (0.5,)) so images are in roughly [-1, 1]
    dummy_data = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
    # DLG jointly optimizes the label — use a soft vector, not one-hot
    dummy_label = torch.randn(1, 10, requires_grad=True, device=device)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    print(f"\n  [DLG] {num_steps} steps → '{save_dir}/'")

    for step in range(num_steps):

        def closure():
            optimizer.zero_grad()
            pred = model(dummy_data)
            loss = criterion(pred, dummy_label.softmax(dim=-1))
            dummy_grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            grad_dist = _gradient_distance(dummy_grads, true_grads)
            grad_dist.backward()
            return grad_dist

        loss_val = optimizer.step(closure)

        if step % 50 == 0:
            img = (dummy_data.detach().clone() * 0.5 + 0.5).clamp(0, 1)
            save_image(img, os.path.join(save_dir, f"step_{step:04d}.png"))
            print(f"  step {step:3d}/{num_steps}  grad_dist={float(loss_val):.6f}")

    img = (dummy_data.detach().clone() * 0.5 + 0.5).clamp(0, 1)
    save_image(img, os.path.join(save_dir, "final.png"))
    print(f"  [DLG] Saved → '{save_dir}/final.png'")

    return dummy_data.detach().cpu()
