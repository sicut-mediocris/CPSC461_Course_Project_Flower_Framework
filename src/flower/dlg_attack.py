"""Deep Leakage from Gradients (DLG) attack — Zhu et al., 2019.

The attack reconstructs private client training data from observed gradient
updates. In a federated learning setting, the server (or any eavesdropper)
can run this against a client's weight delta to recover the original images.

Key approximation used here:
    Under SGD with learning rate `lr`:
        w_after = w_before - lr * gradient
    =>  gradient ≈ (w_before - w_after) / lr

This lets us treat the weight update ΔW as a proxy for the true gradient.
"""

import os

import torch
import torch.nn as nn
from torchvision.utils import save_image


def _gradient_distance(dummy_grads: tuple, true_grads: list) -> torch.Tensor:
    """L2 distance between two lists of gradient tensors."""
    return sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, true_grads))


def weight_delta_to_grads(
    w_before: dict,
    w_after: dict,
    lr: float,
    model: nn.Module,
) -> list:
    """Convert a weight update (ΔW) into approximate gradient tensors.

    Args:
        w_before: model state_dict before client training (w_global).
        w_after:  model state_dict after  client training (w_local).
        lr:       client learning rate.
        model:    the model (used to get parameter order via named_parameters).

    Returns:
        List of gradient tensors in the same order as model.parameters().
    """
    grads = []
    for name, _ in model.named_parameters():
        delta = w_before[name].float() - w_after[name].float()
        grads.append((delta / lr).detach())
    return grads


def run_dlg_attack(
    model: nn.Module,
    true_grads: list,
    num_steps: int = 300,
    device: str = "cpu",
    save_dir: str = "dlg_results",
) -> torch.Tensor:
    """Reconstruct a private image from observed gradients using DLG.

    The attacker initializes random dummy data and iteratively optimizes it
    so that the gradient it produces on the model matches the observed true_grads.
    When the gradient distances converge, dummy_data approximates the real input.

    Args:
        model:      Neural network with the same architecture as the client.
        true_grads: List of gradient tensors (one per model parameter).
        num_steps:  Number of L-BFGS optimization steps.
        device:     Torch device string ('cpu' or 'cuda:0').
        save_dir:   Directory to save intermediate and final reconstructions.

    Returns:
        The reconstructed dummy input tensor (detached, on CPU).
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Start from random noise — same normalization as client data (mean=0.5, std=0.5)
    dummy_data = torch.randn(1, 1, 28, 28, requires_grad=True, device=device)
    # Soft label vector — DLG jointly optimizes data and labels
    dummy_label = torch.randn(1, 10, requires_grad=True, device=device)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    print(f"\n[DLG] Starting attack — {num_steps} steps | saving to '{save_dir}/'")

    for step in range(num_steps):

        def closure():
            optimizer.zero_grad()
            pred = model(dummy_data)
            # Use softmax of dummy_label so it behaves like a probability distribution
            loss = criterion(pred, dummy_label.softmax(dim=-1))
            dummy_grads = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            grad_dist = _gradient_distance(dummy_grads, true_grads)
            grad_dist.backward()
            return grad_dist

        loss_val = optimizer.step(closure)

        if step % 50 == 0:
            # Denormalize: client normalized with mean=0.5, std=0.5 → back to [0,1]
            img = (dummy_data.detach().clone() * 0.5 + 0.5).clamp(0, 1)
            save_image(img, os.path.join(save_dir, f"step_{step:04d}.png"))
            print(f"  step {step:3d}/{num_steps}  grad_dist={float(loss_val):.6f}")

    # Final reconstruction
    img = (dummy_data.detach().clone() * 0.5 + 0.5).clamp(0, 1)
    save_image(img, os.path.join(save_dir, "final.png"))
    print(f"[DLG] Done. Final image saved → '{save_dir}/final.png'")

    return dummy_data.detach().cpu()
