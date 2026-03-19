"""
Deep Leakage from Gradients (DLG) Attack
=========================================

This module implements the DLG attack proposed by Zhu et al. (2019):
  "Deep Leakage from Gradients" — https://arxiv.org/abs/1906.08935

The attack reconstructs private training data from gradient updates that
a client sends to the server during federated learning.

HOW THE ATTACK WORKS (conceptually)
-------------------------------------
In federated learning, each client:
  1. Receives the global model weights from the server.
  2. Trains locally on its private data.
  3. Returns only the gradient update (weight difference) to the server.

The threat assumption is that an honest-but-curious server (or a
man-in-the-middle adversary) intercepts these gradients.

The DLG attack then:
  1. Initialises "dummy" data and labels at random.
  2. Computes what the gradients *would* be if that dummy data had been used.
  3. Measures how different those dummy gradients are from the intercepted
     real gradients (L2 distance in gradient space).
  4. Uses a gradient-based optimizer (LBFGS) to tweak the dummy data so
     that its gradients match the real gradients more closely.
  5. After many iterations the dummy data converges to a reconstruction of
     the real private data.

ENTRY POINT
-----------
Call `run_dlg_attack(model, real_gradients, ...)` to perform the attack.
All other functions are internal helpers.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dlg_attack(
    model: nn.Module,
    real_gradients: List[Tensor],
    num_images: int = 1,
    num_iterations: int = 300,
    lr: float = 1.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct private training data from intercepted gradients.

    This is the main entry point for the DLG attack.

    Parameters
    ----------
    model : nn.Module
        The global model (same architecture used by the victim client).
    real_gradients : list of Tensor
        The gradient tensors intercepted from the victim client.
        Each tensor must have the same shape as the corresponding parameter
        in ``model.parameters()``.
    num_images : int
        Number of training samples to reconstruct (default: 1).
    num_iterations : int
        Gradient-descent iterations for the reconstruction optimiser.
        More iterations → better reconstruction, but slower.
    lr : float
        Learning rate for LBFGS optimiser used during reconstruction.
    device : torch.device, optional
        Device to run the attack on.  Defaults to CPU.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dummy_data : Tensor  shape (num_images, C, H, W)
        Reconstructed images after the attack.
    dummy_labels : Tensor  shape (num_images,)
        Reconstructed class labels.
    """
    if device is None:
        device = torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)

    # Work on a copy of the model so the original is unchanged
    model = copy.deepcopy(model).to(device).eval()

    # Infer input shape from the first convolutional / linear layer
    input_shape = _infer_input_shape(model)

    # -----------------------------------------------------------------------
    # Step 1 – Initialise dummy data and labels at random
    # -----------------------------------------------------------------------
    # The dummy data starts as noise; the optimiser will gradually shape it
    # so that its gradients match the real ones.
    dummy_data = torch.randn(
        (num_images, *input_shape), requires_grad=True, device=device
    )
    dummy_labels = torch.randn(
        (num_images, _num_classes(model)), requires_grad=True, device=device
    )

    # -----------------------------------------------------------------------
    # Step 2 – Optimise dummy data to match real gradients
    # -----------------------------------------------------------------------
    optimizer = torch.optim.LBFGS(
        [dummy_data, dummy_labels], lr=lr, max_iter=20
    )

    criterion = nn.CrossEntropyLoss()

    history: List[float] = []

    for iteration in range(num_iterations):
        def closure() -> Tensor:
            optimizer.zero_grad()

            # Compute gradients for the current dummy data
            dummy_pred = model(dummy_data)
            # Use soft labels (continuous) during optimisation for stability
            dummy_loss = criterion(dummy_pred, dummy_labels.softmax(dim=-1))
            dummy_gradients = torch.autograd.grad(
                dummy_loss,
                model.parameters(),
                create_graph=True,
            )

            # Gradient-matching loss: L2 distance between dummy and real grads
            grad_diff = _gradient_distance(dummy_gradients, real_gradients)
            grad_diff.backward()
            return grad_diff

        gradient_loss = optimizer.step(closure)
        history.append(float(gradient_loss))

        if (iteration + 1) % 50 == 0:
            print(
                f"  [DLG] Iteration {iteration + 1:4d}/{num_iterations}"
                f"  |  Gradient loss: {history[-1]:.6f}"
            )

    # Clamp reconstructed images to the normalised [−1, 1] range used by the
    # Flower training pipeline (mean=0.5, std=0.5 normalisation).
    dummy_data = dummy_data.detach().clamp(-1, 1)

    # Convert soft-label logits back to a hard class index
    dummy_labels = dummy_labels.detach().argmax(dim=-1)

    return dummy_data, dummy_labels


# ---------------------------------------------------------------------------
# Gradient capture helper (used inside client_app.py)
# ---------------------------------------------------------------------------

def capture_gradients(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    device: torch.device,
) -> List[Tensor]:
    """Compute and return per-parameter gradients for a single batch.

    This simulates what an adversary would intercept: the gradient update
    that the client is about to send to the server.

    Parameters
    ----------
    model : nn.Module
        Model with current global weights loaded.
    images : Tensor  shape (N, C, H, W)
        A batch of private training images.
    labels : Tensor  shape (N,)
        Corresponding ground-truth labels.
    device : torch.device
        Device to compute on.

    Returns
    -------
    list of Tensor
        One gradient tensor per model parameter (detached from the graph).
    """
    model = model.to(device)
    model.zero_grad()

    criterion = nn.CrossEntropyLoss()
    outputs = model(images.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()

    # Detach so the returned tensors are independent of the computation graph
    return [p.grad.detach().clone() for p in model.parameters()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gradient_distance(
    dummy_gradients: Tuple[Tensor, ...],
    real_gradients: List[Tensor],
) -> Tensor:
    """Compute L2 distance between dummy and real gradient vectors.

    Summing the squared L2 norms across all layers gives a single scalar
    loss that the outer optimiser can minimise.

    Parameters
    ----------
    dummy_gradients : tuple of Tensor
        Gradients computed from the current dummy data (attached to graph).
    real_gradients : list of Tensor
        Intercepted gradients from the victim client (detached).

    Returns
    -------
    Tensor (scalar)
        Total gradient-matching loss.
    """
    total_loss = sum(
        ((dg - rg) ** 2).sum()
        for dg, rg in zip(dummy_gradients, real_gradients)
    )
    return total_loss


def _infer_input_shape(model: nn.Module) -> Tuple[int, ...]:
    """Return the (C, H, W) input shape expected by *model*.

    Walks the module tree and inspects the first Conv2d or Linear layer.
    Falls back to a sensible default (1, 28, 28) for Fashion-MNIST if the
    shape cannot be determined automatically.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # For Fashion-MNIST the first conv has in_channels=1,
            # but we need spatial size too.  We use 28×28 as the default
            # for the CNN defined in task.py.
            return (module.in_channels, 28, 28)
        if isinstance(module, nn.Linear):
            # Fully-connected networks: square input assumed
            side = int(math.sqrt(module.in_features))
            return (1, side, side)
    # Fallback
    return (1, 28, 28)


def _num_classes(model: nn.Module) -> int:
    """Return the output dimension (number of classes) of *model*."""
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return last_linear.out_features
    return 10  # Fashion-MNIST default


# ---------------------------------------------------------------------------
# Stand-alone demo (run with: python -m attacks.dlg_attack)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running from the repo root without installing the package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from flower.task import Net, load_data

    print("=" * 60)
    print("DLG Attack Demo — Fashion-MNIST / Flower CNN")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------------
    # 1. Load the global model (victim's starting point)
    # -----------------------------------------------------------------------
    model = Net().to(device)

    # -----------------------------------------------------------------------
    # 2. Grab one real training sample (simulates what the victim trains on)
    # -----------------------------------------------------------------------
    trainloader, _ = load_data(partition_id=0, num_partitions=10, batch_size=1)
    real_batch = next(iter(trainloader))
    real_images = real_batch["image"].to(device)
    real_labels = real_batch["label"].to(device)

    print(f"Real label: {real_labels.item()}")

    # -----------------------------------------------------------------------
    # 3. Capture gradients (simulates what the server intercepts)
    # -----------------------------------------------------------------------
    print("Capturing gradients from victim's batch …")
    real_gradients = capture_gradients(model, real_images, real_labels, device)

    # -----------------------------------------------------------------------
    # 4. Run DLG attack
    # -----------------------------------------------------------------------
    print("Running DLG attack …\n")
    dummy_data, dummy_labels = run_dlg_attack(
        model=model,
        real_gradients=real_gradients,
        num_images=1,
        num_iterations=300,
        lr=1.0,
        device=device,
        seed=42,
    )

    print(f"\nReconstructed label: {dummy_labels.item()}")
    print("Attack complete.  Inspect dummy_data to evaluate reconstruction.")



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dlg_attack(
    model: nn.Module,
    real_gradients: List[Tensor],
    num_images: int = 1,
    num_iterations: int = 300,
    lr: float = 1.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> Tuple[Tensor, Tensor]:
    """Reconstruct private training data from intercepted gradients.

    This is the main entry point for the DLG attack.

    Parameters
    ----------
    model : nn.Module
        The global model (same architecture used by the victim client).
    real_gradients : list of Tensor
        The gradient tensors intercepted from the victim client.
        Each tensor must have the same shape as the corresponding parameter
        in ``model.parameters()``.
    num_images : int
        Number of training samples to reconstruct (default: 1).
    num_iterations : int
        Gradient-descent iterations for the reconstruction optimiser.
        More iterations → better reconstruction, but slower.
    lr : float
        Learning rate for LBFGS optimiser used during reconstruction.
    device : torch.device, optional
        Device to run the attack on.  Defaults to CPU.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dummy_data : Tensor  shape (num_images, C, H, W)
        Reconstructed images after the attack.
    dummy_labels : Tensor  shape (num_images,)
        Reconstructed class labels.
    """
    if device is None:
        device = torch.device("cpu")

    if seed is not None:
        torch.manual_seed(seed)

    # Work on a copy of the model so the original is unchanged
    model = copy.deepcopy(model).to(device).eval()

    # Infer input shape from the first convolutional / linear layer
    input_shape = _infer_input_shape(model)

    # -----------------------------------------------------------------------
    # Step 1 – Initialise dummy data and labels at random
    # -----------------------------------------------------------------------
    # The dummy data starts as noise; the optimiser will gradually shape it
    # so that its gradients match the real ones.
    dummy_data = torch.randn(
        (num_images, *input_shape), requires_grad=True, device=device
    )
    dummy_labels = torch.randn(
        (num_images, _num_classes(model)), requires_grad=True, device=device
    )

    # -----------------------------------------------------------------------
    # Step 2 – Optimise dummy data to match real gradients
    # -----------------------------------------------------------------------
    optimizer = torch.optim.LBFGS(
        [dummy_data, dummy_labels], lr=lr, max_iter=20
    )

    criterion = nn.CrossEntropyLoss()

    history: List[float] = []

    for iteration in range(num_iterations):
        def closure() -> Tensor:
            optimizer.zero_grad()

            # Compute gradients for the current dummy data
            dummy_pred = model(dummy_data)
            # Use soft labels (continuous) during optimisation for stability
            dummy_loss = criterion(dummy_pred, dummy_labels.softmax(dim=-1))
            dummy_gradients = torch.autograd.grad(
                dummy_loss,
                model.parameters(),
                create_graph=True,
            )

            # Gradient-matching loss: L2 distance between dummy and real grads
            grad_diff = _gradient_distance(dummy_gradients, real_gradients)
            grad_diff.backward()
            return grad_diff

        gradient_loss = optimizer.step(closure)
        history.append(float(gradient_loss))

        if (iteration + 1) % 50 == 0:
            print(
                f"  [DLG] Iteration {iteration + 1:4d}/{num_iterations}"
                f"  |  Gradient loss: {history[-1]:.6f}"
            )

    # Clamp reconstructed images to the normalised [−1, 1] range used by the
    # Flower training pipeline (mean=0.5, std=0.5 normalisation).
    dummy_data = dummy_data.detach().clamp(-1, 1)

    # Convert soft-label logits back to a hard class index
    dummy_labels = dummy_labels.detach().argmax(dim=-1)

    return dummy_data, dummy_labels


# ---------------------------------------------------------------------------
# Gradient capture helper (used inside client_app.py)
# ---------------------------------------------------------------------------

def capture_gradients(
    model: nn.Module,
    images: Tensor,
    labels: Tensor,
    device: torch.device,
) -> List[Tensor]:
    """Compute and return per-parameter gradients for a single batch.

    This simulates what an adversary would intercept: the gradient update
    that the client is about to send to the server.

    Parameters
    ----------
    model : nn.Module
        Model with current global weights loaded.
    images : Tensor  shape (N, C, H, W)
        A batch of private training images.
    labels : Tensor  shape (N,)
        Corresponding ground-truth labels.
    device : torch.device
        Device to compute on.

    Returns
    -------
    list of Tensor
        One gradient tensor per model parameter (detached from the graph).
    """
    model = model.to(device)
    model.zero_grad()

    criterion = nn.CrossEntropyLoss()
    outputs = model(images.to(device))
    loss = criterion(outputs, labels.to(device))
    loss.backward()

    # Detach so the returned tensors are independent of the computation graph
    return [p.grad.detach().clone() for p in model.parameters()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gradient_distance(
    dummy_gradients: Tuple[Tensor, ...],
    real_gradients: List[Tensor],
) -> Tensor:
    """Compute L2 distance between dummy and real gradient vectors.

    Summing the squared L2 norms across all layers gives a single scalar
    loss that the outer optimiser can minimise.

    Parameters
    ----------
    dummy_gradients : tuple of Tensor
        Gradients computed from the current dummy data (attached to graph).
    real_gradients : list of Tensor
        Intercepted gradients from the victim client (detached).

    Returns
    -------
    Tensor (scalar)
        Total gradient-matching loss.
    """
    total_loss = sum(
        ((dg - rg) ** 2).sum()
        for dg, rg in zip(dummy_gradients, real_gradients)
    )
    return total_loss


def _infer_input_shape(model: nn.Module) -> Tuple[int, ...]:
    """Return the (C, H, W) input shape expected by *model*.

    Walks the module tree and inspects the first Conv2d or Linear layer.
    Falls back to a sensible default (1, 28, 28) for Fashion-MNIST if the
    shape cannot be determined automatically.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # For Fashion-MNIST the first conv has in_channels=1,
            # but we need spatial size too.  We use 28×28 as the default
            # for the CNN defined in task.py.
            return (module.in_channels, 28, 28)
        if isinstance(module, nn.Linear):
            # Fully-connected networks: square input assumed
            import math
            side = int(math.sqrt(module.in_features))
            return (1, side, side)
    # Fallback
    return (1, 28, 28)


def _num_classes(model: nn.Module) -> int:
    """Return the output dimension (number of classes) of *model*."""
    last_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is not None:
        return last_linear.out_features
    return 10  # Fashion-MNIST default


# ---------------------------------------------------------------------------
# Stand-alone demo (run with: python -m attacks.dlg_attack)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Allow running from the repo root without installing the package
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from flower.task import Net, load_data

    print("=" * 60)
    print("DLG Attack Demo — Fashion-MNIST / Flower CNN")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------------
    # 1. Load the global model (victim's starting point)
    # -----------------------------------------------------------------------
    model = Net().to(device)

    # -----------------------------------------------------------------------
    # 2. Grab one real training sample (simulates what the victim trains on)
    # -----------------------------------------------------------------------
    trainloader, _ = load_data(partition_id=0, num_partitions=10, batch_size=1)
    real_batch = next(iter(trainloader))
    real_images = real_batch["image"].to(device)
    real_labels = real_batch["label"].to(device)

    print(f"Real label: {real_labels.item()}")

    # -----------------------------------------------------------------------
    # 3. Capture gradients (simulates what the server intercepts)
    # -----------------------------------------------------------------------
    print("Capturing gradients from victim's batch …")
    real_gradients = capture_gradients(model, real_images, real_labels, device)

    # -----------------------------------------------------------------------
    # 4. Run DLG attack
    # -----------------------------------------------------------------------
    print("Running DLG attack …\n")
    dummy_data, dummy_labels = run_dlg_attack(
        model=model,
        real_gradients=real_gradients,
        num_images=1,
        num_iterations=300,
        lr=1.0,
        device=device,
        seed=42,
    )

    print(f"\nReconstructed label: {dummy_labels.item()}")
    print("Attack complete.  Inspect dummy_data to evaluate reconstruction.")
