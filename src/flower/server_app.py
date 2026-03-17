"""flower: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flower.dlg_attack import run_dlg_attack, weight_delta_to_grads

# ConfigRecord stores the configuration information whereas the context stores the run configuration and other information about the current run. 
# The ArrayRecord is used to store the model parameters as arrays, which can be easily sent between the server and clients during training and evaluation.
#  The MetricRecord is used to store the evaluation metrics, such as accuracy and loss, which can be returned after evaluating the global model on the test set.
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower.task import Net, load_centralized_dataset, test

# Net is the neural network architehcture

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Snapshot initial weights — used by DLG to compute ΔW after training
    w_initial = {k: v.clone() for k, v in global_model.state_dict().items()}

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    w_final = result.arrays.to_torch_state_dict()
    torch.save(w_final, "final_model.pt")

    # --- DLG Attack (optional) ---
    # Treats ΔW = w_initial - w_final as a proxy for the client gradient.
    # Under SGD: w_after = w_before - lr * grad  =>  grad ≈ ΔW / lr
    # The server runs DLG to attempt reconstruction of client training images.
    if context.run_config.get("dlg-attack", False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dlg_steps = int(context.run_config.get("dlg-steps", 300))

        attack_model = Net()
        attack_model.load_state_dict(w_initial)
        attack_model.to(device)

        true_grads = weight_delta_to_grads(w_initial, w_final, lr, attack_model)
        true_grads = [g.to(device) for g in true_grads]

        run_dlg_attack(
            model=attack_model,
            true_grads=true_grads,
            num_steps=dlg_steps,
            device=str(device),
            save_dir="dlg_results",
        )


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})

def weighted_average(metrics):
    """A function that aggregates metrics."""
    total_examples = sum(m["num-examples"] for m in metrics)
    weighted_acc = sum(m["accuracy"] * m["num-examples"] for m in metrics) / total_examples
    weighted_loss = sum(m["train_loss"] * m["num-examples"] for m in metrics) / total_examples
    return {"accuracy": weighted_acc, "train_loss": weighted_loss}
