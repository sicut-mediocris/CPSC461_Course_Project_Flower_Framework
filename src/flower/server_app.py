"""flower: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord

# ConfigRecord stores the configuration information whereas the context stores the run configuration and other information about the current run. 
# The ArrayRecord is used to store the model parameters as arrays, which can be easily sent between the server and clients during training and evaluation.
#  The MetricRecord is used to store the evaluation metrics, such as accuracy and loss, which can be returned after evaluating the global model on the test set.
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from typing import List, Tuple

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
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


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
