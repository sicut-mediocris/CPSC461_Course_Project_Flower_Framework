"""flower: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import parameters_to_ndarrays
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from flower.task import Net, load_centralized_dataset, test
from flower.dlg_attack import ndarrays_to_state_dict, weight_delta_to_grads, run_dlg_attack

# Create ServerApp
app = ServerApp()


class DLGFedAvg(FedAvg):
    """FedAvg strategy extended with a DLG attack hook.

    Before aggregating client updates each round, this strategy intercepts
    each individual client's weight delta ΔW = w_global - w_local and runs
    the DLG attack to attempt reconstruction of that client's private images.

    This is the correct place for the attack: BEFORE FedAvg mixes all clients
    together. Once averaged, individual client signals are lost.
    """

    def __init__(self, *args, dlg_enabled=False, dlg_steps=300, dlg_round=1,
                 lr=0.01, save_dir="dlg_results", **kwargs):
        super().__init__(*args, **kwargs)
        self.dlg_enabled = dlg_enabled
        self.dlg_steps = dlg_steps
        self.dlg_round = dlg_round   # which round to attack (0 = every round)
        self.lr = lr
        self.save_dir = save_dir
        self._w_global_arrays = None  # captured in configure_fit each round
        # Store param names once — used to map flat arrays back to state dicts
        _tmp = Net()
        self._param_names = [name for name, _ in _tmp.named_parameters()]

    def configure_fit(self, server_round, parameters, client_manager):
        """Capture the current global weights before clients train on them."""
        self._w_global_arrays = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        """Intercept per-client results, run DLG, then aggregate normally."""
        attack_this_round = (
            self.dlg_enabled
            and self._w_global_arrays is not None
            and (self.dlg_round == 0 or server_round == self.dlg_round)
        )

        if attack_this_round:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            w_global_sd = ndarrays_to_state_dict(self._w_global_arrays, self._param_names)

            print(f"\n[DLG] Round {server_round} — attacking {len(results)} client(s)")

            for client_idx, (_, fit_res) in enumerate(results):
                # Each client sent back their locally updated weights
                w_local_arrays = parameters_to_ndarrays(fit_res.parameters)
                w_local_sd = ndarrays_to_state_dict(w_local_arrays, self._param_names)

                # Build the attack model initialized to w_global (pre-training state)
                attack_model = Net()
                attack_model.load_state_dict(w_global_sd, strict=False)
                attack_model.to(device)

                # ΔW / lr ≈ gradient of the client's training data
                true_grads = weight_delta_to_grads(w_global_sd, w_local_sd, self.lr, attack_model)
                true_grads = [g.to(device) for g in true_grads]

                client_save_dir = f"{self.save_dir}/round_{server_round}_client_{client_idx}"
                run_dlg_attack(
                    model=attack_model,
                    true_grads=true_grads,
                    num_steps=self.dlg_steps,
                    device=str(device),
                    save_dir=client_save_dir,
                )

        # Proceed with normal FedAvg aggregation — DLG is purely observational
        return super().aggregate_fit(server_round, results, failures)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    dlg_enabled: bool = context.run_config.get("dlg-attack", False)
    dlg_steps: int = int(context.run_config.get("dlg-steps", 300))
    dlg_round: int = int(context.run_config.get("dlg-round", 1))

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Use DLGFedAvg — it behaves exactly like FedAvg but intercepts client
    # updates before aggregation to run the DLG attack when enabled
    strategy = DLGFedAvg(
        fraction_evaluate=fraction_evaluate,
        dlg_enabled=dlg_enabled,
        dlg_steps=dlg_steps,
        dlg_round=dlg_round,
        lr=lr,
        save_dir="dlg_results",
    )

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


def global_evaluate(_server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


def weighted_average(metrics):
    """A function that aggregates metrics."""
    total_examples = sum(m["num-examples"] for m in metrics)
    weighted_acc = sum(m["accuracy"] * m["num-examples"] for m in metrics) / total_examples
    weighted_loss = sum(m["train_loss"] * m["num-examples"] for m in metrics) / total_examples
    return {"accuracy": weighted_acc, "train_loss": weighted_loss}
