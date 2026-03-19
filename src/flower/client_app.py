"""flower: A Flower / PyTorch app."""

import copy
import sys
import os
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower.task import Net, load_data
from flower.task import test as test_fn
from flower.task import train as train_fn

# ---------------------------------------------------------------------------
# DLG attack hook – import the capture helper from attacks/dlg_attack.py.
# The import is guarded so the Flower app still runs normally even when the
# attacks package is not on sys.path.
# ---------------------------------------------------------------------------
_ATTACKS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "attacks")
)
if _ATTACKS_DIR not in sys.path:
    sys.path.insert(0, _ATTACKS_DIR)

try:
    from dlg_attack import capture_gradients as _capture_gradients
    _DLG_AVAILABLE = True
except ImportError:
    _DLG_AVAILABLE = False

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # ---------------------------------------------------------------------------
    # DLG hook: capture gradients from the *first* batch before training begins.
    #
    # In a real attack scenario this is the moment the adversary intercepts
    # the client's gradient update.  Here we store them locally so they can
    # be passed to run_dlg_attack() for reconstruction experiments.
    #
    # A fresh DataLoader (batch_size=1) is created specifically for gradient
    # capture so that trainloader is not consumed and training sees the full
    # dataset as normal.
    # ---------------------------------------------------------------------------
    captured_gradients = None
    if _DLG_AVAILABLE:
        capture_loader, _ = load_data(partition_id, num_partitions, batch_size=1)
        first_batch = next(iter(capture_loader))
        first_images = first_batch["image"].to(device)
        first_labels = first_batch["label"].to(device)
        # Keep a pristine copy of the model (pre-training weights) for the attack
        model_snapshot = copy.deepcopy(model)
        captured_gradients = _capture_gradients(
            model_snapshot, first_images, first_labels, device
        )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
