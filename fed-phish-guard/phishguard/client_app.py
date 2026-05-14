"""fed-phish-guard: A Flower / PyTorch app (federated phishing URL detection)."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from phishguard.data import VOCAB_SIZE, load_local_data, load_sim_data
from phishguard.model import PhishingCNN
from phishguard.train import evaluate as eval_fn
from phishguard.train import summarize_history
from phishguard.train import train as train_fn

# Flower ClientApp
app = ClientApp()


def _load_model(msg: Message, context: Context) -> tuple[PhishingCNN, torch.device]:
    """Construct model from run config and load weights from the received message."""
    model = PhishingCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=context.run_config["embed-dim"],
        num_filters=context.run_config["num-filters"],
        dropout=context.run_config["dropout"],
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def _load_data(context: Context, batch_size: int, device: torch.device):
    """Select Simulation or Deployment data loader based on node_config."""
    # Get datasets from run config (comma-separated string or default)
    datasets_str = context.run_config.get("datasets", "ealvaradob/phishing-dataset")
    dataset_ids = [d.strip() for d in datasets_str.split(",")]

    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        return load_sim_data(
            context.node_config["partition-id"],
            context.node_config["num-partitions"],
            batch_size,
            device,
            dataset_ids=dataset_ids,
        )
    return load_local_data(context.node_config["data-path"], batch_size, device)


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # TODO: complete this function by copying the implementation from the README.
    raise NotImplementedError("Complete ClientApp train for exercise 2")


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # TODO: complete this function by copying the implementation from the README.
    raise NotImplementedError("Complete ClientApp evaluate for exercise 2")
