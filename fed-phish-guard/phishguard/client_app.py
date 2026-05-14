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
    model, device = _load_model(msg, context)
    batch_size = context.run_config["batch-size"]
    trainloader, valloader, _, pos_weight = _load_data(context, batch_size, device)

    history, _ = train_fn(
        model,
        trainloader,
        valloader,
        pos_weight,
        lr=context.run_config["learning-rate"],
        device=device,
        num_epochs=context.run_config["local-epochs"],
    )

    summary = summarize_history(history)
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": summary["avg_train_loss"],
        "val_loss": summary["avg_val_loss"],
        "val_f1": summary["avg_val_f1"],
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model, device = _load_model(msg, context)
    batch_size = context.run_config["batch-size"]
    _, _, testloader, pos_weight = _load_data(context, batch_size, device)

    eval_metrics, _, _ = eval_fn(model, testloader, pos_weight, device)

    metrics = {
        "eval_loss": eval_metrics["loss"],
        "eval_acc": eval_metrics["accuracy"],
        "eval_f1": eval_metrics["f1"],
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
