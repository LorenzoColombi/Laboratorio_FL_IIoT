"""fed-phish-guard: A Flower / PyTorch app (federated phishing URL detection)."""

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from phishguard.data import VOCAB_SIZE
from phishguard.model import PhishingCNN

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # TODO: complete this function by copying the implementation from the README.
    raise NotImplementedError("Complete ServerApp main for exercise 2")
