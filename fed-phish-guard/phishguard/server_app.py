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

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    embed_dim = context.run_config["embed-dim"]
    num_filters = context.run_config["num-filters"]
    dropout = context.run_config["dropout"]

    # Load global model
    global_model = PhishingCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        num_filters=num_filters,
        dropout=dropout,
    )
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
