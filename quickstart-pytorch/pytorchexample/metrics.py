"""Metrics helpers for Flower result formatting and plot export."""

from pathlib import Path
from typing import cast

import matplotlib
from flwr.app import MetricRecord
from flwr.common import Message
from flwr.common.typing import MetricRecordValues
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy import result as strategy_result_module

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def round_metric_value(key: str, value: MetricRecordValues) -> MetricRecordValues:
    """Rounds a single metric value based on its key to improve readability.
    Accuracy metrics are rounded to 2 decimals, others to 4 decimals.
    """
    if isinstance(value, float):
        decimals = 2 if "acc" in key.lower() or "accuracy" in key.lower() else 4
        return round(value, decimals)
    if isinstance(value, list):
        return [round(item, 4) if isinstance(item, float) else item for item in value]
    return value


def round_metric_record(metric_record: MetricRecord | None) -> MetricRecord | None:
    """Rounds all numeric values within a given MetricRecord.
    Returns None if the input record is None.
    """
    if metric_record is None:
        return None

    rounded = MetricRecord()
    for key, value in metric_record.items():
        rounded[key] = round_metric_value(key, value)
    return rounded


def format_metric_value(value: MetricRecordValues) -> str:
    """Formats a metric value (float, int, or list) into a string representation.
    Floats are formatted to 4 decimal places.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        formatted = [f"{item:.4f}" if isinstance(item, float) else str(item) for item in value]
        return f"[{', '.join(formatted)}]"
    return str(value)


strategy_result_module.format_value = format_metric_value


class TrackingFedAvg(FedAvg):
    """FedAvg strategy that stores per-client metrics for post-run plots."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_metrics_by_client: dict[int, dict[int, dict[str, MetricRecordValues]]] = {}
        self.evaluate_metrics_by_client: dict[int, dict[int, dict[str, MetricRecordValues]]] = {}

    def _capture_client_metrics(
        self,
        server_round: int,
        replies: list[Message],
        destination: dict[int, dict[int, dict[str, MetricRecordValues]]],
    ) -> None:
        round_metrics: dict[int, dict[str, MetricRecordValues]] = {}
        for reply in replies:
            if reply.has_error() or "metrics" not in reply.content:
                continue

            metric_record = cast(MetricRecord, reply.content["metrics"])
            round_metrics[int(reply.metadata.src_node_id)] = {
                key: round_metric_value(key, value) for key, value in metric_record.items()
            }

        if round_metrics:
            destination[server_round] = round_metrics

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ):
        replies_list = list(replies)
        self._capture_client_metrics(server_round, replies_list, self.train_metrics_by_client)
        arrays, metrics = super().aggregate_train(server_round, replies_list)
        return arrays, round_metric_record(metrics)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies,
    ):
        replies_list = list(replies)
        self._capture_client_metrics(server_round, replies_list, self.evaluate_metrics_by_client)
        metrics = super().aggregate_evaluate(server_round, replies_list)
        return round_metric_record(metrics)


def plot_metric_history(
    output_path: Path,
    title: str,
    ylabel: str,
    client_history: dict[int, dict[int, dict[str, MetricRecordValues]]],
    client_aggregated_history: dict[int, MetricRecord],
    client_metric_key: str,
    server_history: dict[int, MetricRecord] | None = None,
    server_metric_key: str | None = None,
) -> None:
    """Plots the history of a specific metric over federated learning rounds.
    It plots individual client metrics, aggregated client metrics, and server metrics.
    """
    client_points: dict[int, list[tuple[int, float]]] = {}
    for server_round, metrics_by_client in sorted(client_history.items()):
        for client_id, metric_values in metrics_by_client.items():
            if client_metric_key not in metric_values:
                continue
            client_points.setdefault(client_id, []).append((server_round, float(metric_values[client_metric_key])))

    aggregated_points = [
        (server_round, float(metric_record[client_metric_key]))
        for server_round, metric_record in sorted(client_aggregated_history.items())
        if client_metric_key in metric_record
    ]

    server_points: list[tuple[int, float]] = []
    if server_history is not None and server_metric_key is not None:
        server_points = [
            (server_round, float(metric_record[server_metric_key]))
            for server_round, metric_record in sorted(server_history.items())
            if server_metric_key in metric_record
        ]

    if not client_points and not aggregated_points and not server_points:
        return

    plt.figure(figsize=(10, 6))
    for client_id, points in sorted(client_points.items()):
        rounds = [round_number for round_number, _ in points]
        values = [value for _, value in points]
        plt.plot(rounds, values, marker="o", linewidth=1.5, alpha=0.6, label=f"Client {client_id}")

    if aggregated_points:
        rounds = [round_number for round_number, _ in aggregated_points]
        values = [value for _, value in aggregated_points]
        plt.plot(rounds, values, marker="o", linewidth=2.5, color="black", label="Aggregated clients")

    if server_points:
        rounds = [round_number for round_number, _ in server_points]
        values = [value for _, value in server_points]
        plt.plot(rounds, values, marker="s", linewidth=2.5, linestyle="--", color="tab:red", label="Server evaluation")

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metric_plots(strategy: TrackingFedAvg, result) -> None:
    """Generates and saves standard plots for accuracy and loss metrics.
    Plots are saved in the 'plots' directory.
    """
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    plot_metric_history(
        output_path=plots_dir / "client_accuracy.png",
        title="Client Accuracy (Evaluation) by Round",
        ylabel="Accuracy (%)",
        client_history=strategy.evaluate_metrics_by_client,
        client_aggregated_history=result.evaluate_metrics_clientapp,
        client_metric_key="eval_acc",
        server_history=result.evaluate_metrics_serverapp,
        server_metric_key="accuracy (%)",
    )
    plot_metric_history(
        output_path=plots_dir / "client_evaluation_loss.png",
        title="Client Evaluation Loss by Round",
        ylabel="Loss",
        client_history=strategy.evaluate_metrics_by_client,
        client_aggregated_history=result.evaluate_metrics_clientapp,
        client_metric_key="eval_loss",
        server_history=result.evaluate_metrics_serverapp,
        server_metric_key="loss",
    )
    plot_metric_history(
        output_path=plots_dir / "client_train_loss.png",
        title="Client Training Loss by Round",
        ylabel="Loss",
        client_history=strategy.train_metrics_by_client,
        client_aggregated_history=result.train_metrics_clientapp,
        client_metric_key="train_loss",
    )
