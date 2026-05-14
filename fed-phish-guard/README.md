---
tags: [privacy, nlp, fds]
dataset: [Phishing-URL]
framework: [torch]
---

# Federated Phishing URL Detection

This is a federated learning application built with **PyTorch** and **Flower**, simulating privacy-preserving phishing URL detection across distributed browser clients using [Phishing-Dataset](https://huggingface.co/datasets/ealvaradob/phishing-dataset).

Each client represents a group of users (e.g., browser installations or organizations). URLs are processed locally, and only model updates are shared with the server — raw browsing data never leaves the device.

This app demonstrates how federated learning can improve web security while preserving user privacy.

## Prerequisites

- Python 3.10 or higher
- pip

## Fetch the App

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @mozilla-ai/fed-phish-guard
```

This will create a new directory called `fed-phish-guard` with the following structure:

```shell
fed-phish-guard
├── phishguard
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   ├── model.py        # Defines model
│   ├── data.py         # Defines data loading
│   └── train.py        # Defines training and evaluation
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

## Model

URLs are encoded as raw UTF-8 bytes rather than subword tokens, capturing character-level patterns critical for phishing detection (e.g., `paypa1.com`, homoglyphs, misleading subdomains).

```
Input URL → UTF-8 bytes → byte indices (vocab: 258 tokens)
    ↓
Embedding (258 × 64)
    ↓
Parallel convolutions (k=3, 5, 7 | 128 filters each) → concat
    ↓
Conv block 1: Conv1D(256) → ReLU → BN → MaxPool(2)
    ↓
Conv block 2: Conv1D(128) → ReLU → BN → MaxPool(2)
    ↓
Global max pool → Linear(256) → Dropout(0.3) → Linear(1) → sigmoid
```

~568K trainable parameters. Binary classification: **phishing = 1**, **benign = 0**.

## Run the App

You can run your Flower App in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!TIP]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations, how to use more virtual SuperNodes, and how to configure CPU/GPU usage in your ClientApp.

Install the dependencies defined in `pyproject.toml` as well as the `phishguard` package.

```bash
cd fed-phish-guard && pip install -e .
```

Run with default settings:

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=5e-4"
```

To train on multiple datasets (merged and deduplicated), use a comma-separated list:

```bash
flwr run . --run-config "datasets='ealvaradob/phishing-dataset,kmack/Phishing_urls'"
```

### Run with the Deployment Engine

To run this app using Flower’s Deployment Engine, first prepare the local dataset for each client.\
For example, suppose each client’s data is stored in the directory `/path/to/client_data`.

Next, pass the data path to each `SuperNode` using the `--node-config` option:

```shell
flower-supernode \
    --insecure \
    --superlink <SUPERLINK-FLEET-API> \
    --node-config="data-path=/path/to/client_data"
```

Finally, ensure the environment of each `SuperNode` has all dependencies installed.
Then, launch the run via `flwr run` but pointing to a `SuperLink` connection that specifies the `SuperLink` your `SuperNode` is connected to:

```shell
flwr run . <SUPERLINK-CONNECTION> --stream
```

> [!TIP]
> Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

## Expected Results

### Federated Training (10 clients, 3 rounds)

Note that federated results will vary by number of rounds, clients, and data distribution.
With default settings (`flwr run .`), each client trains on ~83K URLs (IID partitioned):

| Metric   | Value  |
| -------- | ------ |
| Accuracy | ~95.2% |
| F1 Score | ~94.8% |
| ROC-AUC  | ~98.9% |

Sample output:

```
(ClientAppActor) Train Loss: 0.1956 | Time: 2565.6s
(ClientAppActor) Val Loss: 0.1376 | Acc: 0.9516 | F1: 0.9477 | AUC: 0.9890
(ClientAppActor) Best model: epoch 1 with F1=0.9477
```

### Centralized Baseline

For comparison, the non-federated baseline achieves (after 20 epochs):

| Metric   | Value  |
| -------- | ------ |
| Accuracy | ~98.4% |
| F1 Score | ~98.3% |
| ROC-AUC  | ~99.8% |

See [`baseline/`](baseline/) for standalone training scripts that reuse the `phishguard` libraries without federated learning.
