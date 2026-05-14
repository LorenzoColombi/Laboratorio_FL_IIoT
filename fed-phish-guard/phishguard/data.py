"""
Phishing URL Classification - Data Loading & Preprocessing

Step 1: Environment & Dataset Setup
Step 2: Data Preprocessing & DataLoader
"""

import json
from collections import Counter
from typing import Optional

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import WeightedRandomSampler

fds = None  # Cache FederatedDataset
_fds_datasets_key = None  # Track which datasets the cache was built for

# Special tokens
PAD_IDX = 0
UNK_IDX = 1

# Vocabulary size: 256 bytes + PAD + UNK = 258
VOCAB_SIZE = 258


def build_vocab() -> tuple[dict, dict]:
    """Build byte-level vocabulary. Each byte value (0-255) maps to a token index.

    Returns byte2idx and idx2byte mappings.
    Vocabulary size: 258 tokens (256 bytes + PAD + UNK)
    """
    # Index 0 = PAD, Index 1 = UNK, Index 2-257 = bytes 0-255
    byte2idx = {i: i + 2 for i in range(256)}
    idx2byte = {i + 2: i for i in range(256)}

    # Add special tokens
    idx2byte[PAD_IDX] = None  # PAD
    idx2byte[UNK_IDX] = None  # UNK

    return byte2idx, idx2byte


BYTE2IDX, _ = build_vocab()


# -----------------------------------------------------------------------------
# Dataset-specific loaders
# Each returns (urls: list[str], labels: list[int])
# -----------------------------------------------------------------------------


def load_ealvaradob() -> tuple[list[str], list[int]]:
    """Load ealvaradob/phishing-dataset (JSON format)."""
    print("Loading ealvaradob/phishing-dataset...")
    json_path = hf_hub_download(
        repo_id="ealvaradob/phishing-dataset",
        filename="urls.json",
        repo_type="dataset",
    )
    with open(json_path, "r") as f:
        data = json.load(f)
    urls = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    print(f"  Loaded {len(urls):,} URLs")
    return urls, labels


def load_kmack() -> tuple[list[str], list[int]]:
    """Load kmack/Phishing_urls (standard HF dataset)."""
    print("Loading kmack/Phishing_urls...")
    data = load_dataset("kmack/Phishing_urls", split="train+test+valid")
    urls = list(data["text"])
    labels = list(data["label"])
    print(f"  Loaded {len(urls):,} URLs")
    return urls, labels


# Registry of dataset loaders
DATASET_LOADERS = {
    "ealvaradob/phishing-dataset": load_ealvaradob,
    "kmack/Phishing_urls": load_kmack,
}


def load_datasets(dataset_ids: list[str]) -> tuple[list[str], list[int]]:
    """Load and merge multiple datasets by their HuggingFace IDs.

    Args:
        dataset_ids: List of HuggingFace dataset identifiers.

    Returns:
        Tuple of (urls, labels) lists.

    Raises:
        ValueError: If a dataset ID is not in the registry.
    """
    all_urls = []
    all_labels = []

    for dataset_id in dataset_ids:
        if dataset_id not in DATASET_LOADERS:
            raise ValueError(f"Unknown dataset: {dataset_id}. " f"Supported datasets: {list(DATASET_LOADERS.keys())}")
        loader = DATASET_LOADERS[dataset_id]
        urls, labels = loader()
        all_urls.extend(urls)
        all_labels.extend(labels)

    return all_urls, all_labels


def url_to_bytes(url: str) -> bytes:
    """Convert URL string to bytes, decoding percent-encoded sequences.

    1. Decode percent-encoded sequences (%XX -> byte)
    2. Lowercase the URL
    3. Encode to UTF-8 bytes

    Examples:
        "PayPal.com" -> b"paypal.com"
        "caf%C3%A9.com" -> b"caf\xc3\xa9.com" (é decoded)
        "%2F%3F" -> b"/?" (/ and ? decoded)
    """
    from urllib.parse import unquote_to_bytes

    # Lowercase first, then decode percent-encoding to bytes
    # unquote_to_bytes expects a string and returns bytes
    try:
        return unquote_to_bytes(url.lower())
    except Exception:
        # Fallback: just encode as UTF-8 without decoding
        return url.lower().encode("utf-8", errors="replace")


def url_to_indices(url: str, byte2idx: dict, max_len: int = 256) -> list[int]:
    """Convert URL string to list of byte token indices.

    - Decode percent-encoded sequences
    - Lowercase and encode to UTF-8 bytes
    - Truncate or pad to max_len
    - Map bytes to indices
    """
    url_bytes = url_to_bytes(url)
    indices = []
    for byte in url_bytes[:max_len]:
        indices.append(byte2idx.get(byte, UNK_IDX))

    # Pad if needed
    while len(indices) < max_len:
        indices.append(PAD_IDX)

    return indices


class PhishingURLDataset(TorchDataset):
    """PyTorch Dataset for phishing URL classification."""

    def __init__(self, urls: list[str], labels: list[int], char2idx: dict, max_len: int = 256):
        self.urls = urls
        self.labels = labels
        self.char2idx = char2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]

        indices = url_to_indices(url, self.char2idx, self.max_len)

        return {
            "input_ids": torch.tensor(indices, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def create_dataloaders(
    train_data: tuple,
    val_data: tuple,
    test_data: tuple,
    char2idx: dict,
    max_len: int = 256,
    batch_size: int = 128,
    num_workers: int = 2,
    use_weighted_sampler: bool = True,
):
    """Create PyTorch DataLoaders for train, val, and test sets.

    Args:
        train_data: (urls, labels) tuple for training
        val_data: (urls, labels) tuple for validation
        test_data: (urls, labels) tuple for testing
        char2idx: Character to index mapping
        max_len: Maximum URL length
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_weighted_sampler: Use weighted sampling for class imbalance

    Returns:
        train_loader, val_loader, test_loader
    """
    train_urls, train_labels = train_data
    val_urls, val_labels = val_data
    test_urls, test_labels = test_data

    train_dataset = PhishingURLDataset(train_urls, train_labels, char2idx, max_len)
    val_dataset = PhishingURLDataset(val_urls, val_labels, char2idx, max_len)
    test_dataset = PhishingURLDataset(test_urls, test_labels, char2idx, max_len)

    # Weighted sampler for training (handles class imbalance)
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        class_counts = Counter(train_labels)
        weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False  # Sampler handles shuffling

    # pin_memory only supported on CUDA, not MPS
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def get_class_weights(labels: list[int], device: Optional[torch.device] = None):
    """Compute class weights for imbalanced dataset.

    Can be used with BCEWithLogitsLoss pos_weight parameter.
    """
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos])
    if device:
        pos_weight = pos_weight.to(device)
    return pos_weight


def preprocess_dataset(dataset: HFDataset, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42):
    """Preprocess the dataset: deduplicate, split, and compute statistics."""

    # Faster/idiomatic for HF Datasets
    urls = list(dataset["text"])
    labels = list(dataset["label"])

    original_total = len(urls)
    print(f"Total samples: {original_total}")

    # Deduplicate based on normalized form (what the model sees)
    seen = {}
    for url, label in zip(urls, labels):
        normalized = url_to_bytes(url)  # must be hashable (e.g., bytes)
        if normalized not in seen:
            seen[normalized] = (url, label)

    urls = [v[0] for v in seen.values()]
    labels = [v[1] for v in seen.values()]

    dedup_total = len(urls)
    print(f"After deduplication: {dedup_total}")

    # Guard: stratified splits need at least 2 examples per class and enough per split
    if len(set(labels)) < 2:
        raise ValueError("Need at least 2 classes for stratified splitting.")
    class_counts = {c: labels.count(c) for c in set(labels)}
    if min(class_counts.values()) < 2:
        raise ValueError(f"Not enough samples per class after deduplication: {class_counts}")

    # First split: train+val vs test
    train_val_urls, test_urls, train_val_labels, test_labels = train_test_split(
        urls, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_urls, val_urls, train_labels, val_labels = train_test_split(
        train_val_urls,
        train_val_labels,
        test_size=val_ratio,
        stratify=train_val_labels,
        random_state=seed,
    )

    # If labels are 0/1 or bool, ratios work; otherwise you need a mapping.
    def ratio_pos(y):
        if len(y) == 0:
            return float("nan")
        return sum(y) / len(y)

    stats = {
        "original_total": original_total,
        "dedup_total": dedup_total,
        "train": len(train_urls),
        "val": len(val_urls),
        "test": len(test_urls),
        "train_phishing_ratio": ratio_pos(train_labels),
        "val_phishing_ratio": ratio_pos(val_labels),
        "test_phishing_ratio": ratio_pos(test_labels),
    }

    return (
        (train_urls, train_labels),
        (val_urls, val_labels),
        (test_urls, test_labels),
        stats,
    )


def _make_loaders(hf_ds, batch_size: int, device: torch.device):
    """Preprocess a HuggingFace dataset and return (train, val, test loaders,
    pos_weight)."""
    train_data, val_data, test_data, _ = preprocess_dataset(hf_ds)
    pos_weight = get_class_weights(train_data[1], device)
    trainloader, valloader, testloader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        BYTE2IDX,
        batch_size=batch_size,
    )
    return trainloader, valloader, testloader, pos_weight


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    device: torch.device,
    dataset_ids: list[str] | None = None,
    fraction: float = 1.0,
):
    """Load partition data for federated simulation.

    Args:
        partition_id: The partition index for this client.
        num_partitions: Total number of partitions.
        batch_size: Batch size for data loaders.
        device: Device to load tensors to.
        dataset_ids: List of dataset IDs to load. Defaults to ["ealvaradob/phishing-dataset"].

    Returns:
        Tuple of (train_loader, val_loader, test_loader, pos_weight).
    """
    global fds, _fds_datasets_key

    if dataset_ids is None:
        dataset_ids = ["ealvaradob/phishing-dataset"]

    # Create a hashable key to check if we need to rebuild the cache
    datasets_key = tuple(sorted(dataset_ids))

    if fds is None or _fds_datasets_key != datasets_key:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        if len(dataset_ids) == 1 and dataset_ids[0] == "ealvaradob/phishing-dataset":
            # Fast path: load JSON directly for single default dataset
            fds = FederatedDataset(
                dataset="json",
                data_files={
                    "train": "https://huggingface.co/datasets/ealvaradob/phishing-dataset/resolve/main/urls.json"
                },
                partitioners={"train": partitioner},
            )
        else:
            # Load multiple datasets, merge, and create HF Dataset
            urls, labels = load_datasets(dataset_ids)
            print(f"\nTotal URLs before deduplication: {len(urls):,}")

            # Deduplicate based on normalized form
            seen = {}
            for url, label in zip(urls, labels):
                normalized = url_to_bytes(url)
                if normalized not in seen:
                    seen[normalized] = (url, label)
            urls = [v[0] for v in seen.values()]
            labels = [v[1] for v in seen.values()]
            print(f"After deduplication: {len(urls):,}")

            # Create HF Dataset from merged data
            merged_ds = HFDataset.from_dict({"text": urls, "label": labels})

            fds = FederatedDataset(
                dataset=merged_ds,
                partitioners={"train": partitioner},
            )

        _fds_datasets_key = datasets_key

    partition = fds.load_partition(partition_id)

    # Depending on flwr_datasets version, partition may be a DatasetDict with a "train" split
    hf_ds = (
        partition["train"]
        if isinstance(partition, dict) or hasattr(partition, "keys") and "train" in partition
        else partition
    )

    if fraction < 1.0:
        hf_ds = hf_ds.shuffle(seed=42).select(range(int(len(hf_ds) * fraction)))

    return _make_loaders(hf_ds, batch_size, device)


def load_local_data(data_path: str, batch_size: int, device: torch.device):
    """Load local data."""
    return _make_loaders(load_from_disk(data_path), batch_size, device)
