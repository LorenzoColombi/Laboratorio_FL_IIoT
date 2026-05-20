"""
Phishing URL Classification - Training Script

Step 3: CNN Baseline Training
"""

import time

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device).float()

        optimizer.zero_grad()
        logits = model(input_ids).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, pos_weight, device):
    """Evaluate model on a dataset."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device).float()

        logits = model(input_ids).squeeze(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        total_loss += loss.item()
        num_batches += 1
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = {
        "loss": total_loss / num_batches,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics, all_labels, all_preds


def summarize_history(history):
    """Calculate the final train_loss, val_loss, and val_f1 from the last epoch.

    Args:
        history (list of dict): Output from the train() function.

    Returns:
        dict: A dictionary with the final metrics.
    """
    if not history:
        raise ValueError("History is empty.")

    last_epoch = history[-1]

    return {
        "avg_train_loss": last_epoch["train_loss"],
        "avg_val_loss": last_epoch["val_loss"],
        "avg_val_f1": last_epoch["val_f1"],
    }


def train(
    model,
    train_loader,
    val_loader,
    pos_weight,
    lr: float,
    device: torch.device,
    num_epochs: int = 20,
    patience: int = 5,
):
    """Training loop with early stopping."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time

        # Validate
        val_metrics, _, _ = evaluate(model, val_loader, pos_weight, device)
        val_f1 = val_metrics["f1"]

        # Learning rate scheduling
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.1f}s")
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_f1:.4f} | AUC: {val_metrics['auc']:.4f}"
        )
        print(f"LR: {current_lr:.2e}")

        # Track history
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_f1,
                "val_auc": val_metrics["auc"],
                "lr": current_lr,
            }
        )

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest model: epoch {best_epoch} with F1={best_f1:.4f}")
    return history, best_epoch
