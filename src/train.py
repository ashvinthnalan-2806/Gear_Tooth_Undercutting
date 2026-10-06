import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

from src.preprocessing import get_dataloaders, inspect_image_dataset
from src.model import build_gear_cnn_model

def train_image_model(
    dataset_dir: str = "dataset",
    model_name: str = "mobilenet_v2",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    save_path: str = "models/gear_cnn_model.pt"
) -> Dict[str, Any]:
    """
    Trains the transfer learning CNN on gear tooth photographs.
    Strictly enforces real data availability: will not fabricate fake predictions.
    """
    counts = inspect_image_dataset(dataset_dir)
    print("=" * 65)
    print("GEAR TOOTH UNDERCUTTING DETECTOR - IMAGE TRAINING PIPELINE")
    print("=" * 65)
    print(f"Dataset directory: {os.path.abspath(dataset_dir)}")
    print(f"  • Undercut images found:    {counts['undercut']}")
    print(f"  • No Undercut images found: {counts['no_undercut']}")
    print(f"  • Total images:             {counts['total']}")

    if counts["undercut"] == 0 or counts["no_undercut"] == 0:
        error_msg = (
            "\n[!] DATASET MISSING: Cannot train image model without labeled photographs.\n"
            "    Please place gear images into the following directories before training:\n"
            f"      1. Undercut Gears:    {os.path.abspath(os.path.join(dataset_dir, 'undercut'))}\n"
            f"      2. Normal Gears:      {os.path.abspath(os.path.join(dataset_dir, 'no_undercut'))}\n"
            "    Supported formats: .jpg, .jpeg, .png, .bmp, .webp\n"
        )
        print(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "counts": counts
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, idx_to_class = get_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        val_split=0.2
    )

    # Calculate class weights for CrossEntropyLoss to handle class imbalance
    n_undercut = counts["undercut"]
    n_no_undercut = counts["no_undercut"]
    total = counts["total"]
    weight_no_undercut = total / (2.0 * n_no_undercut)
    weight_undercut = total / (2.0 * n_undercut)
    class_weights = torch.tensor([weight_no_undercut, weight_undercut], dtype=torch.float).to(device)
    print(f"Computed loss weights (imbalance handling): [No Undercut: {weight_no_undercut:.3f}, Undercut: {weight_undercut:.3f}]")

    model = build_gear_cnn_model(model_name=model_name, num_classes=2, pretrained=True, freeze_backbone=False)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    best_metrics = {}

    print(f"\nStarting {epochs} epochs of transfer learning ({model_name})...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probas = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                probas = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probas.extend(probas.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        prec = precision_score(all_targets, all_preds, zero_division=0)
        rec = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_targets, all_probas)
        except Exception:
            auc = 0.5

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Acc: {acc:.4f} | F1: {f1:.4f} | Rec: {rec:.4f} | AUC: {auc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                "epoch": epoch,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
                "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist()
            }
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "idx_to_class": idx_to_class,
                "best_metrics": best_metrics,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, save_path)

    print("\n" + "=" * 65)
    print(f"Training Complete! Best model saved to: {os.path.abspath(save_path)}")
    print(f"Best Validation F1-Score: {best_f1:.4f}")
    print("=" * 65)

    return {
        "status": "success",
        "best_metrics": best_metrics,
        "save_path": save_path
    }
