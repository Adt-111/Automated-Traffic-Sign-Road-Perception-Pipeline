"""
Trains the CNN classifier. Uses real GTSRB data if config.TRAIN_DIR
exists, otherwise falls back to the synthetic generator automatically.

    python train_cnn.py               # auto-detects
    python train_cnn.py --synthetic   # force synthetic (fast demo)
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split

import config
from src.cnn_classifier import (
    GTSRBDataset,
    InMemoryDataset,
    TrafficSignCNN,
    build_synthetic_dataset,
    build_transforms,
    evaluate_cnn,
    plot_confusion_matrix,
    plot_training_curves,
    train_cnn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TrafficSignCNN classifier")
    parser.add_argument("--synthetic", action="store_true", help="Force use of the synthetic dataset")
    parser.add_argument("--epochs", type=int, default=config.CNN_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.CNN_BATCH_SIZE)
    args = parser.parse_args()

    config.set_seed()

    use_synthetic = args.synthetic or not os.path.isdir(config.TRAIN_DIR)
    if use_synthetic:
        print(
            f"[train_cnn] Real GTSRB data not found at '{config.TRAIN_DIR}' "
            "(or --synthetic was passed) — training on a synthetic 5-class demo dataset instead.\n"
            "[train_cnn] To train on real GTSRB data, download it and set the GTSRB_ROOT "
            "environment variable to its folder."
        )
        samples = build_synthetic_dataset(n_classes=5, n_per_class=200, image_size=64)
        n_val = int(config.CNN_VAL_SPLIT * len(samples))
        n_train = len(samples) - n_val

        train_full = InMemoryDataset(samples, transform=build_transforms(train=True))
        val_full = InMemoryDataset(samples, transform=build_transforms(train=False))
        generator = torch.Generator().manual_seed(config.SEED)
        train_idx, val_idx = random_split(range(len(samples)), [n_train, n_val], generator=generator)

        train_ds = torch.utils.data.Subset(train_full, train_idx.indices)
        val_ds = torch.utils.data.Subset(val_full, val_idx.indices)
        num_classes = 5
    else:
        print(f"[train_cnn] Loading real GTSRB data from '{config.TRAIN_DIR}'...")
        samples = GTSRBDataset.from_train_dir()
        n_val = int(config.CNN_VAL_SPLIT * len(samples))
        n_train = len(samples) - n_val

        generator = torch.Generator().manual_seed(config.SEED)
        train_samples_idx, val_samples_idx = random_split(range(len(samples)), [n_train, n_val], generator=generator)
        train_samples = [samples[i] for i in train_samples_idx.indices]
        val_samples = [samples[i] for i in val_samples_idx.indices]

        train_ds = GTSRBDataset(train_samples, transform=build_transforms(train=True))
        val_ds = GTSRBDataset(val_samples, transform=build_transforms(train=False))
        num_classes = config.NUM_CLASSES

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"[train_cnn] train={len(train_ds)} val={len(val_ds)} num_classes={num_classes} device={config.DEVICE}")

    model = TrafficSignCNN(num_classes=num_classes)
    history = train_cnn(model, train_loader, val_loader, num_epochs=args.epochs, device=config.DEVICE)

    plot_training_curves(history)
    print(f"[train_cnn] Saved training curves to {config.TRAINING_CURVES_PATH}")

    _, final_acc, preds, labels = evaluate_cnn(model, val_loader, device=config.DEVICE)
    class_names = config.GTSRB_CLASSES if not use_synthetic else None
    plot_confusion_matrix(labels, preds, class_names=class_names)
    print(f"[train_cnn] Saved confusion matrix to {config.CONFUSION_MATRIX_PATH}")
    print(f"[train_cnn] Final validation accuracy: {final_acc:.4f}")
    print(f"[train_cnn] Best model checkpoint saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
