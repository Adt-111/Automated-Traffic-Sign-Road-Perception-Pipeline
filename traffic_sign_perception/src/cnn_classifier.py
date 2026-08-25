"""
CNN for GTSRB traffic-sign classification: 3 conv blocks (Conv/BN/ReLU/
MaxPool/Dropout) into two dense layers. Also has the GTSRB Dataset loader
(with a synthetic fallback so you can run this without downloading
anything), a training loop, and eval/plotting helpers.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm

import config


# CNN architecture

class TrafficSignCNN(nn.Module):
    """
    3 conv blocks (32 -> 64 -> 128 filters, each Conv/BN/ReLU/MaxPool/
    Dropout) then Dense(256) -> Dropout -> Dense(num_classes). Built for
    CNN_IMAGE_SIZE x CNN_IMAGE_SIZE input (48x48 by default, which becomes
    6x6 after the three pools).
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, dropout: float = config.CNN_DROPOUT) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )

        # Infer the flattened feature size dynamically so CNN_IMAGE_SIZE can be
        # changed in config.py without manually recomputing this dimension.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE)
            flat_dim = self._forward_conv(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax class probabilities (batch, num_classes) for already-preprocessed input."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# Preprocessing

def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((config.CNN_IMAGE_SIZE, config.CNN_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ]
    )


def preprocess_bgr_crop(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Converts a raw OpenCV BGR crop (as produced by `classical_detector.Candidate.crop`)
    into a normalized (1, 3, H, W) tensor ready for `TrafficSignCNN`.
    """
    rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
    pil_image = Image.fromarray(rgb.copy())
    transform = build_transforms(train=False)
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


# Dataset

class GTSRBDataset(Dataset):
    """
    Reads the standard Kaggle GTSRB folder layout:
        Train/<class_id>/*.png   (used for both train() and val() via random_split)
        Test/*.png + Test.csv    (ClassId, Path columns)
    """

    def __init__(self, samples: list[tuple[str, int]], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform or build_transforms(train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    @classmethod
    def from_train_dir(cls, train_dir: str = config.TRAIN_DIR) -> list[tuple[str, int]]:
        samples = []
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"GTSRB train directory not found at '{train_dir}'. "
                f"Set the GTSRB_ROOT environment variable or use build_synthetic_dataset()."
            )
        for class_dir in sorted(os.listdir(train_dir)):
            class_path = os.path.join(train_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            try:
                class_id = int(class_dir)
            except ValueError:
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".ppm")):
                    samples.append((os.path.join(class_path, fname), class_id))
        return samples

    @classmethod
    def from_test_csv(cls, data_root: str = config.DATA_ROOT, test_csv: str = config.TEST_CSV) -> list[tuple[str, int]]:
        samples = []
        with open(test_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(data_root, row["Path"])
                samples.append((img_path, int(row["ClassId"])))
        return samples


def build_synthetic_dataset(
    n_classes: int = 5, n_per_class: int = 40, image_size: int = 64, seed: int = config.SEED
) -> list[tuple[np.ndarray, int]]:
    """
    Fake "traffic sign" dataset -- colored shapes with distinct geometry
    per class, so training/eval works without downloading GTSRB. Returns
    in-memory (image, label) pairs; use InMemoryDataset below, not
    GTSRBDataset, to wrap these.
    """
    rng = np.random.default_rng(seed)
    shapes = ["circle", "triangle", "square", "octagon", "diamond"][:n_classes]
    colors = [(220, 20, 20), (20, 20, 220), (220, 180, 20), (20, 160, 20), (160, 20, 160)][:n_classes]

    samples = []
    for class_id, (shape, color) in enumerate(zip(shapes, colors)):
        for _ in range(n_per_class):
            img = np.full((image_size, image_size, 3), 200, dtype=np.uint8)
            jitter_color = tuple(int(np.clip(c + rng.integers(-20, 20), 0, 255)) for c in color)
            _draw_shape(img, shape, jitter_color, rng)
            noise = rng.integers(0, 15, size=img.shape, dtype=np.uint8)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            samples.append((img, class_id))
    return samples


def _draw_shape(img: np.ndarray, shape: str, color: tuple[int, int, int], rng: np.random.Generator) -> None:
    """Draws a simple filled shape onto `img` in-place (used only by build_synthetic_dataset)."""
    import cv2

    h, w = img.shape[:2]
    cx, cy = w // 2 + int(rng.integers(-4, 4)), h // 2 + int(rng.integers(-4, 4))
    r = min(h, w) // 3

    if shape == "circle":
        cv2.circle(img, (cx, cy), r, color, thickness=-1)
    elif shape == "square":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), color, thickness=-1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "diamond":
        pts = np.array([[cx, cy - r], [cx + r, cy], [cx, cy + r], [cx - r, cy]], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == "octagon":
        pts = []
        for i in range(8):
            angle = np.pi / 8 + i * np.pi / 4
            pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
        cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)


class InMemoryDataset(Dataset):
    """Wraps in-memory (np.ndarray image, label) pairs, e.g. from `build_synthetic_dataset`."""

    def __init__(self, samples: list[tuple[np.ndarray, int]], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform or build_transforms(train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_array, label = self.samples[idx]
        pil_image = Image.fromarray(image_array)
        return self.transform(pil_image), label


# Training loop

@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def train_cnn(
    model: TrafficSignCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = config.CNN_NUM_EPOCHS,
    lr: float = config.CNN_LEARNING_RATE,
    weight_decay: float = config.CNN_WEIGHT_DECAY,
    device: torch.device = config.DEVICE,
) -> TrainingHistory:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    history = TrainingHistory()
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, _, _ = evaluate_cnn(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, config.MODEL_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")

    return history


@torch.no_grad()
def evaluate_cnn(
    model: TrafficSignCNN, loader: DataLoader, criterion: nn.Module | None = None, device: torch.device = config.DEVICE
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (avg_loss, accuracy, all_predictions, all_labels)."""
    model.to(device)
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    running_loss, total = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    avg_loss = running_loss / max(total, 1)

    return avg_loss, accuracy, all_preds, all_labels


# Visualization: loss curves + confusion matrix

def plot_training_curves(history: TrainingHistory, save_path: str = config.TRAINING_CURVES_PATH):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history.train_loss) + 1)
    axes[0].plot(epochs, history.train_loss, label="Train Loss", marker="o")
    axes[0].plot(epochs, history.val_loss, label="Val Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.train_acc, label="Train Acc", marker="o")
    axes[1].plot(epochs, history.val_acc, label="Val Acc", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: dict[int, str] | None = None,
    save_path: str = config.CONFUSION_MATRIX_PATH,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tick_labels = [class_names.get(l, str(l))[:15] if class_names else str(l) for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), max(5, len(labels) * 0.35)))
    sns.heatmap(cm, annot=len(labels) <= 15, fmt="d", cmap="Blues", xticklabels=tick_labels, yticklabels=tick_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":  # pragma: no cover
    print("Running cnn_classifier.py smoke test with a synthetic dataset...")
    config.set_seed()

    synthetic_samples = build_synthetic_dataset(n_classes=5, n_per_class=30, image_size=64)
    n_val = int(0.2 * len(synthetic_samples))
    n_train = len(synthetic_samples) - n_val

    full_dataset = InMemoryDataset(synthetic_samples, transform=build_transforms(train=True))
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(config.SEED)
    )
    # Validation should use eval-time (non-augmented) transforms.
    val_ds.dataset = InMemoryDataset(synthetic_samples, transform=build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = TrafficSignCNN(num_classes=5)
    history = train_cnn(model, train_loader, val_loader, num_epochs=3, device=torch.device("cpu"))

    plot_training_curves(history, save_path="/tmp/training_curves_smoke_test.png")
    print("Saved training curves to /tmp/training_curves_smoke_test.png")

    _, _, preds, labels = evaluate_cnn(model, val_loader, device=torch.device("cpu"))
    plot_confusion_matrix(labels, preds, save_path="/tmp/confusion_matrix_smoke_test.png")
    print("Saved confusion matrix to /tmp/confusion_matrix_smoke_test.png")
    print(f"Final val accuracy: {history.val_acc[-1]:.4f}")
    print("cnn_classifier.py smoke test passed.")
