"""Train a frozen-feature VGG16 classifier on the prepared CIFAKE dataset."""

from __future__ import annotations

import argparse
import csv

import matplotlib
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model_utils import (
    CLASS_NAMES,
    MODEL_NAME,
    PRETRAINED_WEIGHTS,
    PROJECT_ROOT,
    build_model,
    get_device,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = MODEL_DIR / "vgg16_cifake_best.pth"
SEED = 42


def create_loaders(
    batch_size: int,
    workers: int,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Load the training and validation splits with ImageNet preprocessing."""
    transform = PRETRAINED_WEIGHTS.transforms()
    datasets = {
        split: ImageFolder(DATA_DIR / split, transform=transform)
        for split in ("train", "val")
    }
    for split, dataset in datasets.items():
        if dataset.class_to_idx != {"FAKE": 0, "REAL": 1}:
            raise ValueError(
                f"Unexpected classes in {split}: {dataset.class_to_idx}. "
                "Expected FAKE and REAL."
            )

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        ),
    }
    return loaders["train"], loaders["val"], datasets["train"].class_to_idx


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation pass and return mean loss and accuracy."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    phase = "train" if training else "val"
    for images, labels in tqdm(loader, desc=phase, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_history(history: list[dict[str, float]]) -> None:
    """Write the epoch metrics and a loss/accuracy plot to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = RESULTS_DIR / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "loss_accuracy_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be greater than zero")
    if args.workers < 0:
        parser.error("workers cannot be negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = get_device(args.device)
    train_loader, val_loader, class_to_idx = create_loaders(
        args.batch_size,
        args.workers,
        device,
        args.seed,
    )
    model = build_model(pretrained=True).to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {device}")
    print(f"Clases: {class_to_idx}")
    print("Backbone: VGG16 ImageNet1K_V1 congelado; clasificador binario entrenable.")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        metrics = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(metrics)
        print(
            f"Época {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_name": MODEL_NAME,
                    "state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in model.state_dict().items()
                    },
                    "class_names": list(CLASS_NAMES),
                    "class_to_idx": class_to_idx,
                    "pretrained_weights": "IMAGENET1K_V1",
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "image_size": list(PRETRAINED_WEIGHTS.transforms().crop_size),
                },
                CHECKPOINT_PATH,
            )
            print(f"Mejor checkpoint guardado en {CHECKPOINT_PATH}")

    save_history(history)
    print(f"Historial y curvas guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
