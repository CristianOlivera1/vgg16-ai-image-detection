"""Train a frozen-feature VGG16 classifier on the prepared CIFAKE dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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
SEED = 42
FINE_TUNE_LAST_BLOCK_START = 24


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


def save_history(history: list[dict[str, float]], run_name: str | None) -> None:
    """Write the epoch metrics and a loss/accuracy plot to results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{run_name}" if run_name else ""
    history_path = RESULTS_DIR / f"training_history{suffix}.csv"
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
    figure_name = "loss_accuracy_curves.png"
    if run_name:
        figure_name = f"loss_accuracy_{run_name}.png"
    figure.savefig(RESULTS_DIR / figure_name, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--strategy",
        choices=("head-only", "finetune-last-block"),
        default="head-only",
        help="Freeze the whole backbone or fine-tune its final convolutional block.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Classifier learning rate (default: 1e-3 head-only, 1e-4 fine-tuning).",
    )
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for the unfrozen VGG16 block.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--run-name",
        help="Unique name for this run's checkpoint and result files.",
    )
    args = parser.parse_args()
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-4 if args.strategy == "finetune-last-block" else 1e-3
    if args.epochs <= 0 or args.batch_size <= 0 or learning_rate <= 0:
        parser.error("epochs, batch-size and learning-rate must be greater than zero")
    if args.workers < 0:
        parser.error("workers cannot be negative")
    if args.backbone_learning_rate <= 0:
        parser.error("backbone-learning-rate must be greater than zero")
    if args.run_name and (
        Path(args.run_name).name != args.run_name
        or args.run_name in {".", ".."}
    ):
        parser.error("run-name must be a filename-safe identifier, not a path")

    fine_tune_last_block = args.strategy == "finetune-last-block"
    if fine_tune_last_block and not args.run_name:
        parser.error(
            "fine-tuning requires --run-name so its checkpoint and metrics "
            "do not overwrite another experiment"
        )
    checkpoint_path = (
        MODEL_DIR / f"{args.run_name}.pth"
        if args.run_name
        else MODEL_DIR / "vgg16_cifake_best.pth"
    )
    if args.run_name:
        result_paths = (
            checkpoint_path,
            RESULTS_DIR / f"training_history_{args.run_name}.csv",
            RESULTS_DIR / f"loss_accuracy_{args.run_name}.png",
        )
        existing_paths = [path for path in result_paths if path.exists()]
        if existing_paths:
            parser.error(
                "run-name already has output files; choose a new name: "
                + ", ".join(str(path) for path in existing_paths)
            )

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
    model = build_model(
        pretrained=True,
        fine_tune_last_block=fine_tune_last_block,
    ).to(device)
    if fine_tune_last_block:
        features = model.features
        if not isinstance(features, nn.Sequential):
            raise TypeError("Expected VGG16 features to be an nn.Sequential module.")
        optimizer = AdamW(
            [
                {
                    "params": features[FINE_TUNE_LAST_BLOCK_START:].parameters(),
                    "lr": args.backbone_learning_rate,
                },
                {"params": model.classifier.parameters(), "lr": learning_rate},
            ]
        )
    else:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = AdamW(trainable_parameters, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {device}")
    print(f"Clases: {class_to_idx}")
    if fine_tune_last_block:
        print(
            "Backbone: VGG16 ImageNet1K_V1; se ajusta el último bloque "
            f"(features[{FINE_TUNE_LAST_BLOCK_START}:]) con lr="
            f"{args.backbone_learning_rate:g}; clasificador con lr="
            f"{learning_rate:g}."
        )
    else:
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
                    "strategy": args.strategy,
                    "learning_rate": learning_rate,
                    "backbone_learning_rate": (
                        args.backbone_learning_rate if fine_tune_last_block else None
                    ),
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                    "image_size": list(PRETRAINED_WEIGHTS.transforms().crop_size),
                },
                checkpoint_path,
            )
            print(f"Mejor checkpoint guardado en {checkpoint_path}")

    save_history(history, args.run_name)
    print(f"Historial y curvas guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
