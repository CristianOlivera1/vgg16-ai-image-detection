"""Evaluate a trained VGG16 classifier on the held-out CIFAKE test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import matplotlib
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_utils import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    PRETRAINED_WEIGHTS,
    PROJECT_ROOT,
    get_device,
    load_checkpoint,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be greater than zero")
    if args.workers < 0:
        parser.error("workers cannot be negative")

    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = ImageFolder(TEST_DIR, transform=PRETRAINED_WEIGHTS.transforms())
    if dataset.class_to_idx != {"FAKE": 0, "REAL": 1}:
        raise ValueError(
            f"Unexpected test classes: {dataset.class_to_idx}. "
            "Expected FAKE and REAL."
        )
    if checkpoint.get("class_to_idx") != dataset.class_to_idx:
        raise ValueError("Test class mapping does not match the model checkpoint.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    all_predictions: list[int] = []
    all_targets: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(labels.tolist())

    report = cast(
        dict[str, Any],
        classification_report(
            all_targets,
            all_predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    )
    matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(len(CLASS_NAMES))),
    )
    counts = Counter(label for _, label in dataset.samples)
    results = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_samples": len(dataset),
        "class_counts": {
            CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))
        },
        "metrics": report,
        "confusion_matrix": {
            "labels": list(CLASS_NAMES),
            "values": matrix.tolist(),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "metrics_report.json"
    report_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(CLASS_NAMES),
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title("VGG16 - conjunto de prueba")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"F1 macro: {report['macro avg']['f1-score']:.4f}")
    print(f"Reporte guardado en {report_path}")
    print(f"Matriz de confusión guardada en {RESULTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
