"""Shared VGG16 model, checkpoint, and device helpers."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "vgg16_cifake_best.pth"
CLASS_NAMES = ("FAKE", "REAL")
MODEL_NAME = "vgg16_gap_v1"
PRETRAINED_WEIGHTS = VGG16_Weights.IMAGENET1K_V1
VGG16_FEATURES = 512


def build_model(pretrained: bool) -> nn.Module:
    """Build VGG16 with frozen convolutional features and a binary head."""
    weights = PRETRAINED_WEIGHTS if pretrained else None
    model = vgg16(weights=weights)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(VGG16_FEATURES, len(CLASS_NAMES)),
    )
    return model


def get_device(name: str) -> torch.device:
    """Resolve auto/cpu/cuda and fail explicitly if CUDA was requested but absent."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a project checkpoint and return its model in evaluation mode."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("model_name") != MODEL_NAME:
        raise ValueError(
            f"Unsupported model architecture in checkpoint: "
            f"{checkpoint.get('model_name')!r}"
        )
    if tuple(checkpoint.get("class_names", ())) != CLASS_NAMES:
        raise ValueError(
            f"Checkpoint classes must be {CLASS_NAMES}; got "
            f"{checkpoint.get('class_names')!r}."
        )

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
