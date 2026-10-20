"""Shared VGG16 model, checkpoint, and device helpers."""

from pathlib import Path
from typing import Any
from PIL import Image, ImageOps

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.vgg import VGG

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "vgg16_cifake_best.pth"
CLASS_NAMES = ("FAKE", "REAL")
MODEL_NAME = "vgg16_gap_v1"
PRETRAINED_WEIGHTS = VGG16_Weights.IMAGENET1K_V1
VGG16_FEATURES = 512
DATASET_SIZE = 32

def prepare_image(image: Image.Image) -> Image.Image:
    """Recorte cuadrado central y reducción a 32×32, como en CIFAKE."""
    return ImageOps.fit(
        image.convert("RGB"),
        (DATASET_SIZE, DATASET_SIZE),
        method=Image.Resampling.BICUBIC,
    )

def build_model(pretrained: bool, fine_tune_last_block: bool = False) -> VGG:
    """Build VGG16 with a binary head and optionally unfreeze its final block."""
    weights = PRETRAINED_WEIGHTS if pretrained else None
    model: VGG = vgg16(weights=weights)
    features = model.features
    if not isinstance(features, nn.Sequential):
        raise TypeError("Expected VGG16 features to be an nn.Sequential module.")

    for parameter in features.parameters():
        parameter.requires_grad = False

    if fine_tune_last_block:
        for parameter in features[24:].parameters():
            parameter.requires_grad = True

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
