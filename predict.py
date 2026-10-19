"""Predict whether an image is REAL or FAKE with a trained VGG16 model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from model_utils import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    PRETRAINED_WEIGHTS,
    get_device,
    load_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image file to classify")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    device = get_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    image_transform = PRETRAINED_WEIGHTS.transforms()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        input_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = model(input_tensor).softmax(dim=1)[0]
    predicted_index = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_index].item())
    print(f"Predicción: {CLASS_NAMES[predicted_index]}")
    print(f"Confianza del modelo: {confidence:.2%}")


if __name__ == "__main__":
    main()
