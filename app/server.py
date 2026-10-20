"""Servidor local: sirve index.html y expone el clasificador VGG16 como API."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from model_utils import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT,
    PRETRAINED_WEIGHTS,
    PROJECT_ROOT,
    get_device,
    load_checkpoint,
    prepare_image,
)

MODEL_DIR = PROJECT_ROOT / "models"
APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

device = get_device("auto")
transform = PRETRAINED_WEIGHTS.transforms()

app = FastAPI(title="VGG16 CIFAKE")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])


def available_models() -> list[str]:
    return sorted(path.stem for path in MODEL_DIR.glob("*.pth"))


@lru_cache(maxsize=4)
def get_model(name: str) -> torch.nn.Module:
    model, _ = load_checkpoint(MODEL_DIR / f"{name}.pth", device)
    return model


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/api/models")
def models() -> dict:
    return {
        "models": available_models(),
        "default": DEFAULT_CHECKPOINT.stem,
        "device": device.type,
    }


@app.post("/api/predict")
def predict(model: str = DEFAULT_CHECKPOINT.stem, file: UploadFile = File(...)) -> dict:
    if model not in available_models():
        raise HTTPException(404, f"Modelo no encontrado: {model}")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "La imagen supera los 20 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "El archivo no es una imagen válida.")

    tensor = transform(prepare_image(image)).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = get_model(model)(tensor).softmax(dim=1)[0].cpu().tolist()

    best = max(range(len(CLASS_NAMES)), key=probabilities.__getitem__)
    return {
        "filename": file.filename,
        "label": CLASS_NAMES[best],
        "confidence": probabilities[best],
        "probabilities": dict(zip(CLASS_NAMES, probabilities)),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)