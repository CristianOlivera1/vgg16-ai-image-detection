"""Create a balanced, reproducible train/validation/test dataset from CIFAKE."""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = Path(r"C:\Users\USUARIO\Downloads\train")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_PER_CLASS = 5_000
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_SIZES = {"train": 0.8, "val": 0.1}
CLASS_DIRECTORIES = {"real": "REAL", "fake": "FAKE"}


def prepare_dataset(
    source_dir: Path,
    output_dir: Path = OUTPUT_DIR,
    images_per_class: int = IMAGES_PER_CLASS,
    seed: int = SEED,
) -> dict[str, dict[str, int]]:
    """Copy an equal, seeded sample of each class into 80/10/10 splits."""
    if images_per_class <= 0:
        raise ValueError("images_per_class must be greater than zero.")

    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset source directory not found: {source_dir}")
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. "
            "Move or rename it before preparing a new dataset."
        )

    rng = random.Random(seed)
    selected_files: dict[str, list[Path]] = {}
    for source_class, output_class in CLASS_DIRECTORIES.items():
        class_dir = source_dir / source_class
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Expected class directory not found: {class_dir}"
            )

        image_files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(image_files) < images_per_class:
            raise ValueError(
                f"Class '{source_class}' contains {len(image_files)} supported "
                f"images; {images_per_class} are required."
            )
        selected_files[output_class] = rng.sample(image_files, images_per_class)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    counts: dict[str, dict[str, int]] = {}
    try:
        for output_class, files in selected_files.items():
            rng.shuffle(files)
            train_end = int(images_per_class * SPLIT_SIZES["train"])
            val_end = train_end + int(images_per_class * SPLIT_SIZES["val"])
            split_files = {
                "train": files[:train_end],
                "val": files[train_end:val_end],
                "test": files[val_end:],
            }
            counts[output_class] = {}

            for split, images in split_files.items():
                destination = staging_dir / split / output_class
                destination.mkdir(parents=True, exist_ok=True)
                for image_path in images:
                    shutil.copy2(image_path, destination / image_path.name)
                counts[output_class][split] = len(images)

        if output_dir.exists():
            output_dir.rmdir()
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare 10,000 balanced CIFAKE images in train/val/test."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=f"Directory containing real/ and fake/ folders (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducible sampling (default: {SEED})",
    )
    args = parser.parse_args()

    print(f"Fuente: {args.source_dir}")
    print(f"Destino: {OUTPUT_DIR}")
    counts = prepare_dataset(args.source_dir, seed=args.seed)
    for split in ("train", "val", "test"):
        real_count = counts["REAL"][split]
        fake_count = counts["FAKE"][split]
        print(
            f"{split}: {real_count + fake_count:,} imágenes "
            f"(REAL={real_count:,}, FAKE={fake_count:,})"
        )
    print("Preparación completada. El dataset original no se modificó.")


if __name__ == "__main__":
    main()
