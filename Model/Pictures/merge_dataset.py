"""Copies the train, test, and validation pairs into one dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SPLITS = ("train", "test", "valid")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def merge_dataset(source_dir: Path, destination_dir: Path) -> int:
    """Copy every image and its label into the combined dataset."""
    destination_images = destination_dir / "images"
    destination_labels = destination_dir / "labels"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    copied_pairs = 0
    for split in SPLITS:
        split_dir = source_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.is_dir():
            raise NotADirectoryError(f"Images folder not found: {images_dir}")
        if not labels_dir.is_dir():
            raise NotADirectoryError(f"Labels folder not found: {labels_dir}")

        images = sorted(
            (
                image
                for image in images_dir.iterdir()
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda image: image.name.lower(),
        )

        for image in images:
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(
                    f"Label not found for {split}/{image.name}: {label.name}"
                )

            destination_stem = f"{split}_{image.stem}"
            shutil.copy2(image, destination_images / f"{destination_stem}{image.suffix}")
            shutil.copy2(label, destination_labels / f"{destination_stem}.txt")
            copied_pairs += 1

    return copied_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy train, test, and valid image/label pairs into one folder."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Images",
        help="Dataset folder containing train, test, and valid.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Images" / "All",
        help="Destination folder for the combined dataset.",
    )
    args = parser.parse_args()

    if not args.source.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {args.source}")
    if args.destination.resolve() == args.source.resolve():
        raise ValueError("The destination must be different from the source folder.")

    copied_pairs = merge_dataset(args.source, args.destination)
    print(f"{copied_pairs} image/label pair(s) copied to {args.destination}.")


if __name__ == "__main__":
    main()
