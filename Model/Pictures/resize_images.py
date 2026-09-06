"""Resize all images in the dataset image folder to 164x164 pixels.

Run from any directory with:
    python resize_images.py

An alternative image folder can be supplied as the first argument.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SIZE = (164, 164)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_IMAGE_FOLDER = (
    Path(__file__).resolve().parents[1] / "Images" / "All" / "images"
)


def redimensionner_images(dossier_images: Path) -> int:
    """Resize each supported image in ``dossier_images`` and return its count."""
    images = sorted(
        (
            fichier
            for fichier in dossier_images.iterdir()
            if fichier.is_file()
            and fichier.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda fichier: fichier.name.lower(),
    )

    resized_count = 0
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                output_format = (
                    "JPEG"
                    if image_path.suffix.lower() in {".jpg", ".jpeg"}
                    else "PNG"
                )
                image = ImageOps.exif_transpose(image)
                if output_format == "JPEG" and image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                resized.save(image_path, format=output_format)
        except UnidentifiedImageError:
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        resized_count += 1

    return resized_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize all JPG, JPEG, and PNG images to 164x164 pixels."
    )
    parser.add_argument(
        "dossier_images",
        type=Path,
        nargs="?",
        default=DEFAULT_IMAGE_FOLDER,
        help=f"Image folder (default: {DEFAULT_IMAGE_FOLDER})",
    )
    args = parser.parse_args()

    if not args.dossier_images.is_dir():
        raise NotADirectoryError(f"Image folder not found: {args.dossier_images}")

    nombre_images = redimensionner_images(args.dossier_images)
    print(f"{nombre_images} image(s) resized to 164x164.")


if __name__ == "__main__":
    main()
