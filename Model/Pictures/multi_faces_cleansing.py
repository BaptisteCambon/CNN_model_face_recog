"""Remove multi-face labels and their corresponding images.

Example:
    python multi_faces_cleansing.py "path/to/images" "path/to/labels"

Every label containing more than one line is considered a multi-face label.
The matching JPG image, when present, is removed as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def supprimer_fichiers(dossier_images: Path, dossier_labels: Path) -> None:
    """Delete multi-face labels and their matching JPG images."""
    labels_supprimes = 0
    images_supprimees = 0

    # Sort labels so that processing order is predictable and repeatable.
    labels = sorted(
        (
            fichier
            for fichier in dossier_labels.iterdir()
            if fichier.is_file() and fichier.suffix.lower() == ".txt"
        ),
        key=lambda fichier: fichier.name.lower(),
    )

    for label in labels:
        # A label with several lines describes more than one detected face.
        if len(label.read_text(encoding="utf-8").splitlines()) <= 1:
            continue

        # Image and label files are paired by their filename stem.
        image = dossier_images / f"{label.stem}.jpg"
        label.unlink()
        labels_supprimes += 1

        if image.is_file():
            image.unlink()
            images_supprimees += 1
        else:
            print(f"Image correspondante introuvable: {image.name}")

    print(
        f"{labels_supprimes} label(s) supprime(s), "
        f"{images_supprimees} image(s) supprimee(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove labels containing multiple lines and their corresponding "
            "JPG images. Pass the image folder first and the label folder second."
        ),
        epilog=(
            "Example: python multi_faces_cleansing.py "
            "path/to/images path/to/labels"
        ),
    )
    parser.add_argument(
        "dossier_images",
        type=Path,
        help="Folder containing the JPG images to clean.",
    )
    parser.add_argument(
        "dossier_labels",
        type=Path,
        help="Folder containing the .txt labels to inspect.",
    )
    args = parser.parse_args()

    # Validate both folders before deleting any files.
    if not args.dossier_images.is_dir():
        raise NotADirectoryError(
            f"Image folder not found: {args.dossier_images}"
        )
    if not args.dossier_labels.is_dir():
        raise NotADirectoryError(
            f"Label folder not found: {args.dossier_labels}"
        )

    supprimer_fichiers(args.dossier_images, args.dossier_labels)


if __name__ == "__main__":
    main()
