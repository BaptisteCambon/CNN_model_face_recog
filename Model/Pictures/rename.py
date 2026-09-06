"""Rename image files and their matching labels with consecutive numbers.

Example:
    python rename.py "path/to/images" "path/to/labels"

Each image must have a text label with the same original filename stem.
For example, ``cat_01.jpg`` must have a matching ``cat_01.txt`` label.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSIONS_IMAGE = {".png", ".jpg", ".jpeg"}


def renommer_fichiers(dossier_images: Path, dossier_labels: Path) -> None:
    """Rename all matching image/label pairs in the two supplied folders."""
    # Sort by filename so that the same input always produces the same numbering.
    images = sorted(
        (
            fichier
            for fichier in dossier_images.iterdir()
            if fichier.is_file() and fichier.suffix.lower() in EXTENSIONS_IMAGE
        ),
        key=lambda fichier: fichier.name.lower(),
    )

    paires = []
    labels_attendus = set()
    for image in images:
        # Labels are expected to be text files with the image's original stem.
        label = dossier_labels / f"{image.stem}.txt"
        labels_attendus.add(label.name)
        if not label.is_file():
            raise FileNotFoundError(
                f"Label introuvable pour l'image {image.name}: {label.name}"
            )
        paires.append((image, label))

    labels_sans_image = sorted(
        fichier.name
        for fichier in dossier_labels.iterdir()
        if fichier.is_file()
        and fichier.suffix.lower() == ".txt"
        and fichier.name not in labels_attendus
    )
    if labels_sans_image:
        raise ValueError(
            "Labels sans image correspondante: " + ", ".join(labels_sans_image)
        )

    # Rename to temporary names first. This prevents collisions when a destination
    # name (for example, "1.jpg") already exists in one of the folders.
    fichiers_temporaires = []
    for index, (image, label) in enumerate(paires, start=1):
        image_temp = image.with_name(f".renommage_{index}{image.suffix}")
        label_temp = label.with_name(f".renommage_{index}.txt")
        image.rename(image_temp)
        label.rename(label_temp)
        fichiers_temporaires.append((image_temp, label_temp, index))

    for image_temp, label_temp, index in fichiers_temporaires:
        image_destination = dossier_images / f"{index}{image_temp.suffix}"
        label_destination = dossier_labels / f"{index}.txt"
        image_temp.rename(image_destination)
        label_temp.rename(label_destination)

    print(f"{len(paires)} paire(s) renommee(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rename matching images and labels with consecutive numbers. "
            "Pass the image folder first and the label folder second."
        ),
        epilog=(
            "Example: python rename.py path/to/images path/to/labels"
        ),
    )
    parser.add_argument(
        "dossier_images",
        type=Path,
        help="Folder containing the images to rename.",
    )
    parser.add_argument(
        "dossier_labels",
        type=Path,
        help="Folder containing the matching .txt labels.",
    )
    args = parser.parse_args()

    # Validate both inputs before changing any filenames.
    if not args.dossier_images.is_dir():
        raise NotADirectoryError(
            f"Image folder not found: {args.dossier_images}"
        )
    if not args.dossier_labels.is_dir():
        raise NotADirectoryError(
            f"Label folder not found: {args.dossier_labels}"
        )

    renommer_fichiers(args.dossier_images, args.dossier_labels)


if __name__ == "__main__":
    main()
