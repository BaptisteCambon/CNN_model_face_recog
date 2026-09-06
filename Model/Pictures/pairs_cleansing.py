"""Remove images and labels that do not have a matching pair.

Example:
    python pairs_cleansing.py "path/to/images" "path/to/labels"

Files are paired by filename stem. For example, ``person_01.jpg`` matches
``person_01.txt``. Any image or label without a matching partner is deleted.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png"}


def supprimer_fichiers_orphelins(
    dossier_images: Path, dossier_labels: Path
) -> None:
    """Delete images and labels whose filename stems do not match."""
    # Build lookups by filename stem so extensions do not affect matching.
    images = {
        fichier.stem: fichier
        for fichier in dossier_images.iterdir()
        if fichier.is_file() and fichier.suffix.lower() in EXTENSIONS_IMAGE
    }
    labels = {
        fichier.stem: fichier
        for fichier in dossier_labels.iterdir()
        if fichier.is_file() and fichier.suffix.lower() == ".txt"
    }

    # Sort before deletion so results are predictable and easier to inspect.
    labels_orphelins = sorted(
        (fichier for nom, fichier in labels.items() if nom not in images),
        key=lambda fichier: fichier.name.lower(),
    )
    images_orphelines = sorted(
        (fichier for nom, fichier in images.items() if nom not in labels),
        key=lambda fichier: fichier.name.lower(),
    )

    # Remove labels first, then images. Only files without a matching stem
    # are included in these lists, so valid image/label pairs are preserved.
    for label in labels_orphelins:
        label.unlink()

    for image in images_orphelines:
        image.unlink()

    print(
        f"{len(labels_orphelins)} label(s) orphelin(s) supprime(s), "
        f"{len(images_orphelines)} image(s) orpheline(s) supprimee(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove labels without a matching image and images without a "
            "matching label. Pass the image folder first and the label folder second."
        ),
        epilog=(
            "Example: python pairs_cleansing.py "
            "path/to/images path/to/labels"
        ),
    )
    parser.add_argument(
        "dossier_images",
        type=Path,
        help="Folder containing the images to check.",
    )
    parser.add_argument(
        "dossier_labels",
        type=Path,
        help="Folder containing the .txt labels to check.",
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

    supprimer_fichiers_orphelins(args.dossier_images, args.dossier_labels)


if __name__ == "__main__":
    main()
