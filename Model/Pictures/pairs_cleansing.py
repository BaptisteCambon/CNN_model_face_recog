"""Supprime les labels et images qui n'ont pas de fichier correspondant."""

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png"}


def supprimer_fichiers_orphelins(
    dossier_images: Path, dossier_labels: Path
) -> None:
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

    labels_orphelins = sorted(
        (fichier for nom, fichier in labels.items() if nom not in images),
        key=lambda fichier: fichier.name.lower(),
    )
    images_orphelines = sorted(
        (fichier for nom, fichier in images.items() if nom not in labels),
        key=lambda fichier: fichier.name.lower(),
    )

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
            "Supprime les labels sans image correspondante et les images "
            "sans label correspondant."
        )
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Images" / "valid" / "images",
        help="Dossier des images (par defaut: Model/Images/valid/images).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Images" / "valid" / "labels",
        help="Dossier des labels (par defaut: Model/Images/valid/labels).",
    )
    args = parser.parse_args()

    if not args.images.is_dir():
        raise NotADirectoryError(f"Dossier d'images introuvable: {args.images}")
    if not args.labels.is_dir():
        raise NotADirectoryError(f"Dossier de labels introuvable: {args.labels}")

    supprimer_fichiers_orphelins(args.images, args.labels)


if __name__ == "__main__":
    main()
