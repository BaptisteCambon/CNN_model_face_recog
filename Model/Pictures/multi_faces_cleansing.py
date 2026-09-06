"""Supprime les labels de plus de deux lignes et leurs images correspondantes."""

from __future__ import annotations

import argparse
from pathlib import Path


def supprimer_fichiers(dossier_images: Path, dossier_labels: Path) -> None:
    labels_supprimes = 0
    images_supprimees = 0

    labels = sorted(
        (
            fichier
            for fichier in dossier_labels.iterdir()
            if fichier.is_file() and fichier.suffix.lower() == ".txt"
        ),
        key=lambda fichier: fichier.name.lower(),
    )

    for label in labels:
        if len(label.read_text(encoding="utf-8").splitlines()) <= 1:
            continue

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
            "Supprime les labels contenant plus de deux lignes et "
            "leurs images JPG correspondantes."
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

    supprimer_fichiers(args.images, args.labels)


if __name__ == "__main__":
    main()
