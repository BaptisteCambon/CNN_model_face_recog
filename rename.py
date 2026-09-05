"""Renomme les images et leurs labels avec des numeros consecutifs."""

from __future__ import annotations

import argparse
from pathlib import Path


EXTENSIONS_IMAGE = {".png", ".jpg", ".jpeg"}


def renommer_fichiers(dossier_images: Path, dossier_labels: Path) -> None:
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

    # Les noms temporaires eviteront les collisions avec les noms numeriques existants.
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
        description="Renomme les images et labels correspondants avec des numeros."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(__file__).resolve().parent / "Images" / "valid" / "images",
        help="Dossier des images (par defaut: Images/valid/images).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parent / "Images" / "valid" / "labels",
        help="Dossier des labels (par defaut: Images/valid/labels).",
    )
    args = parser.parse_args()

    if not args.images.is_dir():
        raise NotADirectoryError(f"Dossier d'images introuvable: {args.images}")
    if not args.labels.is_dir():
        raise NotADirectoryError(f"Dossier de labels introuvable: {args.labels}")

    renommer_fichiers(args.images, args.labels)


if __name__ == "__main__":
    main()
