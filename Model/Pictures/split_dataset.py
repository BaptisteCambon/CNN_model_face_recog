"""Split an image detection dataset into train, test, and valid folders.

The source dataset is expected to contain ``images`` and ``labels`` folders.
Images and their matching YOLO label files are copied together, and the
positive/negative ratio is kept approximately equal in every split.

Run from the repository root:
    python Model\\Pictures\\split_dataset.py

By default, this creates ``Model\\Images\\train``, ``test``, and ``valid`` using
an 80/15/5 split. Existing split folders must be removed explicitly with
``--overwrite``.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "test", "valid")
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "Images" / "All"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "Images"


def _split_count(total: int, ratios: tuple[float, float, float]) -> list[int]:
    """Return counts whose sum is total and which are closest to the ratios."""
    exact_counts = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact_counts]
    for index in sorted(
        range(len(counts)),
        key=lambda item: exact_counts[item] - counts[item],
        reverse=True,
    )[: total - sum(counts)]:
        counts[index] += 1
    return counts


def _split_items(
    items: list[Path],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, list[Path]]:
    rng.shuffle(items)
    counts = _split_count(len(items), ratios)
    result: dict[str, list[Path]] = {}
    start = 0
    for split, count in zip(SPLITS, counts):
        result[split] = items[start : start + count]
        start += count
    return result


def split_dataset(
    source: Path,
    output: Path,
    ratios: tuple[float, float, float],
    seed: int,
    overwrite: bool = False,
) -> dict[str, int]:
    """Copy the source dataset into reproducible, stratified split folders."""
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise NotADirectoryError(
            f"Source must contain images and labels folders: {source}"
        )

    images = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not images:
        raise ValueError(f"No supported images found in {images_dir}")

    missing_labels = [
        image for image in images if not (labels_dir / f"{image.stem}.txt").is_file()
    ]
    if missing_labels:
        names = ", ".join(image.name for image in missing_labels[:5])
        suffix = "..." if len(missing_labels) > 5 else ""
        raise FileNotFoundError(
            f"Missing labels for {len(missing_labels)} image(s): {names}{suffix}"
        )

    for split in SPLITS:
        split_dir = output / split
        if split_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{split_dir} already exists; use --overwrite to replace it"
                )
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    positive: list[Path] = []
    negative: list[Path] = []
    for image in images:
        label = labels_dir / f"{image.stem}.txt"
        if label.read_text(encoding="utf-8").strip():
            positive.append(image)
        else:
            negative.append(image)

    rng = random.Random(seed)
    assignments = {split: [] for split in SPLITS}
    for group in (positive, negative):
        group_assignments = _split_items(group, ratios, rng)
        for split in SPLITS:
            assignments[split].extend(group_assignments[split])

    counts: dict[str, int] = {}
    for split, split_images in assignments.items():
        split_images.sort(key=lambda path: path.name.lower())
        split_dir = output / split
        for image in split_images:
            shutil.copy2(image, split_dir / "images" / image.name)
            shutil.copy2(
                labels_dir / f"{image.stem}.txt",
                split_dir / "labels" / f"{image.stem}.txt",
            )
        counts[split] = len(split_images)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified train/test/valid image and label splits."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing train, test, and valid folders.",
    )
    args = parser.parse_args()

    ratios = (args.train_ratio, args.test_ratio, args.valid_ratio)
    if any(ratio <= 0 for ratio in ratios) or not math.isclose(
        sum(ratios), 1.0, abs_tol=1e-6
    ):
        parser.error("train, test, and valid ratios must be positive and sum to 1")

    counts = split_dataset(args.source, args.output, ratios, args.seed, args.overwrite)
    print(
        f"Split {sum(counts.values())} images: "
        + ", ".join(f"{split}={counts[split]}" for split in SPLITS)
    )


if __name__ == "__main__":
    main()
