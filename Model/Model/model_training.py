from __future__ import annotations
import argparse
import random
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from architecture import CustomFaceDetector
from loss import FaceDetectionLoss
from preprocessing import (
    GRID_SIZE,
    box_to_letterboxed,
    compute_letterbox_params,
    encode_grid_targets,
    paste_into_canvas,
)


def download_s3_dataset(s3_uri: str, cache_root: Path) -> Path:
    """Download the S3 train/valid tree into matching local directories."""
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError(
            "S3 datasets require boto3. Install it with: python -m pip install boto3"
        ) from error

    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected an S3 URI such as s3://bucket/prefix, got: {s3_uri}")

    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/").rstrip("/")
    dataset_root = cache_root
    print(f"Checking S3 dataset {s3_uri} and downloading files to {dataset_root}...", flush=True)
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    checked = 0

    pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/" if prefix else "")
    for page_number, page in enumerate(pages, start=1):
        objects = page.get("Contents", [])
        print(f"Checking S3 listing page {page_number} ({len(objects)} objects)...", flush=True)
        for object_info in objects:
            key = object_info["Key"]
            relative_key = key[len(prefix) + 1 :] if prefix else key
            relative_path = Path(relative_key)
            if (
                len(relative_path.parts) < 3
                or relative_path.parts[0] not in {"train", "valid"}
                or relative_path.parts[1] not in {"images", "labels"}
                or not relative_path.name
            ):
                continue

            destination = dataset_root / relative_path
            destination.resolve().relative_to(dataset_root.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.is_file() or destination.stat().st_size != object_info["Size"]:
                client.download_file(bucket, key, str(destination))
                downloaded += 1
                if downloaded % 100 == 0:
                    print(f"Downloaded {downloaded} S3 files so far...", flush=True)
            checked += 1

    if downloaded:
        print(f"Downloaded {downloaded} dataset files from {s3_uri} to {dataset_root}", flush=True)
    else:
        print(f"Using cached S3 dataset at {dataset_root} ({checked} files checked)", flush=True)
    return dataset_root


# python Model\Model\model_training.py --data s3://baptistecambon/face_recog --epochs 30 --batch-size 32 --learning-rate 0.001
# 
# Model\Model\face_detector.pt
#
# python Model\Model\model_training.py `
#  --data s3://baptistecambon/face_recog `
#  --epochs 50 `
#  --batch-size 16 `
#  --learning-rate 0.001

class FaceDataset(Dataset):
    """Loads every normalized YOLO box in a label file (zero or more faces
    per image) and encodes them into a grid target for the multi-box
    detector. Segmentation-style YOLO labels are converted to their
    enclosing box.
    """

    def __init__(
        self, split_dir: Path, image_size: int = 128, training: bool = False
    ) -> None:
        self.images_dir = split_dir / "images"
        self.labels_dir = split_dir / "labels"
        self.image_size = image_size
        self.training = training
        self.images = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (self.labels_dir / f"{path.stem}.txt").is_file()
        )
        if not self.images:
            raise ValueError(f"No labelled images found in {split_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        original = Image.open(image_path).convert("RGB")
        width, height = original.size

        # Letterbox (aspect-ratio-preserving, black-padded) instead of a plain
        # stretch resize. This MUST match the transform used at inference time
        # in face_tracker.py, or the box regressor is trained on one geometry
        # and evaluated on another.
        params = compute_letterbox_params(width, height, self.image_size)
        resized = original.resize(
            (params.resized_width, params.resized_height), Image.Resampling.BILINEAR
        )
        canvas = paste_into_canvas(
            np.asarray(resized, dtype=np.float32) / 255.0, params, self.image_size
        )

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        lines = [line.split() for line in label_path.read_text().splitlines() if line.strip()]

        letterboxed_boxes: list[tuple[float, float, float, float]] = []
        for line in lines:
            try:
                values = [float(value) for value in line]
            except ValueError as error:
                raise ValueError(f"YOLO label contains non-numeric values: {label_path}") from error
            if len(values) == 5:
                raw_box = tuple(values[1:])
            elif len(values) > 5 and len(values[1:]) % 2 == 0:
                points = torch.tensor(values[1:], dtype=torch.float32).reshape(-1, 2)
                min_corner = points.min(dim=0).values
                max_corner = points.max(dim=0).values
                center = (min_corner + max_corner) / 2
                size = max_corner - min_corner
                raw_box = tuple(torch.cat((center, size)).tolist())
            else:
                raise ValueError(
                    f"Expected a YOLO box or polygon label with normalized values: {label_path}"
                )
            if any(v < 0 or v > 1 for v in values):
                raise ValueError(f"Box values must be normalized to [0, 1]: {label_path}")

            # Remap each box from the original image's coordinates into the
            # letterboxed canvas' coordinates (same frame the model trains on).
            letterboxed_boxes.append(
                box_to_letterboxed(raw_box, width, height, params, self.image_size)
            )

        if self.training and letterboxed_boxes and random.random() < 0.5:
            canvas = canvas[:, ::-1, :].copy()
            letterboxed_boxes = [(1.0 - cx, cy, w, h) for cx, cy, w, h in letterboxed_boxes]

        confidence_grid, box_grid = encode_grid_targets(letterboxed_boxes, GRID_SIZE)

        image_tensor = torch.from_numpy(canvas).permute(2, 0, 1)
        return image_tensor, torch.from_numpy(confidence_grid), torch.from_numpy(box_grid)


def run_epoch(
    model: CustomFaceDetector,
    dataloader: DataLoader,
    criterion: FaceDetectionLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    phase = "Training" if training else "Validation"
    batch_count = len(dataloader)
    for batch_number, (images, target_confs, target_boxes) in enumerate(dataloader, start=1):
        images = images.to(device)
        target_confs = target_confs.to(device)
        target_boxes = target_boxes.to(device)
        if training:
            optimizer.zero_grad()
        pred_confs, pred_boxes = model(images)
        batch_loss = criterion(pred_confs, pred_boxes, target_confs, target_boxes)
        if training:
            batch_loss.backward()
            optimizer.step()
        total_loss += batch_loss.item()
        if batch_number % 100 == 0 or batch_number == batch_count:
            print(f"{phase}: batch {batch_number}/{batch_count}", flush=True)
    return total_loss / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom face detector.")
    parser.add_argument(
        "--data",
        default=str(Path(__file__).resolve().parent.parent / "Images"),
        help="Local dataset path or S3 URI, e.g. s3://baptistecambon/face_recog",
    )
    parser.add_argument(
        "--s3-cache",
        type=Path,
        default=Path.home() / ".cache" / "cnn_model_face_recog",
        help="Local dataset directory for S3 downloads (train/ and valid/ are created inside)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "face_detector.pt")
    args = parser.parse_args()

    data_root = (
        download_s3_dataset(args.data, args.s3_cache)
        if args.data.startswith("s3://")
        else Path(args.data)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Indexing train and validation datasets in {data_root}...", flush=True)
    train_dataset = FaceDataset(data_root / "train", training=True)
    valid_dataset = FaceDataset(data_root / "valid")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    train_positive = sum(
        bool((train_dataset.labels_dir / f"{path.stem}.txt").read_text().strip())
        for path in train_dataset.images
    )
    valid_positive = sum(
        bool((valid_dataset.labels_dir / f"{path.stem}.txt").read_text().strip())
        for path in valid_dataset.images
    )
    print(
        f"Dataset: {len(train_dataset)} train ({train_positive} positive, "
        f"{len(train_dataset) - train_positive} negative), "
        f"{len(valid_dataset)} validation ({valid_positive} positive, "
        f"{len(valid_dataset) - valid_positive} negative)"
    )
    if len(valid_dataset) - valid_positive < 10:
        print("Warning: validation has fewer than 10 negative images; metrics may be unreliable.")
    model = CustomFaceDetector().to(device)
    criterion = FaceDetectionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_valid_loss = float("inf")

    print(f"Training on {device} ({len(train_loader.dataset)} train, {len(valid_loader.dataset)} valid)")
    for epoch in range(1, args.epochs + 1):
        print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{args.epochs} - train: {train_loss:.4f} - valid: {valid_loss:.4f}")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict(), "valid_loss": valid_loss},
                args.output,
            )
            print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()