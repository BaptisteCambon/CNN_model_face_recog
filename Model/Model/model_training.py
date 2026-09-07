from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from architecture import CustomFaceDetector
from loss import FaceDetectionLoss
from preprocessing import box_to_letterboxed, compute_letterbox_params, paste_into_canvas

# python Model\Model\model_training.py --epochs 30 --batch-size 32 --learning-rate 0.001
# 
# Model\Model\face_detector.pt
#
# python Model\Model\model_training.py `
#   --epochs 50 `
#   --batch-size 16 `
#   --output Model\Model\face_detector.pt

class FaceDataset(Dataset):
    """Loads one normalized YOLO box (or an empty label) per image.

    Segmentation-style YOLO labels are converted to their enclosing box so
    they remain usable by this single-box detector.
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
        confidence = torch.tensor([1.0 if lines else 0.0], dtype=torch.float32)
        box = torch.zeros(4, dtype=torch.float32)
        if lines:
            if len(lines) != 1:
                raise ValueError(f"Expected one YOLO box or polygon label: {label_path}")
            try:
                values = [float(value) for value in lines[0]]
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
                    f"Expected one YOLO box or polygon label with normalized values: {label_path}"
                )
            if torch.any(torch.tensor(values) < 0) or torch.any(torch.tensor(values) > 1):
                raise ValueError(f"Box values must be normalized to [0, 1]: {label_path}")

            # Remap the box from the original image's coordinates into the
            # letterboxed canvas' coordinates (same frame the model trains on).
            box = torch.tensor(
                box_to_letterboxed(raw_box, width, height, params, self.image_size),
                dtype=torch.float32,
            )

            if self.training and random.random() < 0.5:
                canvas = canvas[:, ::-1, :].copy()
                box[0] = 1.0 - box[0]

        image_tensor = torch.from_numpy(canvas).permute(2, 0, 1)
        return image_tensor, confidence, box


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
    for images, target_confs, target_boxes in dataloader:
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
    return total_loss / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the custom face detector.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "Images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "face_detector.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = FaceDataset(args.data / "train", training=True)
    valid_dataset = FaceDataset(args.data / "valid")
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