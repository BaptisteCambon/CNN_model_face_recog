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

# python Model\Model\model_training.py --epochs 30 --batch-size 32 --learning-rate 0.001
# 
# Model\Model\face_detector.pt
#
# python Model\Model\model_training.py `
#   --epochs 50 `
#   --batch-size 16 `
#   --output Model\Model\face_detector.pt

class FaceDataset(Dataset):
    """Loads one normalized YOLO box (or an empty label) per image."""

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
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        lines = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        confidence = torch.tensor([1.0 if lines else 0.0], dtype=torch.float32)
        box = torch.zeros(4, dtype=torch.float32)
        if lines:
            if len(lines) != 1 or len(lines[0]) != 5:
                raise ValueError(f"Expected one YOLO label with 5 values: {label_path}")
            box = torch.tensor([float(value) for value in lines[0][1:]], dtype=torch.float32)
            if torch.any(box < 0) or torch.any(box > 1):
                raise ValueError(f"Box values must be normalized to [0, 1]: {label_path}")
            if self.training and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                pixels = np.asarray(image, dtype=np.float32) / 255.0
                image_tensor = torch.from_numpy(pixels).permute(2, 0, 1)
                box[0] = 1.0 - box[0]
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