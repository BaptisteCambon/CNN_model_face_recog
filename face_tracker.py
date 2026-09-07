"""Track a face from the default camera with the custom detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Model" / "Model"
IMAGE_SIZE = 128

sys.path.insert(0, str(MODEL_DIR))
from architecture import CustomFaceDetector  # noqa: E402
from preprocessing import (  # noqa: E402
    GRID_SIZE,
    box_from_letterboxed,
    compute_letterbox_params,
    decode_grid_predictions,
    non_max_suppression,
    paste_into_canvas,
)


def load_model(model_path: Path, device: torch.device) -> CustomFaceDetector:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported model checkpoint format: {model_path}")

    model = CustomFaceDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_faces(
    model: CustomFaceDetector,
    frame: np.ndarray,
    device: torch.device,
    confidence_threshold: float = 0.5,
    nms_iou_threshold: float = 0.4,
    nms_center_distance_ratio_threshold: float = 0.5,
) -> list[tuple[float, np.ndarray]]:
    """Returns a list of (confidence, [x1, y1, x2, y2]) detections, one per
    face found in the frame, already de-duplicated with NMS."""
    height, width = frame.shape[:2]
    # Preserve the camera aspect ratio using the exact same letterbox transform
    # as FaceDataset in model_training.py (see preprocessing.py). Any mismatch
    # here means the model is evaluated on a different geometry than it was
    # trained on.
    params = compute_letterbox_params(width, height, IMAGE_SIZE)
    resized = cv2.resize(frame, (params.resized_width, params.resized_height))
    letterboxed = paste_into_canvas(resized, params, IMAGE_SIZE)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    pixels = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    image = pixels.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        confidence, box = model(image)
        confidence = torch.sigmoid(confidence)

    confidence_grid = confidence[0, 0].cpu().numpy()  # (GRID_SIZE, GRID_SIZE)
    box_grid = box[0].cpu().numpy()  # (4, GRID_SIZE, GRID_SIZE)

    raw_detections = decode_grid_predictions(
        confidence_grid, box_grid, GRID_SIZE, confidence_threshold
    )
    detections = non_max_suppression(
        raw_detections, nms_iou_threshold, nms_center_distance_ratio_threshold
    )

    results = []
    for score, normalized_box in detections:
        center_x, center_y, box_width, box_height = box_from_letterboxed(
            normalized_box, width, height, params, IMAGE_SIZE
        )
        x1 = int(center_x - box_width / 2)
        y1 = int(center_y - box_height / 2)
        x2 = int(center_x + box_width / 2)
        y2 = int(center_y + box_height / 2)
        coordinates = np.clip(
            [x1, y1, x2, y2], [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1]
        )
        results.append((score, coordinates))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Track faces with the custom trained model.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_DIR / "face_detector.pt",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum face confidence to draw a box (default: 0.5). Raise this "
        "to reduce false positives on hands/clothing/background, at the cost "
        "of possibly missing harder-to-see faces.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.4,
        help="IoU threshold for merging overlapping detections (default: 0.4).",
    )
    parser.add_argument(
        "--nms-center-distance",
        type=float,
        default=0.5,
        help="Merge detections whose centers are closer than this, relative to "
        "box size, even if their IoU is low (default: 0.5). Lower this if a "
        "single face is still showing up as two boxes (e.g. side profiles); "
        "raise it if legitimately separate nearby faces get merged into one.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera}")

    print(f"Tracking with {device}. Press q or Esc to stop.")
    try:
        while True:
            captured, frame = camera.read()
            if not captured:
                raise RuntimeError("Unable to read a frame from the camera")

            detections = predict_faces(
                model,
                frame,
                device,
                confidence_threshold=args.threshold,
                nms_iou_threshold=args.nms_iou,
                nms_center_distance_ratio_threshold=args.nms_center_distance,
            )
            for confidence, coordinates in detections:
                x1, y1, x2, y2 = (int(value) for value in coordinates)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Face {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Custom face tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()