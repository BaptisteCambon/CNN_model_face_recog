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
from preprocessing import box_from_letterboxed, compute_letterbox_params, paste_into_canvas  # noqa: E402


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


def predict_face(
    model: CustomFaceDetector,
    frame: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray]:
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

    normalized_box = tuple(box[0].cpu().numpy().tolist())
    center_x, center_y, box_width, box_height = box_from_letterboxed(
        normalized_box, width, height, params, IMAGE_SIZE
    )
    x1 = int(center_x - box_width / 2)
    y1 = int(center_y - box_height / 2)
    x2 = int(center_x + box_width / 2)
    y2 = int(center_y + box_height / 2)
    coordinates = np.clip([x1, y1, x2, y2], [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1])
    return float(torch.sigmoid(confidence[0, 0]).item()), coordinates


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
        help="Minimum face confidence to draw a box (default: 0.5).",
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

            confidence, coordinates = predict_face(model, frame, device)
            if confidence >= args.threshold:
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