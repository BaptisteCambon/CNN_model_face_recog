"""Shared image/box preprocessing.

Both training (model_training.py) and live inference (face_tracker.py) MUST
use the exact same resize strategy, or the model learns one geometry and is
evaluated on another. This module is the single source of truth for that
letterbox (aspect-ratio-preserving, black-padded) transform, for mapping
normalized YOLO-style boxes [cx, cy, w, h] between the original image and the
letterboxed square used by the model, and for the grid encode/decode logic
that lets the model predict multiple faces per image.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Single source of truth for the detection grid. CustomFaceDetector's conv
# stack downsamples IMAGE_SIZE by exactly 16x (four stride-2 pools), so
# GRID_SIZE must equal IMAGE_SIZE // 16. If you change the conv stack, update
# this too -- architecture.py imports it directly rather than hardcoding it.
IMAGE_SIZE = 128
GRID_SIZE = IMAGE_SIZE // 16


class LetterboxParams(NamedTuple):
    scale: float
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int


def compute_letterbox_params(width: int, height: int, target_size: int) -> LetterboxParams:
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_x = (target_size - resized_width) // 2
    pad_y = (target_size - resized_height) // 2
    return LetterboxParams(scale, resized_width, resized_height, pad_x, pad_y)


def paste_into_canvas(resized: np.ndarray, params: LetterboxParams, target_size: int) -> np.ndarray:
    """Paste an already-resized image into a black target_size x target_size canvas."""
    canvas = np.zeros((target_size, target_size, resized.shape[2]), dtype=resized.dtype)
    canvas[
        params.pad_y : params.pad_y + params.resized_height,
        params.pad_x : params.pad_x + params.resized_width,
    ] = resized
    return canvas


def box_to_letterboxed(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    params: LetterboxParams,
    target_size: int,
) -> tuple[float, float, float, float]:
    """Map a normalized [cx, cy, w, h] box from the original image into the
    normalized coordinates of the letterboxed target_size x target_size canvas."""
    cx, cy, w, h = box
    cx_px, cy_px = cx * width, cy * height
    w_px, h_px = w * width, h * height
    cx_new = (cx_px * params.scale + params.pad_x) / target_size
    cy_new = (cy_px * params.scale + params.pad_y) / target_size
    w_new = (w_px * params.scale) / target_size
    h_new = (h_px * params.scale) / target_size
    return cx_new, cy_new, w_new, h_new


def box_from_letterboxed(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    params: LetterboxParams,
    target_size: int,
) -> tuple[float, float, float, float]:
    """Inverse of box_to_letterboxed: map a normalized box predicted on the
    letterboxed canvas back into pixel coordinates of the original width x height frame."""
    cx_new, cy_new, w_new, h_new = box
    cx_px = (cx_new * target_size - params.pad_x) / params.scale
    cy_px = (cy_new * target_size - params.pad_y) / params.scale
    w_px = (w_new * target_size) / params.scale
    h_px = (h_new * target_size) / params.scale
    return cx_px, cy_px, w_px, h_px


# ---------------------------------------------------------------------------
# Grid encode / decode for multi-box detection.
#
# The model predicts a GRID_SIZE x GRID_SIZE map. Each cell predicts:
#   - a confidence logit: "is a face centered in this cell?"
#   - a box [offset_x, offset_y, w, h] where offset_x/offset_y are the face
#     center's position WITHIN the cell (0-1), and w/h are the face size
#     normalized to the whole IMAGE_SIZE canvas (0-1), matching the encoding
#     used by the original single-box design so the loss can stay similar.
#
# Only one face can be assigned per cell (a standard YOLO-style simplification).
# With GRID_SIZE=8 on a 128x128 input, each cell covers a 16x16px region --
# fine for typical webcam framing, but two face centers landing in the same
# cell (very small, very close faces) will collide; the later one is dropped.
# ---------------------------------------------------------------------------


def encode_grid_targets(
    boxes: list[tuple[float, float, float, float]],
    grid_size: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """boxes: list of [cx, cy, w, h], normalized to the letterboxed canvas (0-1).
    Returns (confidence, box) arrays shaped (1, grid_size, grid_size) and
    (4, grid_size, grid_size)."""
    confidence = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    box_target = np.zeros((4, grid_size, grid_size), dtype=np.float32)
    for cx, cy, w, h in boxes:
        grid_x = min(int(cx * grid_size), grid_size - 1)
        grid_y = min(int(cy * grid_size), grid_size - 1)
        if confidence[0, grid_y, grid_x] == 1.0:
            continue  # cell already claimed by another face; keep the first
        confidence[0, grid_y, grid_x] = 1.0
        offset_x = cx * grid_size - grid_x
        offset_y = cy * grid_size - grid_y
        box_target[:, grid_y, grid_x] = (offset_x, offset_y, w, h)
    return confidence, box_target


def decode_grid_predictions(
    confidence: np.ndarray,
    box: np.ndarray,
    grid_size: int = GRID_SIZE,
    confidence_threshold: float = 0.5,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """confidence: (grid_size, grid_size) probabilities (already sigmoided).
    box: (4, grid_size, grid_size) predictions (already sigmoided).
    Returns a list of (score, [cx, cy, w, h]) in letterboxed-canvas-normalized
    coordinates (0-1), one per cell above threshold. Caller should run NMS."""
    detections = []
    for grid_y in range(grid_size):
        for grid_x in range(grid_size):
            score = float(confidence[grid_y, grid_x])
            if score < confidence_threshold:
                continue
            offset_x, offset_y, w, h = box[:, grid_y, grid_x]
            cx = (grid_x + offset_x) / grid_size
            cy = (grid_y + offset_y) / grid_size
            detections.append((score, (float(cx), float(cy), float(w), float(h))))
    return detections


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
    ax2, ay2 = box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2
    bx1, by1 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
    bx2, by2 = box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def non_max_suppression(
    detections: list[tuple[float, tuple[float, float, float, float]]],
    iou_threshold: float = 0.4,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Greedy NMS: since a coarse grid can fire in neighboring cells for the
    same face, this merges duplicate detections down to one box per face."""
    remaining = sorted(detections, key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if _iou(best[1], d[1]) < iou_threshold]
    return kept