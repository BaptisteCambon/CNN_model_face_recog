# 1/ Import dependencies
from __future__ import annotations
from typing import NamedTuple
import numpy as np

# 2/ The network shrinks the image by half four times, so a 128×128 
# image becomes an 8×8 grid of "cells" by the time it reaches the output.
IMAGE_SIZE = 128
GRID_SIZE = IMAGE_SIZE // 16

# 3/ Letterbox resizing: scale the image to fit in a square canvas,
# preserving aspect ratio, and pad the rest with black. 
class LetterboxParams(NamedTuple):
    scale: float
    resized_width: int
    resized_height: int
    pad_x: int
    pad_y: int

# 4/ Compute the letterbox parameters for a given image size and target size. It returns
# the scale factor, the resized width and height, and the padding in x and y directions.
def compute_letterbox_params(width: int, height: int, target_size: int) -> LetterboxParams:
    """
    Compute the letterbox parameters for a given image size and target size.

    Scale : picks the smaller of the two possible scale factors, so that after resizing,
            neither dimension overflows 128. Using min (not max) is the key trick that preserves
            aspect ratio instead of stretching the image.
    resized_width/height : the dimensions of the image after resizing, rounded to the nearest integer.
    pad_x/pad_y : how much empty space is left on each axis once the resized image is centered in the 128*128 square.
    """
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pad_x = (target_size - resized_width) // 2
    pad_y = (target_size - resized_height) // 2
    return LetterboxParams(scale, resized_width, resized_height, pad_x, pad_y)

# 5/ Makes an all-black (zeros) 128×128 square, then copies the resized image into the middle of it using array slicing.
def paste_into_canvas(resized: np.ndarray, params: LetterboxParams, target_size: int) -> np.ndarray:
    """
    Paste an already-resized image into a black target_size x target_size canvas.

    resized : the image after resizing, with shape (resized_height, resized_width, channels).
    params : the LetterboxParams computed for this image.
    target_size : the size of the target canvas.
    """
    canvas = np.zeros((target_size, target_size, resized.shape[2]), dtype=resized.dtype)
    canvas[
        params.pad_y : params.pad_y + params.resized_height,
        params.pad_x : params.pad_x + params.resized_width,
    ] = resized
    return canvas

# 6/ This maps the original bounding boxes into the letterboxed canvas coordinates.
def box_to_letterboxed(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    params: LetterboxParams,
    target_size: int,
) -> tuple[float, float, float, float]:
    """
    Map a normalized [cx, cy, w, h] box from the original image into the
    normalized coordinates of the letterboxed target_size x target_size
    canvas. So cx=0.5, cy=0.5 means "dead center of the image," regardless of
    the original aspect ratio.

    box : a tuple of (cx, cy, w, h) in normalized coordinates (0-1) of the original image.
    width : the original width of the image.
    height : the original height of the image.
    params : the LetterboxParams computed for this image.
    target_size : the size of the target canvas.
    """
    cx, cy, w, h = box
    cx_px, cy_px = cx * width, cy * height
    w_px, h_px = w * width, h * height
    cx_new = (cx_px * params.scale + params.pad_x) / target_size
    cy_new = (cy_px * params.scale + params.pad_y) / target_size
    w_new = (w_px * params.scale) / target_size
    h_new = (h_px * params.scale) / target_size
    return cx_new, cy_new, w_new, h_new

# 7/ This is the inverse of box_to_letterboxed: it maps a normalized box predicted on the
# letterboxed canvas back into pixel coordinates of the original width x height frame.
def box_from_letterboxed(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    params: LetterboxParams,
    target_size: int,
) -> tuple[float, float, float, float]:
    """
    Inverse of box_to_letterboxed: map a normalized box predicted on the
    letterboxed canvas back into pixel coordinates of the original width x height frame.

    box : a tuple of (cx, cy, w, h) in normalized coordinates (0-1) of the letterboxed canvas.
    width : the original width of the image.
    height : the original height of the image.
    params : the LetterboxParams computed for this image.
    target_size : the size of the target canvas.
    """
    cx_new, cy_new, w_new, h_new = box
    cx_px = (cx_new * target_size - params.pad_x) / params.scale
    cy_px = (cy_new * target_size - params.pad_y) / params.scale
    w_px = (w_new * target_size) / params.scale
    h_px = (h_new * target_size) / params.scale
    return cx_px, cy_px, w_px, h_px

# 8/ Context : Each picture is divided into a grid of cells. 
# - Each cell predicts whether it contains a face (confidence) and the bounding box for that face (box).
def encode_grid_targets(
    boxes: list[tuple[float, float, float, float]],
    grid_size: int = GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    boxes: list of [cx, cy, w, h], normalized to the letterboxed canvas (0-1).
    Returns (confidence, box) arrays shaped (1, grid_size, grid_size) and
    (4, grid_size, grid_size).

    Each cell in the grid can only claim one face. If multiple faces fall into
    the same cell, only the first one is kept. This is a limitation of the
    coarse grid representation, but in practice it works well enough for
    most applications.

    confidence: a binary array where 1 indicates the presence of a face in that cell.
    box: an array containing the bounding box parameters (offset_x, offset_y, w, h) for the face in that cell.
    """
    confidence = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    box_target = np.zeros((4, grid_size, grid_size), dtype=np.float32)
    for cx, cy, w, h in boxes:
        # Picks the cell in the grid where the face center falls. 
        grid_x = min(int(cx * grid_size), grid_size - 1)
        grid_y = min(int(cy * grid_size), grid_size - 1)
        if confidence[0, grid_y, grid_x] == 1.0:
            continue  # cell already claimed by another face : keep the first
        confidence[0, grid_y, grid_x] = 1.0
        # Where inside that cell the center of the face actually sits
        offset_x = cx * grid_size - grid_x
        offset_y = cy * grid_size - grid_y
        # The box is stored as (offset_x, offset_y, w, h) in the cell's coordinates.
        box_target[:, grid_y, grid_x] = (offset_x, offset_y, w, h)
    return confidence, box_target

# 9/ It turns the network's raw grid output back into real boxes.
def decode_grid_predictions(
    confidence: np.ndarray,
    box: np.ndarray,
    grid_size: int = GRID_SIZE,
    confidence_threshold: float = 0.5,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """
    Decodes the grid predictions into a list of detected faces. 

    confidence: a (grid_size, grid_size) array of predicted face presence scores.
    box: a (4, grid_size, grid_size) array of predicted bounding box parameters
    grid_size: the size of the grid (default is GRID_SIZE).
    confidence_threshold: the minimum score for a detection to be considered valid.
    """
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

# 10/ Context : the grid is coarse, so a single real face can sometimes make two neighboring
# cells both fire above threshold (their offsets point to roughly the same spot).
# Non-max suppression (NMS) cleans that up.
def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes. box_a and box_b are tuples of (cx, cy, w, h)
    in normalized coordinates (0-1). Returns the IoU value, which is a float between 0 and 1.
    A value of 0 means no overlap, and a value of 1 means perfect overlap.
    """
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

# 11/ _iou alone isn't always enough to catch duplicate detections, because a coarse grid can split one
# face across two cells with slightly different predicted box sizes.
def _center_distance_ratio(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """
    Final ratio: near 0 means "centers are essentially on top of each other" (likely the same face);
    larger values mean the centers are far apart relative to the boxes' size.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    avg_size = ((aw + ah) / 2 + (bw + bh) / 2) / 2
    if avg_size <= 0:
        return float("inf")
    return distance / avg_size

# 12/ Non-max suppression (NMS) is a post-processing step to remove duplicate detections. It keeps the
# highest-scoring detection and removes others that overlap significantly with it.
def non_max_suppression(
    detections: list[tuple[float, tuple[float, float, float, float]]],
    iou_threshold: float = 0.4,
    center_distance_ratio_threshold: float = 0.5,
) -> list[tuple[float, tuple[float, float, float, float]]]:
    """
    Greedy NMS: since a coarse grid can fire in neighboring cells for the
    same face, this merges duplicate detections down to one box per face.

    Two detections are merged if EITHER:
      - their IoU exceeds iou_threshold (standard NMS), or
      - their centers are close relative to their size, even with low IoU
        (center_distance_ratio_threshold). This catches the case where one
        face straddles two grid cells that predict noticeably different box
        sizes -- e.g. a side profile -- so plain IoU-based NMS alone leaves
        both as separate detections.
    """
    remaining = sorted(detections, key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, tuple[float, float, float, float]]] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            d
            for d in remaining
            if _iou(best[1], d[1]) < iou_threshold
            and _center_distance_ratio(best[1], d[1]) >= center_distance_ratio_threshold
        ]
    return kept
