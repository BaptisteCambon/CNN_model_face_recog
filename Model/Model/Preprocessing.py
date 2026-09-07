"""Shared image/box preprocessing.

Both training (model_training.py) and live inference (face_tracker.py) MUST
use the exact same resize strategy, or the model learns one geometry and is
evaluated on another. This module is the single source of truth for that
letterbox (aspect-ratio-preserving, black-padded) transform, and for mapping
normalized YOLO-style boxes [cx, cy, w, h] between the original image and the
letterboxed square used by the model.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


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