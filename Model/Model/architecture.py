import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import GRID_SIZE


class CustomFaceDetector(nn.Module):
    """Grid-based detector: predicts a GRID_SIZE x GRID_SIZE map instead of a
    single global box, so multiple faces in one image can each be detected by
    their own cell. See preprocessing.py for the encode/decode logic that
    turns this grid output into (and from) actual face boxes."""

    def __init__(self):
        super(CustomFaceDetector, self).__init__()

        # Extraction de caractéristiques (Convolutions)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fourth stage: brings a 128x128 input down to an 8x8 grid (128/16=8),
        # matching preprocessing.GRID_SIZE. If you change IMAGE_SIZE or the
        # number of pooling stages, GRID_SIZE must be updated to match.
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Per-cell detection head: 1 confidence logit + 4 box values per cell,
        # implemented as a 1x1 conv so it's applied independently at every
        # grid location (no flattening / no single global prediction).
        self.head = nn.Conv2d(in_channels=128, out_channels=5, kernel_size=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x)))  # 64x64   -> 32x32
        x = self.pool(F.relu(self.conv3(x)))  # 32x32   -> 16x16
        x = self.pool(F.relu(self.conv4(x)))  # 16x16   -> 8x8  (== GRID_SIZE)

        out = self.head(x)  # (batch, 5, GRID_SIZE, GRID_SIZE)
        assert out.shape[-1] == GRID_SIZE, (
            f"Conv stack produced a {out.shape[-1]}x{out.shape[-1]} grid but "
            f"preprocessing.GRID_SIZE is {GRID_SIZE}; keep these in sync."
        )

        # Return logits for confidence; loss and inference apply sigmoid explicitly.
        confidence = out[:, 0:1, :, :]
        bbox = torch.sigmoid(out[:, 1:5, :, :])  # offset_x, offset_y, w, h all in [0, 1]

        return confidence, bbox