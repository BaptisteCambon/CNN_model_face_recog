import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetectionLoss(nn.Module):
    """Operates on the grid outputs from the multi-box CustomFaceDetector.

    Shapes: pred_conf/target_conf are (batch, 1, S, S); pred_box/target_box
    are (batch, 4, S, S). target_conf is 1 for a cell containing a face
    center and 0 otherwise (see preprocessing.encode_grid_targets).

    With a grid, most cells are empty in any given image (e.g. 1-2 faces out
    of 64 cells at an 8x8 grid), which is the opposite imbalance from the old
    single-box design. Instead of up-weighting negatives (which made sense
    for a single whole-image "is there a face" decision), we now DOWN-weight
    the many easy negative cells relative to the rare positive ones -- the
    standard YOLO-style lambda_obj / lambda_noobj split.
    """

    def __init__(self, noobj_weight: float = 0.5, obj_weight: float = 1.0, box_weight: float = 5.0):
        super(FaceDetectionLoss, self).__init__()
        self.noobj_weight = noobj_weight
        self.obj_weight = obj_weight
        self.box_weight = box_weight

    def forward(self, pred_conf, pred_box, target_conf, target_box):
        sample_weights = torch.where(
            target_conf == 0,
            torch.full_like(target_conf, self.noobj_weight),
            torch.full_like(target_conf, self.obj_weight),
        )
        loss_conf = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, weight=sample_weights
        )

        # Only compute box loss on cells that actually contain a face.
        object_mask = target_conf.expand_as(pred_box)
        num_objects = target_conf.sum().clamp(min=1.0)
        loss_box = F.mse_loss(
            pred_box * object_mask, target_box * object_mask, reduction="sum"
        ) / num_objects

        return loss_conf + (self.box_weight * loss_box)