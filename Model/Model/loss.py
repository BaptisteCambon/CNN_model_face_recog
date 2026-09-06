import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceDetectionLoss(nn.Module):
    def __init__(self):
        super(FaceDetectionLoss, self).__init__()
        self.mse = nn.MSELoss()      # Mean Squared Error (pour la précision du cadre)
        self.negative_weight = 2.0

    def forward(self, pred_conf, pred_box, target_conf, target_box):
        # Penalize false positives more heavily, which is important for camera
        # backgrounds and is numerically stable for logits.
        sample_weights = torch.where(
            target_conf == 0,
            torch.full_like(target_conf, self.negative_weight),
            torch.ones_like(target_conf),
        )
        loss_conf = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, weight=sample_weights
        )
        
        # On ne calcule la perte sur la boîte que si un visage est présent dans la vérité terrain
        mask = target_conf.expand_as(pred_box)
        loss_box = self.mse(pred_box * mask, target_box * mask)
        
        return loss_conf + (5.0 * loss_box) # Poids plus fort sur la localisation