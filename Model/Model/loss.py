import torch.nn as nn


class FaceDetectionLoss(nn.Module):
    def __init__(self):
        super(FaceDetectionLoss, self).__init__()
        self.bce = nn.BCELoss()      # Binary Cross Entropy (pour la présence du visage)
        self.mse = nn.MSELoss()      # Mean Squared Error (pour la précision du cadre)

    def forward(self, pred_conf, pred_box, target_conf, target_box):
        loss_conf = self.bce(pred_conf, target_conf)
        
        # On ne calcule la perte sur la boîte que si un visage est présent dans la vérité terrain
        mask = target_conf.expand_as(pred_box)
        loss_box = self.mse(pred_box * mask, target_box * mask)
        
        return loss_conf + (5.0 * loss_box) # Poids plus fort sur la localisation