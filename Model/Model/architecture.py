import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomFaceDetector(nn.Module):
    def __init__(self):
        # Fonction d'initialisation du modèle
        super(CustomFaceDetector, self).__init__()
        
        # Extraction de caractéristiques (Convolutions), in_channels = 3 pour les 3 canaux de couleurs, out_channels = 16 types de motifs, kernel_size = taille du filtre
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Pooling pour réduire la dimensionnalité et capturer les caractéristiques les plus importantes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Pour une image d'entrée de 128x128 -> après 3 maxpools : 16x16x64
        self.fc_input_dim = 64 * 16 * 16
        # Couche entièrement connectée pour combiner les caractéristiques extraites
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        
        # Têtes de sortie
        # Return logits; the loss and inference code apply sigmoid explicitly.
        self.classifier = nn.Linear(128, 1)
        self.box_regressor = nn.Linear(128, 4)   # Coordonnées [x, y, w, h] normalisées (0 à 1)

    def forward(self, x):
        # Passages convolutifs + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x))) # 128x128 -> 64x64
        x = self.pool(F.relu(self.conv2(x))) # 64x64   -> 32x32
        x = self.pool(F.relu(self.conv3(x))) # 32x32   -> 16x16
        
        # Aplatissement des données (Flatten)
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        
        # Calcul des deux prédictions
        confidence = self.classifier(x)
        bbox = torch.sigmoid(self.box_regressor(x))
        
        return confidence, bbox
