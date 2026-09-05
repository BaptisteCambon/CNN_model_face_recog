import torch.optim as optim
import architecture.py
import loss.py 

# Initialisation du modèle, de la perte et de l'optimiseur
model = architecture.CustomFaceDetector()
criterion = loss.FaceDetectionLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(dataloader):
    model.train()
    total_loss = 0.0
    
    for images, target_confs, target_boxes in dataloader:
        # 1. Remise à zéro des gradients
        optimizer.zero_grad()
        
        # 2. Passage avant (Forward)
        pred_confs, pred_boxes = model(images)
        
        # 3. Calcul de l'erreur
        loss = criterion(pred_confs, pred_boxes, target_confs, target_boxes)
        
        # 4. Rétropropagation (Backward)
        loss.backward()
        
        # 5. Mise à jour des poids du réseau
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

print("Modèle prêt pour l'entraînement !")