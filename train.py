import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from logo_dataset import LogoDataset 
from model import SimpleCenterNet
from utils import focal_loss

# --- CONFIGURAZIONE ---
save_dir = 'checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_img_dir = "datasetLOGOS/train"
train_ann_file = os.path.join(train_img_dir, "_annotations.coco.json")


def train():
    # --- IPERPARAMETRI ---
    batch_size = 16
    learning_rate = 1e-3
    epochs = 35
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join(save_dir, "centernet_logo_simplified.pth")
    
    # Flag per decidere se usare MSE o Focal Loss 
    use_focal = True 

    print(f"Addestramento su: {device}")

    # 1. Caricamento Dataset
    train_dataset = LogoDataset(train_img_dir, train_ann_file)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Modello, Loss e Ottimizzatore
    model = SimpleCenterNet().to(device)
    
    # MSE per la heatmap (opzione iniziale) o L1 per l'offset
    criterion_hm = nn.MSELoss() if not use_focal else focal_loss
    criterion_reg = nn.L1Loss(reduction='sum') 
    
    #ottimizzatore Adam che aggiorna tutti i parametri del modello (sia quelli della heatmap che dell'offset)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    # 3. Ciclo di Addestramento
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            t_hm = targets['hm'].to(device) #target reale per la heatmap
            t_off = targets['reg'].to(device) #target reale per l'offset

            # Forward pass 
            p_hm, p_off = model(inputs)

            # --- CALCOLO LOSS ---
            
            # 1. Heatmap Loss
            if use_focal:
                loss_heatmap = focal_loss(p_hm, t_hm)
            else:
                loss_heatmap = criterion_hm(p_hm, t_hm)
            
            # 2. Offset Loss (calcolata solo dove c'è un oggetto reale)
            #la loss della offset me la calcolo solo sui pixel dove t_hm è 1 (cioè dove c'è un logo) 
            mask = (t_hm == 1).float() 
            num_objects = mask.sum() + 1e-4#evito la divisione per zero se non ci sono oggetti nell'immagine con questo parametro, la loss dell'offset sarà molto bassa ma non zero
            
            # Applichiamo la maschera su entrambi i canali dell'offset (dx, dy)
            mask_off = mask.repeat(1, 2, 1, 1) # Da [B, 1, H, W] a [B, 2, H, W]
            loss_offset = criterion_reg(p_off * mask_off, t_off * mask_off) / num_objects

            # Somma totale: diamo peso maggiore alla heatmap per stabilizzare il training
            total_loss = (loss_heatmap * 1.0) + (loss_offset * 0.8)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoca [{epoch+1}/{epochs}], Loss Totale: {running_loss/len(train_loader):.6f}")

    # Salvataggio del modello ottimizzato
    torch.save(model.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")

if __name__ == "__main__":
    train()