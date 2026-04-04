import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader
# Assicurati che il nome del file del dataset sia corretto (es. dataset.py)
from logo_dataset import LogoDataset 
from model import SimpleCenterNet

# --- CONFIGURAZIONE ---
save_dir = 'checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

train_img_dir = "datasetLOGOS/train"
train_ann_file = os.path.join(train_img_dir, "_annotations.coco.json")

# --- LOSS FUNCTIONS ---
# Implementazione della Focal Loss per la heatmap (consigliata per problemi di rilevamento con classi sbilanciate)
def focal_loss(preds, targets, alpha=2, beta=4):
   
        # Clamping per evitare log(0) o log(1) - dà stabilità numerica
    preds = torch.clamp(preds, min=1e-4, max=1 - 1e-4)
    
    # Creazione di maschere per posizioni positive (dove target == 1) e negative (dove target < 1)
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    # Peso che attenua la loss vicino ai centri (usa la gaussiana nel target)
    neg_weights = torch.pow(1 - targets, beta)

    pos_loss = torch.log(preds) * torch.pow(1 - preds, alpha) * pos_inds
    neg_loss = torch.log(1 - preds) * torch.pow(preds, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    #la funzione somma tutte le perdite e le divide per num pos (il num dei loghi presenti nell'immagine) 
    #se non ci sono oggetti (num_pos == 0), restituisce solo la perdita negativa, evitando la divisione per zero.
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos

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
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    # 3. Ciclo di Addestramento
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            t_hm = targets['hm'].to(device)
            t_off = targets['reg'].to(device)

            # Forward pass (ritorna solo 2 output ora)
            p_hm, p_off = model(inputs)

            # --- CALCOLO LOSS ---
            
            # 1. Heatmap Loss
            if use_focal:
                loss_heatmap = focal_loss(p_hm, t_hm)
            else:
                loss_heatmap = criterion_hm(p_hm, t_hm)
            
            # 2. Offset Loss (calcolata solo dove c'è un oggetto reale)
            # Creiamo una maschera dai punti dove la heatmap target è esattamente 1
            mask = (t_hm == 1).float() 
            num_objects = mask.sum() + 1e-4
            
            # Applichiamo la maschera su entrambi i canali dell'offset (dx, dy)
            # Espandiamo la maschera per coprire i 2 canali dell'offset
            mask_off = mask.repeat(1, 2, 1, 1) 
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