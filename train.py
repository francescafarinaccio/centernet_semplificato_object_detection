#per l'addestramento del modello, definisco qui il ciclo di training, la loss e l'ottimizzatore. 

from pyexpat import model

import torch
import torch.nn as nn
import os #per gestire le cartelle e i file
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CenterNetDataset
from model import SimpleCenterNet

save_dir='checkpoints' #cartella dove salvare i pesi del modello
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Cartella '{save_dir}' creata con successo")

# Definizione della loss per CenterNet, che combina la heatmap loss (MSE) con le regression loss (L1) per offset e size, bilanciando i pesi tra di loro.
def criterion_centernet(preds, targets):
    # Predizioni del modello: [batch, canali, 32, 32]
    p_hm, p_off, p_sz = preds
    # Target del dataset: [batch, canali, 32, 32]
    t_hm, t_off, t_sz = targets

    # 1. Heatmap Loss (MSE) - Focalizza la posizione generale
    loss_hm = nn.MSELoss()(p_hm, t_hm)

    # 2. Offset Loss (L1 + Maschera) - Precisione sub-pixel
    # Moltiplichiamo per t_hm per ignorare lo sfondo
    loss_off = nn.L1Loss()(p_off * t_hm, t_off * t_hm)

    # 3. Size Loss (L1 + Maschera) - Dimensioni (W, H)
    loss_sz = nn.L1Loss()(p_sz * t_hm, t_sz * t_hm)

    # Somma finale con pesi bilanciati
    # Spesso si usa 1.0 per hm e off, e 0.1 per sz
    total_loss = loss_hm + 1.0 * loss_off + 0.1 * loss_sz
    
    return total_loss

# La funzione di training esegue il ciclo di addestramento per un certo numero di epoche, iterando sui dati, calcolando la loss e aggiornando i pesi del modello. 
# Alla fine salva i pesi in un file .pth.
def train():
    
    # --- IPERPARAMETRI ---
    batch_size = 16
    learing_rate = 1e-3  # ovvero 0.001
    epochs = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = os.path.join(save_dir, f"centernet_v1.pth")
     
    
    print(f"Addestramento su: {device}")

    # 2. Dati
    dataset = CenterNetDataset(num_samples=2000) # Un po' di campioni per imparare
    train_loader = DataLoader(dataset, batch_size, shuffle=True)

    # 3. Modello, Loss e Optimizer
    model = SimpleCenterNet().to(device)
    criterion_hm = nn.MSELoss()
    criterion_reg = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learing_rate) 

    # 4. Ciclo di Addestramento (Training Loop)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 1. Estraiamo i dati dal dataloader
        for inputs, t_hm, t_off, t_sz in train_loader:
    
    # Spostiamo i dati sulla GPU se disponibile
            inputs = inputs.to(device)
            t_hm = t_hm.to(device)
            t_off = t_off.to(device)
            t_sz = t_sz.to(device)

    # 2. Forward pass: il modello restituisce una tupla
            p_hm, p_off, p_sz = model(inputs)

    # 3. Calcolo delle singole loss
    #alla heatmpa non applico nessuna maschera perché deve imparare a ricostruire l'intera mappa, compresi gli zeri (sfondo) e la campana gaussiana (centro)
            loss_heatmap = criterion_hm(p_hm, t_hm)
    
    # Creiamo una msschera binaria 1 dove c'è l'oggetto, 0 altrove
    #per l'offset e la size non voglio utilizzare ma schera che dà importanza solo al centro dell'immagine ma deve guardare a tutta l'immagine
            mask = (t_hm > 0).float()
    # Usiamo la heatmap reale come maschera per focalizzarci solo sui centri
            loss_offset = criterion_reg(p_off * mask, t_off * mask)
            loss_size = criterion_reg(p_sz * mask, t_sz * mask)

    # 4. Loss Totale (bilanciata con i pesi lambda)
            total_loss = loss_heatmap + (1.0 * loss_offset) + (0.1 * loss_size)

    # 5. Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            # Accumula la loss per il reporting
            running_loss += total_loss.item()
            optimizer.step()

        print(f"Epoca [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.6f}") # Stampa la loss media per epoca

    
  # Salva i pesi del modello alla fine
    torch.save(model.state_dict(), save_path)
    print("Addestramento completato e modello salvato in :", save_path)

if __name__ == "__main__":
    train()
