#per l'addestramento del modello, definisco qui il ciclo di training, 
# la loss e l'ottimizzatore. Alla fine salvo i pesi del modello addestrato in un file .pth

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CenterNetDataset
from model import SimpleCenterNet

def train():
    # 1. Setup Device (Usa la GPU se disponibile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Addestramento su: {device}")

    # 2. Dati
    dataset = CenterNetDataset(num_samples=2000) # Un po' di campioni per imparare
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. Modello, Loss e Optimizer
    model = SimpleCenterNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Ciclo di Addestramento (Training Loop)
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, heatmaps in train_loader:
            images, heatmaps = images.to(device), heatmaps.to(device)

            # Reset dei gradienti
            optimizer.zero_grad()

            # Forward pass (Predizione)
            outputs = model(images)

            # Calcolo della Loss
            loss = criterion(outputs, heatmaps)

            # Backward pass (Calcolo gradienti e aggiornamento)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoca [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.6f}")

    # Salva i pesi del modello alla fine
    torch.save(model.state_dict(), "centernet_model.pth")
    print("Addestramento completato e modello salvato!")

if __name__ == "__main__":
    train()
