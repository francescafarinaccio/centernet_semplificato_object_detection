import torch
import torch.nn as nn

class SimpleCenterNet(nn.Module):
    def __init__(self):
        super(SimpleCenterNet, self).__init__()

        # Encoder - Blocco 1: 128x128 -> 64x64
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Encoder - Blocco 2: 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder / Neck: Mantiene 32x32 ma elabora le feature
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Head di Output: 32x32x64 -> 32x32x1 (Heatmap)
        # Usiamo la Sigmoide perché vogliamo valori tra 0 e 1
        self.head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.decoder(x)
        return self.head(x)

def predict(self, x, threshold=0.3, stride=4):
        """
        Esegue l'inferenza e restituisce le coordinate (x, y) riscalate.
        """
        self.eval() # Imposta il modello in modalità valutazione
        with torch.no_grad():
            heatmap = self.forward(x)
            
            # 1. Trova i massimi locali (NMS via MaxPool)
            hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
            keep = (hmax == heatmap).float()
            peaks = heatmap * keep
            
            # 2. Estrai coordinate sopra soglia
            # nonzero restituisce [N, 4] -> [Batch, Channel, Y, X]
            indices = torch.nonzero(peaks > threshold)
            
            # 3. Riscalamento coordinate (moltiplichiamo per lo stride)
            # Prendiamo solo Y (indice 2) e X (indice 3) e moltiplichiamo
            if indices.shape[0] > 0:
                coords = indices[:, 2:].float() * stride
                # Invertiamo per avere (x, y) invece di (y, x)
                coords = coords[:, [1, 0]] 
                return coords
            return torch.tensor([])