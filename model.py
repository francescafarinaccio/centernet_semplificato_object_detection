import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCenterNet(nn.Module):
    def __init__(self):
        super(SimpleCenterNet, self).__init__()

        # ENCODER (Downsampling 1/4)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2), # -> 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), # -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # DECODER (Ritorno a 1/4 o risoluzione piena se necessario)
        self.neck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # HEADS
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        
        # L'offset head restituisce 2 canali: dx e dy
        self.offset_head = nn.Conv2d(64, 2, kernel_size=1)

    #funzione forward che restituisce sia heatmap che offset
    def forward(self, x):
        features = self.enc(x)
        features = self.neck(features)
        
        hm = self.heatmap_head(features)
        off = self.offset_head(features)
        return hm, off

    # Funzione di inferenza per ottenere le coordinate dei picchi
    @torch.no_grad() # Non calcoliamo i gradienti durante l'inferenza
    def predict(self, x, threshold=0.3, stride=4):
        self.eval()
        # Ottieni SOLO la heatmap per l'inferenza dei picchi
        hm, off = self.forward(x)
        
        # NMS via MaxPool
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
        keep = (hmax == hm).float()
        peaks = hm * keep
        
        # Estrazione coordinate
        indices = torch.nonzero(peaks > threshold)
        
        if indices.shape[0] > 0:
            # indices è [N, 4] -> (batch, channel, y, x)
            # Prendiamo y e x
            batch_id = indices[:, 0]
            y = indices[:, 2].float()
            x_coord = indices[:, 3].float()
            
            # Aggiungiamo l'offset se disponibile (opzionale ma consigliato)
            # off ha shape [Batch, 2, H, W] -> canale 0 è dy, canale 1 è dx
            dy = off[batch_id, 0, y.long(), x_coord.long()]
            dx = off[batch_id, 1, y.long(), x_coord.long()]
            
            # Applichiamo stride e offset
            res_x = (x_coord + dx) * stride
            res_y = (y + dy) * stride
            
            # Restituiamo un tensore di shape [N, 2] con le coordinate (x, y) dei picchi
            return torch.stack([res_x, res_y], dim=1)
        # Se non ci sono picchi sopra la soglia, restituiamo un tensore vuoto
        return torch.tensor([])