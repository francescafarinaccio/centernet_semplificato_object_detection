import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_peaks

class SimpleCenterNet(nn.Module):
    def __init__(self):
        super(SimpleCenterNet, self).__init__()

        # ENCODER (Downsampling 1/4)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # DECODER 
        self.neck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # HEADS
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid() #schiaccia i valori tra 0 e 1
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

        # Richiamiamo la funzione di utility per ottenere indici e punteggi
        indices, scores = get_peaks(hm, threshold=threshold)

        if indices.shape[0] > 0:
            # Estraiamo le informazioni dagli indici restituiti [B, C, Y, X]
            batch_id = indices[:, 0]
            y_coord  = indices[:, 2].float()
            x_coord  = indices[:, 3].float()
    
            # Recupero degli offset (usando gli indici interi per accedere ai tensori)
            # Ricorda: off ha shape [Batch, 2, H, W] -> canale 0 è dy, canale 1 è dx
            dy = off[batch_id, 0, y_coord.long(), x_coord.long()]
            dx = off[batch_id, 1, y_coord.long(), x_coord.long()]
    
            # Applicazione dello stride e correzione con l'offset
            res_x = (x_coord + dx) * stride
            res_y = (y_coord + dy) * stride
    
            # Restituiamo le coordinate finali (N, 2)
            return torch.stack([res_x, res_y], dim=1)

        # Se non ci sono picchi sopra la soglia
        return torch.tensor([])