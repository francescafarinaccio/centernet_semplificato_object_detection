import torch
import torch.nn as nn
import torch.nn.functional as F

# Questo file definisce il modello semplice di CenterNet, con un encoder, un decoder e due head separate per la heatmap e l'offset. Include anche una funzione di predizione che estrae i picchi locali dalla heatmap e li riscalano alle coordinate originali dell'immagine.
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
       # Head per la Heatmap (1 canale)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Head per l'Offset (2 canali)
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1)
    # Niente Sigmoid qui!
        )

        # Head per la Size (2 canali) 
        self.size_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1)
        )

    # La funzione forward definisce il percorso dei dati attraverso la rete. Prende un input, lo passa attraverso gli encoder, il decoder e infine 
    # la head per ottenere la heatmap predetta.
    def forward(self, x): #passaggio attraverso i blocchi comuni a entrambe le head
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.decoder(x)
        #produco i due output separati
        hm = self.heatmap_head(x)
        off= self.offset_head(x)
        sz= self.size_head(x)
        return hm, off, sz

    # La funzione predict esegue l'inferenza e restituisce le coordinate (x, y) riscalate. Utilizza la stessa logica di estrazione dei picchi locali vista in utils.py, ma con un ulteriore passo di riscalamento per tornare alle coordinate originali dell'immagine.
def predict(self, x, threshold=0.3, stride=4):
        self.eval() # Imposta il modello in modalità valutazione
        with torch.no_grad():
            heatmap = self.forward(x)
            
            # 1. Trova i massimi locali (NMS via MaxPool)
            # Un kernel 3x3 confronta ogni pixel con i suoi vicini. il valore più alto è un potenziale centro
            hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
            #crea una maschera binaria (0 o 1) dove 1 indica i pixel che sono massimi locali (hmax == heatmap). così azzera gli altri pixel
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