# utilizzo questo file per le formule matematiche e funzioni di utilità che voglio tenere separate dal resto del codice


import torch
import torch.nn.functional as F

#funzione di estrazione dei picchi locali da una heatmap usando MaxPool2d come NMS
def get_peaks(heatmap, threshold=0.3):
    """
    Estrae i picchi locali da una heatmap usando MaxPool2d come NMS.
    """
    # 1. Trova i massimi locali
    # Un kernel 3x3 confronta ogni pixel con i suoi vicini
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    
    # 2. Mantieni solo i pixel che sono rimasti invariati (erano già i massimi)
    keep = (hmax == heatmap).float()
    peaks = heatmap * keep
    
    # 3. Filtra per soglia di confidenza
    # Restituisce gli indici dove il valore è > threshold
    # Il formato è [N, 4] -> [Batch, Canale, Y, X]
    indices = torch.nonzero(peaks > threshold)
    
    return indices