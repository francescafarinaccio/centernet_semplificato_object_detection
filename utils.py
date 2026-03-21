# utilizzo questo file per le formule matematiche e funzioni di utilità che voglio tenere separate dal resto del codice


import torch
import torch.nn.functional as F
import numpy as np

#funzione di estrazione dei picchi locali da una heatmap usando MaxPool2d come NMS
def get_peaks(heatmap, threshold=0.3):
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
#funzione per disegnare una gaussiana su una heatmap, usata per creare le heatmap di addestramento
def draw_gaussian(heatmap, center, sigma):
    gh, gw = heatmap.shape
    x, y = center
    y_grid, x_grid = np.ogrid[0:gh, 0:gw]
    d2 = (x_grid - x)**2 + (y_grid - y)**2
    g = np.exp(-d2 / (2 * sigma**2))
    return np.maximum(heatmap, g)

# Questa funzione prende le predizioni di heatmap, offset e size e restituisce le coordinate finali del bounding box
def get_final_box(pred_hm, pred_off, pred_sz, stride=4, img_size=128):
    # 1. Troviamo il pixel più "caldo" (il centro nella mappa 32x32)
    idx = torch.argmax(pred_hm)
    iy = idx // 32
    ix = idx % 32

    # 2. Recuperiamo l'offset e la dimensione previsti in quel punto specifico
    # (Usiamo .item() per trasformare il tensore in un numero Python)
    off_x = pred_off[0, iy, ix].item()
    off_y = pred_off[1, iy, ix].item()
    
    w_rel = pred_sz[0, iy, ix].item()
    h_rel = pred_sz[1, iy, ix].item()

    # 3. Calcoliamo il centro reale nell'immagine 128x128
    cx = (ix + off_x) * stride
    cy = (iy + off_y) * stride

    # 4. Calcoliamo larghezza e altezza reali
    W = w_rel * img_size
    H = h_rel * img_size

    # 5. Coordinate per il disegno (Top-Left e Bottom-Right)
    x1, y1 = int(cx - W/2), int(cy - H/2)
    x2, y2 = int(cx + W/2), int(cy + H/2)

    return (x1, y1), (x2, y2)