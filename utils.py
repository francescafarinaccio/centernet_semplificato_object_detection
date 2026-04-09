
import torch
import torch.nn.functional as F
import numpy as np


#funzione per disegnare una gaussiana sulla heatmap (usata in logo_dataset.py)
def draw_gaussian(heatmap, center, radius, k=1):

    diameter = 2 * radius + 1
    sigma = diameter / 6
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    # Definiamo i confini della sottomatrice per non uscire dai bordi
    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)

    masked_heatmap  = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
    
    # Generiamo la gaussiana locale
    y_grid, x_grid = np.ogrid[-top:bottom + 1, -left:right + 1]
    g = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    
    # Applichiamo il massimo tra il valore esistente e la nuova gaussiana
    np.maximum(masked_heatmap, g * k, out=masked_heatmap)
    return heatmap

# Funzione per estrarre i picchi dalla heatmap (usata in inference.py)
def get_peaks(heatmap, threshold=0.4):
    # Applichiamo NMS tramite MaxPool per trovare i picchi locali
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    peaks = heatmap * keep
    
    # Estrazione indici e score
    # indices: [N, 4] -> (B, C, Y, X)
    indices = torch.nonzero(peaks > threshold)
    scores = peaks[peaks > threshold]
    
    return indices, scores

#
def get_gaussian_radius(box_size):
    h, w = box_size
    # Il raggio minimo garantisce che la "macchia" sia visibile anche per loghi piccoli
    return max(2, int(min(w, h) * 0.3))

# Funzione per decodificare le predizioni del modello in coordinate reali (usata in inference.py)
def decode_predictions(pred_hm, pred_off, threshold=0.4, stride=4):

    indices, scores = get_peaks(pred_hm, threshold)
    #processo ogni logo ed estraggo le coordinate finali scalate con l'offset
    results = []
    for i in range(len(indices)):
        batch, ch, iy, ix = indices[i]
        score = scores[i].item()
        
        # Recupero offset (canale 0 = dx, canale 1 = dy)
        off_x = pred_off[batch, 0, iy, ix].item()
        off_y = pred_off[batch, 1, iy, ix].item()
        
        # Coordinate finali scalate
        cx = (ix.item() + off_x) * stride
        cy = (iy.item() + off_y) * stride
        
        results.append({
            'center': (cx, cy),
            'score': score
        })
        
    return results


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