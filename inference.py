import torch
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random

# Import dai tuoi file locali
from model import SimpleCenterNet
from utils import get_peaks, decode_predictions
from logo_dataset import LogoDataset 

# Percorsi
VAL_DIR = "datasetLOGOS/valid" # Meglio testare sulla cartella valid
VAL_ANN_FILE = os.path.join(VAL_DIR, "_annotations.coco.json")
checkpoint_path = "checkpoints/centernet_logo_simplified.pth"

def run_inference(model_path=checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carica Modello
    model = SimpleCenterNet().to(device)
    if not os.path.exists(model_path):
        print(f"ERRORE: Checkpoint non trovato in {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Modello caricato da {model_path}")

    # 2. Inizializzazione Dataset
    dataset = LogoDataset(img_dir=VAL_DIR, ann_file=VAL_ANN_FILE)
    
    # 3. Selezione casuale di un'immagine dal dataset
    idx = random.randint(0, len(dataset) - 1)
    image_tensor, target = dataset[idx]
    
    # 4. Esecuzione Modello
    image_input = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        # Il modello ora restituisce solo Heatmap e Offset
        p_hm, p_off = model(image_input)

    # 5. Decodifica Predizioni (Usa la funzione in utils.py)
    # Restituisce una lista di dict con 'center' e 'score'
    detections = decode_predictions(p_hm, p_off, threshold=0.3, stride=4)
    
    print(f"DEBUG - Massimo heatmap: {p_hm.max().item():.4f}")
    print(f"Loghi trovati: {len(detections)}")

    # 6. Visualizzazione
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # --- DENORMALIZZAZIONE PER LA VISUALIZZAZIONE ---
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img_viz = image_tensor.cpu() * std + mean
    img_viz = torch.clamp(img_viz, 0, 1)
    img_show = img_viz.permute(1, 2, 0).numpy()

    # Subplot 1: Immagine + Centri Predetti
    ax[0].imshow(img_show)
    ax[0].set_title(f"Rilevamento Loghi ({len(detections)} trovati)")
    
    for det in detections:
        cx, cy = det['center']
        score = det['score']
        # Disegna una croce rossa sul centro predetto
        ax[0].plot(cx, cy, 'r+', markersize=12, markeredgewidth=2)
        ax[0].text(cx, cy - 5, f"{score:.2f}", color='red', fontsize=10, fontweight='bold')

    # Subplot 2: Heatmap
    heatmap_show = p_hm.squeeze().cpu().numpy()
    im = ax[1].imshow(heatmap_show, cmap='magma', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    ax[1].set_title("Heatmap Predetta (32x32)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_inference()