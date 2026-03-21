import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import SimpleCenterNet
from dataset import CenterNetDataset
import numpy as np
from utils import get_peaks

# Questo file esegue un test di visualizzazione per verificare che il modello addestrato riesca a trovare i centri degli oggetti in un'immagine di test. 
# Carica il modello, prende un'immagine dal dataset, fa la predizione e visualizza i risultati.
#utile in fase di debug 

def visual_test():
    # 1. Setup e caricamento modello
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCenterNet().to(device)
    model.load_state_dict(torch.load("centernet_model.pth"))
    model.eval()
    
    # 2. Prepariamo un'immagine dal dataset
    dataset = CenterNetDataset(num_samples=1)
    img, target_heatmap = dataset[0]
    
    # Prepariamo l'input per il modello (aggiungiamo il batch con unsqueeze)
    #unsqueeze(0) aggiunge una dimensione all'inizio, trasformando [1, 128, 128] in [1, 1, 128, 128]
    input_tensor = img.unsqueeze(0).to(device)
    
    # 3. Inferenza
    #no_grad() disabilita il calcolo dei gradienti, risparmiando memoria e velocizzando l'inferenza
    with torch.no_grad():
        output_heatmap = model(input_tensor)
    
    # 4. Estrazione picchi (Threshold 0.3)
    # get_peaks restituisce [batch_idx, class_idx, y, x]
    #tutto ciò che è sopra la soglia di 0.3 viene considerato un picco (centro predetto)
    peaks = get_peaks(output_heatmap, threshold=0.3)
    
    # 5. Visualizzazione
    plt.figure(figsize=(10, 5))
    
    # Mostriamo l'immagine originale
    ax = plt.subplot(1, 2, 1)
    ax.imshow(img.squeeze(), cmap='gray') # squeeze per togliere il canale (1, 128, 128) -> (128, 128)
    ax.set_title("Immagine con Centro Predetto")
    
    stride = 4
    if len(peaks) > 0:
        for p in peaks:
            y_heat, x_heat = p[2], p[3]
            # Moltiplichiamo per lo stride per tornare a 128x128
            #item estrae il valore scalare da un tensore 0-dimensionale, utile per convertire coordinate da tensori a numeri
            x_orig = x_heat.item() * stride
            y_orig = y_heat.item() * stride
            
            # Disegniamo un cerchio rosso sul centro
            circle = patches.Circle((x_orig, y_orig), radius=3, color='red', fill=True)
            ax.add_patch(circle)
            print(f"Centro trovato a: x={x_orig}, y={y_orig}")
    
    # Mostriamo la heatmap predetta per vedere cosa "vede" la rete
    plt.subplot(1, 2, 2)
    plt.imshow(output_heatmap[0, 0].cpu(), cmap='viridis')
    plt.title("Heatmap Predetta (32x32)")
    
    plt.show()

if __name__ == "__main__":
    visual_test()