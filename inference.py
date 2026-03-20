import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import CenterNetDataset
from model import SimpleCenterNet
from utils import get_peaks # Assicurati di avere utils.py pronto!

def run_inference(model_path="centernet_model.pth"):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Carica Modello e Pesi
    model = SimpleCenterNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval() # Modalità valutazione (no dropout, no batchnorm update)
    print("Modello caricato correttamente.")

    # 3. Prendi un'immagine di test
    # Ne creiamo una sola per semplicità
    test_dataset = CenterNetDataset(num_samples=1)
    image, true_heatmap = test_dataset[0]
    image_input = image.unsqueeze(0).to(device) # Aggiungi dimensione batch [1, 1, 128, 128]

    # 4. Inferenza (Predizione)
    with torch.no_grad(): # Disabilita gradienti per risparmiare memoria
        pred_heatmap = model(image_input)

    # Sposta heatmap su CPU per post-processing
    pred_heatmap = pred_heatmap.cpu()

    # 5. Estrai i centri (con stride 4)
    # get_peaks restituisce [N, 4] -> [Batch, Channel, Y, X]
    indices = get_peaks(pred_heatmap, threshold=0.3)

    # Visualizzazione
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image[0].numpy(), cmap='gray') # Mostra immagine originale (128x128)
    ax.set_title("Predizione vs Ground Truth")

    # Disegna centri trovati
    stride = 4 # Fattore di scala 128/32
    if indices.shape[0] > 0:
        for idx in indices:
            # Prendi coordinate Y (indice 2) e X (indice 3)
            y, x = idx[2], idx[3]
            # Moltiplica per lo stride per tornare a 128x128
            orig_x = x.item() * stride
            orig_y = y.item() * stride

            # Disegna punto rosso
            circle = patches.Circle((orig_x, orig_y), radius=3, color='red', fill=True, label='Predetto')
            ax.add_patch(circle)
            print(f"Oggetto trovato a coordinate: ({orig_x:.1f}, {orig_y:.1f})")
    else:
        print("Nessun oggetto trovato sopra la soglia.")

    plt.legend(["Predetto (Punto Rosso)"])
    plt.show()

if __name__ == "__main__":
    run_inference()
