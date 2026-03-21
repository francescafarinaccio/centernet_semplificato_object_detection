import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import CenterNetDataset
from model import SimpleCenterNet
import model
from utils import get_peaks 

#in inference, carico il modello addestrato, prendo un'immagine di test, faccio la predizione e visualizzo i risultati.



def run_inference(model_path="checkpoints/centernet_v1.pth"):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Carica Modello e Pesi
    model = SimpleCenterNet().to(device)
    model.load_state_dict(torch.load(model_path))
    #in modalità valutazione, il modello non aggiorna i pesi e non applica dropout o batchnorm
    model.eval() 
    print("Modello caricato correttamente.")

    # 3. Prendi un'immagine di test
    test_dataset = CenterNetDataset(num_samples=1)
    image, true_heatmap, true_offset, true_size = test_dataset[0]
    image_input = image.unsqueeze(0).to(device) # Aggiungi dimensione batch [1, 1, 128, 128]

    # 4. Inferenza (Predizione)
    with torch.no_grad(): # Disabilita gradienti per risparmiare memoria
        p_hm, p_off, p_sz = model(image_input)

    # Sposta heatmap su CPU per post-processing
    p_hm = p_hm.cpu()

    # 5. Estrai i centri (con stride 4)
    # get_peaks restituisce [N, 4] -> [Batch, Channel, Y, X]
    indices = get_peaks(p_hm, threshold=0.3)

    # Visualizzazione
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image[0].numpy(), cmap='gray') # Mostra immagine originale (128x128)
    ax.set_title("Predizione vs Ground Truth")

    # Disegna centri trovati
    stride = 4 # Fattore di scala 
    if indices.shape[0] > 0:
        for idx in indices:
            y, x = idx[2], idx[3] # Coordinate nella heatmap 32x32

            # 1. Estrai offset e dimensioni (usando indici y, x)
            off_x = p_off[0, 0, y, x].item()
            off_y = p_off[0, 1, y, x].item()
            w_norm = p_sz[0, 0, y, x].item()
            h_norm = p_sz[0, 1, y, x].item()
            # Prendi coordinate Y (indice 2) e X (indice 3)
            
            # 2. Calcola il centro preciso (con offset) e riportalo a 128px
            center_x = x.item() * stride
            center_y = y.item() * stride
            # 3. Denormalizza le dimensioni (W e H)
            w_pixel = w_norm * 128
            h_pixel = h_norm * 128

            # 4. Trova l'angolo in alto a sinistra per il rettangolo
            box_x = center_x - (w_pixel / 2)
            box_y = center_y - (h_pixel / 2)

            # Disegna punto rosso nel cerchio per indicare il centro predetto
            circle = patches.Circle((center_x, center_y), radius=3, color='red', fill=True, label='Predetto')
            #disegnno la BB
            rect = patches.Rectangle((box_x, box_y), w_pixel, h_pixel, linewidth=1, edgecolor='red', facecolor='none')
            
            ax.add_patch(rect)
            ax.add_patch(circle)
            print(f"Oggetto trovato a coordinate: ({center_x:.1f}, {center_y:.1f})")


    else:
        print("Nessun oggetto trovato sopra la soglia.")

    plt.legend(["Predetto (Punto Rosso)"])
    plt.show()

if __name__ == "__main__":
    run_inference()
