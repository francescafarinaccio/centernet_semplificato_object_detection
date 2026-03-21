import torch
from torch.utils.data import Dataset
from utils import draw_gaussian
import numpy as np
import cv2

# Questo file definisce il dataset personalizzato per l'addestramento del modello CenterNet.

class CenterNetDataset(Dataset):
    def __init__(self, num_samples=100, img_size=128, output_size=32, sigma=2):
        self.num_samples = num_samples
        self.img_size = img_size
        self.output_size = output_size
        self.sigma = sigma
        self.stride = img_size // output_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # non abbiamo ancora un dataset reale quindi creiamo un'immagine vuota (tutto nero) e disegniamo un cerchio bianco (valore 1) in una posizione casuale, che rappresenta il centro dell'oggetto.
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        # Generiamo un centro casuale per l'oggetto, evitando i bordi (10 pixel di margine)
        cx = np.random.randint(10, self.img_size - 10)
        cy = np.random.randint(10, self.img_size - 10)

        #cv2.circle disegna un cerchio sull'immagine. in questo caso, disegniamo un cerchio bianco (valore 1) sul centro dell'oggetto. così il modello impara a riconoscere quel punto come centro
        cv2.circle(img, (cx, cy), 5, 1, -1)

        # 1. Prepariamo la matrice degli offset (2 canali: Dx e Dy)
        offset_map = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)

        #calcolo il centro in coordinate della heatmap (output_size x output_size) dividendo per lo stride
        heatmap = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        ctx_hm = cx / self.stride
        cty_hm = cy / self.stride

# 2. Calcoliamo gli indici interi per la posizione nella griglia
        ix, iy = int(ctx_hm), int(cty_hm)

# 1. Prepariamo la matrice degli offset
        offset_map = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        if 0 <= ix < self.output_size and 0 <= iy < self.output_size:
            offset_map[0, iy, ix] = ctx_hm - ix
            offset_map[1, iy, ix] = cty_hm - iy

        # 2. Prepariamo la mappa delle dimensioni
        size_map = np.zeros((2, self.output_size, self.output_size), dtype=np.float32)
        if 0 <= ix < self.output_size and 0 <= iy < self.output_size:
          size_map[0, iy, ix] = 10 / self.img_size
          size_map[1, iy, ix] = 10 / self.img_size


        heatmap = draw_gaussian(heatmap, (ctx_hm, cty_hm), self.sigma)

        img_tensor = torch.tensor(img).unsqueeze(0).float() # [1, 128, 128]
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).float() # [1, 32, 32]
        offset_map_tensor = torch.tensor(offset_map).unsqueeze(0).float() # [2, 32, 32]
        size_map_tensor = torch.tensor(size_map).unsqueeze(0).float() # [2, 32, 32]

        return img_tensor, heatmap_tensor, offset_map_tensor, size_map_tensor