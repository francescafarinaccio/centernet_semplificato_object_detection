import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from utils import draw_gaussian, get_gaussian_radius

class LogoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file) #leggo il json con le annotazioni
        self.img_dir = img_dir
        self.transform = transform
        
        # Filtriamo solo le immagini che hanno annotazioni (loghi) per evitare campioni inutili 
        all_img_ids = self.coco.getImgIds()
        self.img_ids = []
        for img_id in all_img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)

    def __getitem__(self, idx): #Restituisce un'immagine e il suo target (heatmap + offset)
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids) # Carico tutte le annotazioni per questa immagine
        img_info = self.coco.loadImgs(img_id)[0]

        # 1. Caricamento e Resize Immagine
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        w_orig, h_orig = image.size
        
        # Target fissato a 128x128 
        input_h, input_w = 128, 128
        image_resized = image.resize((input_w, input_h))
        
        # Trasformazione opzionale (normalizzazione, data augmentation, ecc.)
        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            # Normalizzazione base se non passi trasformazioni esterne
            img_np = np.array(image_resized).astype(np.float32) / 255.0 # Normalizzazione [0, 1]
            image_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # Converti in CxHxW
            # Normalizzazione ImageNet standard
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # Perché è un tensore di shape [3, 1, 1] che si adatta a CxHxW
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) 
            image_tensor = (image_tensor - mean) / std # Normalizzazione

        
        # 2. Preparazione Output (Heatmap e Offset)
        stride = 4
        out_h, out_w = input_h // stride, input_w // stride # Output a 1/4 della risoluzione originale (32x32)
        
        # Inizializziamo heatmap e offset a zero
        hm = np.zeros((1, out_h, out_w), dtype=np.float32) # 1 canale per la heatmap
        reg = np.zeros((2, out_h, out_w), dtype=np.float32) # 2 canali per l'offset (dx, dy)

        scale_x, scale_y = input_w / w_orig, input_h / h_orig # Fattori di scala per adattare le coordinate originali alla risoluzione 128x128

        # 3. Ciclo su tutti i loghi presenti nell'immagine
        for ann in anns:
            x, y, w, h = ann['bbox']
            
            # Coordinate riscalate a 128x128
            x128, y128 = x * scale_x, y * scale_y
            w128, h128 = w * scale_x, h * scale_y
            
            # Centro su scala 32x32 (floating point per l'offset)
            ct = np.array([
                (x128 + w128 / 2) / stride, 
                (y128 + h128 / 2) / stride
            ], dtype=np.float32)
            
            ct_int = ct.astype(np.int32)

            # Verifica che il centro discretizzato sia all'interno dei confini dell'output
            if 0 <= ct_int[0] < out_w and 0 <= ct_int[1] < out_h:
                # Disegno Gaussiana (usiamo un raggio minimo di 2 per visibilità)
                radius = get_gaussian_radius((h128 / stride, w128 / stride))
                radius = max(2, int(radius))
                draw_gaussian(hm[0], ct_int, radius)
                
                # Calcolo Offset: la differenza tra il centro reale e quello discretizzato
                # reg[0] = dx, reg[1] = dy
                reg[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                reg[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]

        return image_tensor, { #trasformo gli array numpy in tensori torch
            'hm': torch.from_numpy(hm),
            'reg': torch.from_numpy(reg)
        }

    # La lunghezza del dataset è il numero di immagini con annotazioni
    def __len__(self):
        return len(self.img_ids)