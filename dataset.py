import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

def draw_gaussian(heatmap, center, sigma):
    gh, gw = heatmap.shape
    x, y = center
    y_grid, x_grid = np.ogrid[0:gh, 0:gw]
    d2 = (x_grid - x)**2 + (y_grid - y)**2
    g = np.exp(-d2 / (2 * sigma**2))
    return np.maximum(heatmap, g)

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
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        cx = np.random.randint(10, self.img_size - 10)
        cy = np.random.randint(10, self.img_size - 10)

        cv2.circle(img, (cx, cy), 5, 1, -1)

        heatmap = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        ctx_hm = cx / self.stride
        cty_hm = cy / self.stride

        heatmap = draw_gaussian(heatmap, (ctx_hm, cty_hm), self.sigma)

        img_tensor = torch.tensor(img).unsqueeze(0).float() # [1, 128, 128]
        heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).float() # [1, 32, 32]

        return img_tensor, heatmap_tensor
