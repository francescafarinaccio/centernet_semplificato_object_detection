from dataset import CenterNetDataset
from torch.utils.data import DataLoader

def main():
    # Inizializziamo il dataset e il loader
    dataset = CenterNetDataset(num_samples=100)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Prendiamo un batch per testare
    images, heatmaps = next(iter(train_loader))

    print(f"Batch immagini: {images.shape}")    # Dovrebbe essere [16, 1, 128, 128]
    print(f"Batch heatmaps: {heatmaps.shape}")  # Dovrebbe essere [16, 1, 32, 32]
    print("Setup dei dati completato con successo!")

if __name__ == "__main__":
    main()
