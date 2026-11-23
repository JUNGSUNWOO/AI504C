"""AI504c Midterm Project — Problem 3: Variational Autoencoder.

Run in the same directory as data_preprocess.py:
    python problem3_revised.py

Expected exports from data_preprocess.py:
    x_train_img: (N, 28, 28), pixel values in [0, 1]
    y_train: (N,) digit labels
    class_indices: dict with keys 0, 1, 9 and 100+ training indices each
"""
from pathlib import Path
import csv
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_preprocess import x_train_img, y_train, class_indices

SEED = 42
LATENT_DIM = 2
HIDDEN_DIM = 128
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 20
RESULTS_DIR = Path(__file__).resolve().parent / "result" / "problem3"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder_hidden = nn.Linear(784, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, 784)

    def encode(self, x):
        hidden = torch.relu(self.encoder_hidden(x))
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    @staticmethod
    def reparameterize(mu, logvar):
        # z = mu + sigma * epsilon, epsilon ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        hidden = torch.relu(self.decoder_hidden(z))
        return torch.sigmoid(self.decoder_output(hidden))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


def vae_loss(x, reconstruction, mu, logvar):
    """Squared L2 reconstruction + KL(N(mu,diag(sigma^2)) || N(0,I)).

    Reduction: sum over 784 pixels / latent dimensions per image,
    then mean across images in the minibatch. KL weight = 1.
    """
    reconstruction_per_image = (x - reconstruction).pow(2).sum(dim=1)
    kl_per_image = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    reconstruction_loss = reconstruction_per_image.mean()
    kl_loss = kl_per_image.mean()
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss


def train_vae(model, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_sum = reconstruction_sum = kl_sum = 0.0
        num_images = 0
        for (images,) in loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(images)
            total, reconstruction, kl = vae_loss(images, reconstructed, mu, logvar)
            total.backward()
            optimizer.step()

            count = images.size(0)
            num_images += count
            total_sum += total.item() * count
            reconstruction_sum += reconstruction.item() * count
            kl_sum += kl.item() * count

        metrics = {
            'epoch': epoch,
            'total_loss': total_sum / num_images,
            'reconstruction_loss': reconstruction_sum / num_images,
            'kl_loss': kl_sum / num_images,
        }
        history.append(metrics)
        print(f"Epoch {epoch:02d}/{EPOCHS} | total={metrics['total_loss']:.4f} "
              f"| recon={metrics['reconstruction_loss']:.4f} "
              f"| KL={metrics['kl_loss']:.4f}")
    return history


def plot_loss(history):
    epochs = [row['epoch'] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row['total_loss'] for row in history], marker='o', markersize=3)
    ax.set(xlabel='Epoch', ylabel='Mean loss per image', title='Problem 3.1: VAE Total Loss')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'problem3_1_vae_total_loss.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label in [('total_loss', 'Total'),
                       ('reconstruction_loss', 'Reconstruction (sum over pixels)'),
                       ('kl_loss', 'KL (sum over latent dimensions)')]:
        ax.plot(epochs, [row[key] for row in history], label=label)
    ax.set(xlabel='Epoch', ylabel='Mean loss per image', title='VAE Loss Components')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'problem3_1_vae_loss_components.png', dpi=180)
    plt.close(fig)


def plot_latent_means(model, x_train, device):
    model.eval()
    fig, ax = plt.subplots(figsize=(7, 6))
    with torch.no_grad():
        for digit in (0, 1, 9):
            indices = np.asarray(class_indices[digit], dtype=int)[:100]
            if len(indices) != 100:
                raise ValueError(f'class_indices[{digit}] must contain at least 100 indices')
            if not np.all(np.asarray(y_train)[indices] == digit):
                raise ValueError(f'class_indices[{digit}] includes other digit labels')
            samples = x_train[indices].to(device)
            mu, _ = model.encode(samples)
            coords = mu.cpu().numpy()
            ax.scatter(coords[:, 0], coords[:, 1], s=17, alpha=0.7, label=str(digit))
    ax.set(xlabel='Latent mean μ₁', ylabel='Latent mean μ₂',
           title='Problem 3.2: VAE Latent Means (100 per digit)')
    ax.legend(title='MNIST digit')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'problem3_2_vae_latent_mean.png', dpi=180)
    plt.close(fig)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Use original [0,1] MNIST pixels, NOT standardized PCA inputs.
    x_train = torch.as_tensor(np.asarray(x_train_img), dtype=torch.float32).reshape(-1, 784)
    if x_train.shape[1] != 784:
        raise ValueError('Expected 28x28 MNIST images')
    if x_train.min() < 0 or x_train.max() > 1:
        raise ValueError('Expected pixel values in [0, 1]')
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(TensorDataset(x_train), batch_size=BATCH_SIZE,
                        shuffle=True, generator=generator)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}, {len(x_train)} images')
    model = VAE().to(device)
    history = train_vae(model, loader, device)

    with (RESULTS_DIR / 'problem3_vae_training_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'total_loss', 'reconstruction_loss', 'kl_loss'])
        writer.writeheader()
        writer.writerows(history)
    plot_loss(history)
    plot_latent_means(model, x_train, device)
    torch.save({'model_state_dict': model.state_dict(), 'latent_dim': LATENT_DIM,
                'hidden_dim': HIDDEN_DIM, 'seed': SEED}, CHECKPOINT_DIR / 'vae_M2.pth')
    print('Saved plots and CSV to results/, model checkpoint to checkpoints/')


if __name__ == '__main__':
    main()
