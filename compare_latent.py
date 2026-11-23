"""
AI504c: PCA / AE / VAE qualitative + quantitative comparison.

Run after:
    python problem2.py
    python problem3.py
    python compare_latent.py

Outputs: result/comparison/
  comparison_latent.png
  comparison_test_reconstruction.png
  comparison_reconstruction_examples.png
  comparison_metrics.csv
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_preprocess import (
    x_train_pca, x_test_pca, x_test, x_train_img,
    x_test_img, train_mean, train_scale, class_indices,
)
from problem3 import VAE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "result" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1024
LATENT_DIM = 2
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class AE(nn.Module):
    """Matches the M=2 architecture of problem2.py."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.decoder(self.encode(x))


def checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}\nRun problem2.py and problem3.py first."
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def load_models():
    ae_ckpt = checkpoint(ROOT / "checkpoints" / "autoencoder_M2.pth")
    vae_ckpt = checkpoint(ROOT / "checkpoints" / "vae_M2.pth")
    if int(ae_ckpt.get("latent_dim", 2)) != 2:
        raise ValueError("AE checkpoint is not M=2")
    if int(vae_ckpt.get("latent_dim", 2)) != 2:
        raise ValueError("VAE checkpoint is not M=2")
    ae = AE().to(DEVICE)
    ae.load_state_dict(ae_ckpt["model_state_dict"])
    vae = VAE(latent_dim=2, hidden_dim=128).to(DEVICE)
    vae.load_state_dict(vae_ckpt["model_state_dict"])
    return ae.eval(), vae.eval()


def fit_pca_basis():
    """Training-only standardized PCA, no built-in PCA function."""
    print("Fitting PCA basis from training data...")
    X = x_train_pca.to(torch.float64)
    cov = X.T @ X / X.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    top2 = torch.argsort(eigenvalues, descending=True)[:2]
    return eigenvectors[:, top2].to(torch.float32)


@torch.no_grad()
def pca_reconstruct(standardized, basis):
    # Restore pixel units; do not clip before calculating squared error.
    return (standardized @ basis) @ basis.T * train_scale + train_mean


@torch.no_grad()
def test_metrics(ae, vae, basis):
    """Mean over 10,000 test images of sum of 784 squared pixel errors."""
    total = {"PCA": 0.0, "AE": 0.0, "VAE_mu": 0.0}
    count = 0
    for start in range(0, len(x_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(x_test))
        originals = x_test[start:end]  # [0,1] pixels
        standardized = x_test_pca[start:end]
        pca_hat = pca_reconstruct(standardized, basis)
        originals_gpu = originals.to(DEVICE)
        ae_hat = ae(originals_gpu).cpu()
        mu, _ = vae.encode(originals_gpu)
        vae_hat = vae.decode(mu).cpu()  # deterministic mean-based evaluation

        for name, pred in (
            ("PCA", pca_hat), ("AE", ae_hat), ("VAE_mu", vae_hat)
        ):
            total[name] += (originals - pred).square().sum().item()
        count += end - start

    assert count == 10000, f"Expected 10,000 test images; got {count}"
    return {name: value / count for name, value in total.items()}


@torch.no_grad()
def latent_figure(ae, vae, basis):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {0: "tab:blue", 1: "tab:orange", 9: "tab:green"}
    for digit in (0, 1, 9):
        indices = np.asarray(class_indices[digit], dtype=int)
        pca_z = (x_train_pca[indices] @ basis).numpy()
        inputs = x_train_img[indices].reshape(-1, 784).to(DEVICE)
        ae_z = ae.encode(inputs).cpu().numpy()
        vae_mu, _ = vae.encode(inputs)
        vae_z = vae_mu.cpu().numpy()
        for ax, values in zip(axes, (pca_z, ae_z, vae_z)):
            ax.scatter(values[:, 0], values[:, 1],
                       label=str(digit), s=17, alpha=0.7,
                       color=colors[digit])

    for ax, title in zip(axes, (
        "PCA (M=2)", "AE (M=2)", "VAE latent mean (M=2)"
    )):
        ax.set_title(title)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(alpha=0.2)
        ax.legend(title="Digit")
    fig.suptitle("Identical 100 training samples per digit (0, 1, 9)")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_latent.png", dpi=200)
    plt.close(fig)


def metric_figure(metrics):
    names = ["PCA", "AE", "VAE_mu"]
    values = [metrics[name] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(["PCA", "AE", "VAE (mean z)"], values)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(), f"{value:.4f}",
                ha="center", va="bottom")
    ax.set_ylabel("Test mean squared L2 error per image")
    ax.set_title("MNIST Test Reconstruction: M=2 (10,000 images)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_test_reconstruction.png", dpi=200)
    plt.close(fig)


@torch.no_grad()
def examples_figure(ae, vae, basis):
    # Same first 5 test images for all three models.
    idx = torch.arange(5)
    original = x_test[idx]
    pca_hat = pca_reconstruct(x_test_pca[idx], basis)
    ae_hat = ae(original.to(DEVICE)).cpu()
    mu, _ = vae.encode(original.to(DEVICE))
    vae_hat = vae.decode(mu).cpu()
    rows = [original, pca_hat, ae_hat, vae_hat]
    labels = ["Original", "PCA", "AE", "VAE (mean z)"]

    fig, axes = plt.subplots(4, 5, figsize=(9, 7))
    for i, (images, label) in enumerate(zip(rows, labels)):
        for j in range(5):
            # Clip only for visualization, never for metrics.
            axes[i, j].imshow(images[j].reshape(28, 28).clamp(0, 1),
                              cmap="gray", vmin=0, vmax=1)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(label, rotation=0,
                                      labelpad=44, va="center")
    fig.suptitle("Same five MNIST test images; all models M=2")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_reconstruction_examples.png", dpi=200)
    plt.close(fig)


def main():
    print(f"Device: {DEVICE}")
    ae, vae = load_models()
    basis = fit_pca_basis()
    metrics = test_metrics(ae, vae, basis)
    latent_figure(ae, vae, basis)
    metric_figure(metrics)
    examples_figure(ae, vae, basis)

    with (OUT / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["model", "latent_dim", "test_images",
                         "mean_squared_L2_per_image", "evaluation"])
        for name in ("PCA", "AE", "VAE_mu"):
            writer.writerow([name, 2, 10000, f"{metrics[name]:.8f}",
                             "VAE: decoder(mu), no sampling" if name == "VAE_mu"
                             else "deterministic reconstruction"])

    print("\nTest reconstruction errors (mean sum of squared pixels):")
    for name, value in metrics.items():
        print(f"  {name:7s}: {value:.6f}")
    print(f"\nSaved 3 PNGs and 1 CSV to: {OUT}")
    print("Note: VAE is evaluated using decoder(mu); this is NOT")
    print("the sampled expected reconstruction loss used in VAE training.")
    print("Latent axis scales/signs are not directly comparable.")


if __name__ == "__main__":
    main()
