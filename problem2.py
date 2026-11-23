
# ============================================================
# AI504c Midterm Project
# Problem 2. Nonlinear Dimensionality Reduction
# Autoencoder
#
# 2.1 (a) Training and Test Learning Curve (M=2)
# 2.1 (b) Comparison with PCA (M=2, 5, 10)
# 2.2 (a) 2D Visualization using Autoencoder
# ============================================================

from pathlib import Path
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# 0. Import preprocessed data
# ------------------------------------------------------------


# Save figures in a folder dedicated to this problem.
RESULTS_DIR = Path(__file__).resolve().parent / "result" / "problem2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(filename):
    path = RESULTS_DIR / filename
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    print(f"[Saved] {path}")

from data_preprocess import (
    x_train_img,
    x_test_img,
    x_train_pca,
    x_test_pca,
    train_mean,
    train_scale,
    class_indices
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

INPUT_DIM = 784
HIDDEN_DIM = 128

LATENT_DIMS = [2, 5, 10]

LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Problem 2. Autoencoder")
print("=" * 60)

print("Device:", DEVICE)


# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. Prepare Autoencoder Dataset
# ============================================================

# Autoencoder uses original [0, 1] pixel data.
# No centering or standardization is applied.

x_train_ae = x_train_img.reshape(-1, INPUT_DIM).float()
x_test_ae = x_test_img.reshape(-1, INPUT_DIM).float()

train_dataset_ae = TensorDataset(x_train_ae)
test_dataset_ae = TensorDataset(x_test_ae)

print("\nDataset Information")
print("-" * 40)

print("Train:", x_train_ae.shape)
print("Test :", x_test_ae.shape)

print(
    "Train pixel range:",
    x_train_ae.min().item(),
    "~",
    x_train_ae.max().item()
)


# ============================================================
# 3. Define Autoencoder Model
# ============================================================

class Autoencoder(nn.Module):

    def __init__(
        self,
        input_dim=784,
        hidden_dim=128,
        latent_dim=2
    ):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(input_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, latent_dim)

        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim),

            nn.Sigmoid()

        )


    def encode(self, x):

        return self.encoder(x)


    def decode(self, z):

        return self.decoder(z)


    def forward(self, x):

        z = self.encode(x)

        reconstructed = self.decode(z)

        return reconstructed


# ============================================================
# 4. Define Reconstruction Loss
# ============================================================

def reconstruction_loss(original, reconstructed):

    """
    J_M = (1/N) * sum ||x_n - x_tilde_n||^2

    Sum over 784 pixels first,
    then average over images.
    """

    loss_per_image = (
        (original - reconstructed) ** 2
    ).sum(dim=1)

    return loss_per_image.mean()


# ============================================================
# 5. Evaluation Function
# ============================================================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for (images,) in dataloader:

            images = images.to(DEVICE)

            reconstructed = model(images)

            loss = reconstruction_loss(
                images,
                reconstructed
            )

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# 6. Training Function
# ============================================================

def train_autoencoder(latent_dim, plot_learning_curve=False):

    print("\n" + "=" * 60)
    print(f"Training Autoencoder (M={latent_dim})")
    print("=" * 60)

    set_seed(SEED)

    # DataLoader
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset_ae,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset_ae,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = Autoencoder(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=latent_dim
    ).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_history = []
    test_history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0
        total_samples = 0

        for (images,) in train_loader:

            images = images.to(DEVICE)

            # Forward
            reconstructed = model(images)

            loss = reconstruction_loss(
                images,
                reconstructed
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size = images.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        # Average training error
        train_error = (
            total_train_loss / total_samples
        )

        # Test error
        test_error = evaluate(
            model,
            test_loader
        )

        train_history.append(train_error)
        test_history.append(test_error)

        print(
            f"Epoch [{epoch+1:02d}/{EPOCHS}] "
            f"| Train J: {train_error:.6f} "
            f"| Test J: {test_error:.6f}"
        )

    # --------------------------------------------------------
    # Save model checkpoint
    # --------------------------------------------------------

    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_path = (
        f"checkpoints/autoencoder_M{latent_dim}.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "latent_dim": latent_dim,
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "seed": SEED
        },
        checkpoint_path
    )

    print("Checkpoint saved:", checkpoint_path)

    # --------------------------------------------------------
    # Plot learning curve
    # --------------------------------------------------------

    if plot_learning_curve:

        epochs_range = range(1, EPOCHS + 1)

        plt.figure(figsize=(9, 5))

        plt.plot(
            epochs_range,
            train_history,
            marker="o",
            label="Training"
        )

        plt.plot(
            epochs_range,
            test_history,
            marker="s",
            label="Test"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Error (J_M)")

        plt.title(
            f"Autoencoder Learning Curve (M={latent_dim})"
        )

        plt.xticks(epochs_range)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        save_figure(f"problem2_1a_ae_learning_curve_M{latent_dim}.png")
        plt.show()

    return model, train_history, test_history


# ============================================================
# Problem 2.1 (a)
# Train Autoencoder with M=2
# Plot Training and Test Learning Curves
# ============================================================

print("\n" + "=" * 60)
print("Problem 2.1 (a)")
print("=" * 60)

model_2, train_history_2, test_history_2 = (
    train_autoencoder(
        latent_dim=2,
        plot_learning_curve=True
    )
)


# ============================================================
# Problem 2.1 (b)
# Train Autoencoder with M=5 and M=10
# ============================================================

print("\n" + "=" * 60)
print("Problem 2.1 (b)")
print("=" * 60)

models = {
    2: model_2
}

for M in [5, 10]:

    model, train_history, test_history = (
        train_autoencoder(
            latent_dim=M,
            plot_learning_curve=False
        )
    )

    models[M] = model


# ============================================================
# 7. Calculate Autoencoder Reconstruction Errors
# ============================================================

print("\n" + "=" * 60)
print("Autoencoder Reconstruction Errors")
print("=" * 60)

train_loader_eval = DataLoader(
    train_dataset_ae,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader_eval = DataLoader(
    test_dataset_ae,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

ae_train_errors = {}
ae_test_errors = {}

for M in LATENT_DIMS:

    model = models[M]

    train_error = evaluate(
        model,
        train_loader_eval
    )

    test_error = evaluate(
        model,
        test_loader_eval
    )

    ae_train_errors[M] = train_error
    ae_test_errors[M] = test_error

    print(
        f"M={M:3d} | "
        f"Train J={train_error:.6f} | "
        f"Test J={test_error:.6f}"
    )


# ============================================================
# 8. PCA Baseline for Fair Comparison
# ============================================================

print("\n" + "=" * 60)
print("Calculating PCA Baseline")
print("=" * 60)

# PCA basis is fitted using training data only.

N_TRAIN = x_train_pca.shape[0]

cov_matrix = (
    x_train_pca.T @ x_train_pca
) / N_TRAIN

eigenvalues, eigenvectors = torch.linalg.eigh(
    cov_matrix
)

sorted_indices = torch.argsort(
    eigenvalues,
    descending=True
)

B = eigenvectors[:, sorted_indices]


def pca_reconstruct(X, M):

    B_M = B[:, :M]

    Z = X @ B_M

    X_reconstructed = Z @ B_M.T

    # Restore original [0, 1] pixel space
    R_reconstructed = (
        X_reconstructed * train_scale
        + train_mean
    )

    return R_reconstructed


def calculate_pca_error(X_original, X_standardized, M):

    reconstructed = pca_reconstruct(
        X_standardized,
        M
    )

    original_flat = X_original.reshape(-1, INPUT_DIM)

    squared_error = (
        (original_flat - reconstructed) ** 2
    ).sum(dim=1)

    return squared_error.mean().item()


pca_train_errors = {}
pca_test_errors = {}

for M in LATENT_DIMS:

    train_error = calculate_pca_error(
        x_train_ae,
        x_train_pca,
        M
    )

    test_error = calculate_pca_error(
        x_test_ae,
        x_test_pca,
        M
    )

    pca_train_errors[M] = train_error
    pca_test_errors[M] = test_error

    print(
        f"M={M:3d} | "
        f"Train J={train_error:.6f} | "
        f"Test J={test_error:.6f}"
    )


# ============================================================
# 9. Compare PCA and Autoencoder
# ============================================================

print("\n" + "=" * 60)
print("PCA vs Autoencoder")
print("=" * 60)

print("\nTraining Errors")

for M in LATENT_DIMS:

    print(
        f"M={M:3d} | "
        f"PCA={pca_train_errors[M]:.6f} | "
        f"AE={ae_train_errors[M]:.6f}"
    )


print("\nTest Errors")

for M in LATENT_DIMS:

    print(
        f"M={M:3d} | "
        f"PCA={pca_test_errors[M]:.6f} | "
        f"AE={ae_test_errors[M]:.6f}"
    )


# ------------------------------------------------------------
# Plot Training Error Comparison
# ------------------------------------------------------------

x_axis = np.arange(len(LATENT_DIMS))

width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x_axis - width / 2,
    [pca_train_errors[M] for M in LATENT_DIMS],
    width,
    label="PCA"
)

plt.bar(
    x_axis + width / 2,
    [ae_train_errors[M] for M in LATENT_DIMS],
    width,
    label="Autoencoder"
)

plt.xticks(
    x_axis,
    LATENT_DIMS
)

plt.xlabel("Latent Dimension (M)")
plt.ylabel("Reconstruction Error (J_M)")

plt.title("Training Error: PCA vs Autoencoder")

plt.legend()
plt.grid(axis="y")

plt.tight_layout()
save_figure("problem2_1b_train_pca_vs_ae.png")
plt.show()


# ------------------------------------------------------------
# Plot Test Error Comparison
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.bar(
    x_axis - width / 2,
    [pca_test_errors[M] for M in LATENT_DIMS],
    width,
    label="PCA"
)

plt.bar(
    x_axis + width / 2,
    [ae_test_errors[M] for M in LATENT_DIMS],
    width,
    label="Autoencoder"
)

plt.xticks(
    x_axis,
    LATENT_DIMS
)

plt.xlabel("Latent Dimension (M)")
plt.ylabel("Reconstruction Error (J_M)")

plt.title("Test Error: PCA vs Autoencoder")

plt.legend()
plt.grid(axis="y")

plt.tight_layout()
save_figure("problem2_1b_test_pca_vs_ae.png")
plt.show()


# ============================================================
# Problem 2.2 (a)
# 2D Visualization using Autoencoder
# ============================================================

print("\n" + "=" * 60)
print("Problem 2.2 (a)")
print("=" * 60)

model_2.eval()

digits = [0, 1, 9]

colors = {
    0: "blue",
    1: "red",
    9: "green"
}

plt.figure(figsize=(9, 7))

with torch.no_grad():

    for digit in digits:

        # Same fixed indices used in Problem 1.5
        indices = class_indices[digit]

        X_digit = x_train_ae[indices].to(DEVICE)

        # Encode images into 2D latent space
        Z_digit = model_2.encode(X_digit)

        Z_digit = Z_digit.cpu().numpy()

        plt.scatter(
            Z_digit[:, 0],
            Z_digit[:, 1],
            c=colors[digit],
            label=f"Digit {digit}",
            alpha=0.7,
            s=35
        )

plt.xlabel("Latent Dimension 1 (z1)")
plt.ylabel("Latent Dimension 2 (z2)")

plt.title(
    "2D Visualization of MNIST using Autoencoder"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
save_figure("problem2_2_ae_latent_scatter.png")
plt.show()


# ============================================================
# 10. Final Summary
# ============================================================

print("\n" + "=" * 60)
print("Problem 2 Completed")
print("=" * 60)

print("Device:", DEVICE)

print("\nFinal Reconstruction Errors")

for M in LATENT_DIMS:

    print(f"\nLatent Dimension M={M}")

    print(
        f"PCA Train: {pca_train_errors[M]:.6f}"
    )

    print(
        f"AE Train : {ae_train_errors[M]:.6f}"
    )

    print(
        f"PCA Test : {pca_test_errors[M]:.6f}"
    )

    print(
        f"AE Test  : {ae_test_errors[M]:.6f}"
    )

print("\nAll Autoencoder experiments completed!")
