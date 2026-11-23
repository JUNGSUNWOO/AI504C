
# ============================================================
# AI504c Midterm Project
# Problem 1. Principal Component Analysis (PCA)
#
# 1.1 First five principal components
# 1.2 Reconstruction using five components
# 1.3 Reconstruction error with different M
# 1.4 Reconstruction of test images
# 1.5 2D visualization using PCA
#
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------
# 0. Import preprocessed data
# ------------------------------------------------------------


# Save figures in a folder dedicated to this problem.
RESULTS_DIR = Path(__file__).resolve().parent / "result" / "problem1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_figure(filename):
    path = RESULTS_DIR / filename
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    print(f"[Saved] {path}")

from data_preprocess import (
    x_train_pca,
    x_test_pca,
    x_train_img,
    x_test_img,
    train_mean,
    train_scale,
    y_train,
    sample_index_1,
    sample_indices_5,
    class_indices
)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------

SEED = 42

IMAGE_SIZE = 28
INPUT_DIM = 784

COMPONENTS = [2, 5, 10, 20, 50, 200]

N_TRAIN = x_train_pca.shape[0]
N_TEST = x_test_pca.shape[0]

torch.manual_seed(SEED)

print("=" * 60)
print("Problem 1. Principal Component Analysis")
print("=" * 60)

print("Training data:", x_train_pca.shape)
print("Test data    :", x_test_pca.shape)


# ============================================================
# 2. PCA: Eigen Decomposition
# ============================================================

# Covariance matrix of standardized training data
#
# C = (1/N) X^T X
#
# X: (60000, 784)
# C: (784, 784)

print("\n[Step 1] Computing covariance matrix...")

cov_matrix = (
    x_train_pca.T @ x_train_pca
) / N_TRAIN

print("Covariance matrix:", cov_matrix.shape)


# Eigenvalue decomposition
#
# C B = B Lambda

print("\n[Step 2] Eigenvalue decomposition...")

eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)

# Sort eigenvalues in descending order
sorted_indices = torch.argsort(
    eigenvalues,
    descending=True
)

eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Principal component basis
#
# B = [b1, b2, ..., bM]
#
# Shape: (784, 784)

B = eigenvectors

print("Eigenvalues shape :", eigenvalues.shape)
print("Eigenvectors shape:", B.shape)

print("\nTop 10 eigenvalues:")
print(eigenvalues[:10])


# ============================================================
# 3. Define PCA projection and reconstruction functions
# ============================================================

def pca_encode(X, M):
    """
    Project standardized data into M-dimensional latent space.

    X: (N, 784)
    B: (784, M)

    Return:
        Z: (N, M)
    """

    B_M = B[:, :M]

    Z = X @ B_M

    return Z


def pca_decode(Z, M):
    """
    Reconstruct standardized data from latent representation.

    Z: (N, M)

    Return:
        X_reconstructed: (N, 784)
    """

    B_M = B[:, :M]

    X_reconstructed = Z @ B_M.T

    return X_reconstructed


def inverse_standardization(X_standardized):
    """
    Restore original pixel space.

    r = mean + scale * x
    """

    X_original = (
        X_standardized * train_scale
        + train_mean
    )

    return X_original


def reconstruct_images(X, M):
    """
    Complete PCA reconstruction.

    Input:
        X: standardized data

    Output:
        Reconstructed images in original pixel space [0, 1]
    """

    Z = pca_encode(X, M)

    X_reconstructed = pca_decode(Z, M)

    R_reconstructed = inverse_standardization(
        X_reconstructed
    )

    return R_reconstructed


def reconstruction_error(X_original, X_reconstructed):
    """
    Calculate average per-image squared L2 reconstruction error.

    J_M = (1/K) * sum ||r_n - r_tilde_n||^2
    """

    # Flatten images into 784-dimensional vectors
    X_original = X_original.reshape(-1, 784)
    X_reconstructed = X_reconstructed.reshape(-1, 784)

    # Squared L2 error for each image
    squared_error = (
        (X_original - X_reconstructed) ** 2
    ).sum(dim=1)

    # Average over all images
    J_M = squared_error.mean()

    return J_M.item()


# ============================================================
# Problem 1.1
# Plot the first five principal components
# ============================================================

print("\n" + "=" * 60)
print("Problem 1.1. First Five Principal Components")
print("=" * 60)

fig, axes = plt.subplots(
    1, 5,
    figsize=(15, 4)
)

# Symmetric color scale around zero
vmax = B[:, :5].abs().max().item()

for i, ax in enumerate(axes):

    component = B[:, i].reshape(
        IMAGE_SIZE,
        IMAGE_SIZE
    )

    im = ax.imshow(
        component.numpy(),
        cmap="gray",
        vmin=-vmax,
        vmax=vmax
    )

    ax.set_title(f"PC {i+1}")
    ax.axis("off")

fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    shrink=0.7,
    label="Eigenvector value"
)

plt.suptitle(
    "First Five Principal Components",
    fontsize=15
)

save_figure("problem1_1_first_five_components.png")
plt.show()

print("Symmetric color range:", -vmax, "~", vmax)


# ============================================================
# Problem 1.2
# Reconstruction using the first five components
# ============================================================

print("\n" + "=" * 60)
print("Problem 1.2. Reconstruction using Five Components")
print("=" * 60)

# Select one fixed training image
idx = sample_index_1

# Original standardized image
x = x_train_pca[idx]

# Original pixel-space image
r = x_train_img[idx]

# Project into 5-dimensional latent space
z = pca_encode(x.unsqueeze(0), 5).squeeze(0)

# Reconstruct standardized image
x_reconstructed = pca_decode(
    z.unsqueeze(0), 5
).squeeze(0)

# Restore original pixel space
r_reconstructed = inverse_standardization(
    x_reconstructed
)

# ------------------------------------------------------------
# Individual component contributions
#
# z_m * D_s * b_m
#
# ------------------------------------------------------------

contributions = []

for m in range(5):

    contribution = (
        z[m] * train_scale * B[:, m]
    )

    contributions.append(
        contribution.reshape(28, 28)
    )

# ------------------------------------------------------------
# Plot original, reconstruction and contributions
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2, 4,
    figsize=(14, 7)
)

# Original image
axes[0, 0].imshow(
    r.reshape(28, 28).numpy(),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 0].set_title("Original")

# Reconstructed image
axes[0, 1].imshow(
    r_reconstructed.reshape(28, 28).numpy(),
    cmap="gray",
    vmin=0,
    vmax=1
)

axes[0, 1].set_title("Reconstruction (M=5)")

# Five individual component contributions
for m in range(5):

    ax = axes.flat[m + 2]

    contribution = contributions[m]

    contribution_vmax = contribution.abs().max().item()

    ax.imshow(
        contribution.numpy(),
        cmap="gray",
        vmin=-contribution_vmax,
        vmax=contribution_vmax
    )

    ax.set_title(f"PC {m+1} Contribution")

# Hide unused axes
axes[1, 3].axis("off")

for ax in axes.flat:
    ax.axis("off")

plt.suptitle(
    f"Image Reconstruction using Five PCs (Index={idx})",
    fontsize=14
)

plt.tight_layout()
save_figure("problem1_2_reconstruction_contributions.png")
plt.show()

# ------------------------------------------------------------
# Single image reconstruction error
# ------------------------------------------------------------

# Reshape both images into 784-dimensional vectors
r_flat = r.reshape(-1)
r_reconstructed_flat = r_reconstructed.reshape(-1)

single_error = torch.sum(
    (r_flat - r_reconstructed_flat) ** 2
).item()

print("Sample index:", idx)
print("Latent representation z:")
print(z.numpy())

print("\nSingle image reconstruction error:")
print(f"||r - r_tilde||^2 = {single_error:.6f}")


# ------------------------------------------------------------
# Average training reconstruction error J_5
# ------------------------------------------------------------

print("\nCalculating J_5 over all training images...")

train_reconstructed_5 = reconstruct_images(
    x_train_pca,
    5
)

J5 = reconstruction_error(
    x_train_img,
    train_reconstructed_5
)

print(f"Training J_5 = {J5:.6f}")


# ============================================================
# Problem 1.3
# Reconstruction error with different numbers of PCs
# ============================================================

print("\n" + "=" * 60)
print("Problem 1.3. Reconstruction Error with Different M")
print("=" * 60)

# ------------------------------------------------------------
# 1.3 (a) Visualize five fixed training images
# ------------------------------------------------------------

for M in COMPONENTS:

    print(f"\nPlotting reconstruction with M={M}")

    fig, axes = plt.subplots(
        5, 2,
        figsize=(7, 15)
    )

    for i, sample_idx in enumerate(sample_indices_5):

        # Original image
        original = x_train_img[sample_idx]

        # Reconstructed image
        reconstructed = reconstruct_images(
            x_train_pca[sample_idx].unsqueeze(0),
            M
        ).squeeze(0)

        # Original
        axes[i, 0].imshow(
            original.reshape(28, 28).numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 0].set_title(
            f"Original (Index={sample_idx})"
        )

        # Reconstruction
        axes[i, 1].imshow(
            reconstructed.reshape(28, 28).numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[i, 1].set_title(
            f"Reconstruction (M={M})"
        )

        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    plt.suptitle(
        f"PCA Reconstruction (M={M})",
        fontsize=15
    )

    plt.tight_layout()
    save_figure(f"problem1_3_reconstruction_M{M}.png")
    plt.show()


# ------------------------------------------------------------
# 1.3 (b) Calculate J_M for training dataset
# ------------------------------------------------------------

train_errors = []

print("\nTraining reconstruction errors")

for M in COMPONENTS:

    reconstructed = reconstruct_images(
        x_train_pca,
        M
    )

    J_M = reconstruction_error(
        x_train_img,
        reconstructed
    )

    train_errors.append(J_M)

    print(f"M={M:3d} | J_M={J_M:.6f}")


# ------------------------------------------------------------
# Plot training reconstruction error
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    COMPONENTS,
    train_errors,
    marker="o",
    linewidth=2,
    label="Training"
)

plt.xlabel("Number of Principal Components (M)")
plt.ylabel("Reconstruction Error (J_M)")

plt.title("Training Reconstruction Error vs M")

plt.xticks(COMPONENTS)
plt.grid(True)
plt.legend()

save_figure("problem1_3_train_error_vs_M.png")
plt.show()


# ============================================================
# Problem 1.4
# Reconstruction of test images
# ============================================================

print("\n" + "=" * 60)
print("Problem 1.4. Reconstruction of Test Images")
print("=" * 60)

test_errors = []

print("\nTest reconstruction errors")

for M in COMPONENTS:

    reconstructed = reconstruct_images(
        x_test_pca,
        M
    )

    J_M = reconstruction_error(
        x_test_img,
        reconstructed
    )

    test_errors.append(J_M)

    print(f"M={M:3d} | J_M={J_M:.6f}")


# ------------------------------------------------------------
# Compare training and test reconstruction errors
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    COMPONENTS,
    train_errors,
    marker="o",
    linewidth=2,
    label="Training"
)

plt.plot(
    COMPONENTS,
    test_errors,
    marker="s",
    linewidth=2,
    label="Test"
)

plt.xlabel("Number of Principal Components (M)")
plt.ylabel("Reconstruction Error (J_M)")

plt.title("PCA Reconstruction Error: Train vs Test")

plt.xticks(COMPONENTS)
plt.grid(True)
plt.legend()

save_figure("problem1_4_train_vs_test_error.png")
plt.show()


# ============================================================
# Problem 1.5
# 2D visualization of MNIST using PCA
# ============================================================

print("\n" + "=" * 60)
print("Problem 1.5. 2D Visualization using PCA")
print("=" * 60)

# ------------------------------------------------------------
# Select 100 images for each digit: 0, 1, 9
# ------------------------------------------------------------

digits = [0, 1, 9]

colors = {
    0: "blue",
    1: "red",
    9: "green"
}

plt.figure(figsize=(9, 7))

for digit in digits:

    # Fixed sample indices from preprocessing
    indices = class_indices[digit]

    # Standardized input data
    X_digit = x_train_pca[indices]

    # Project into 2D latent space
    Z_digit = pca_encode(
        X_digit,
        2
    )

    # Plot
    plt.scatter(
        Z_digit[:, 0].numpy(),
        Z_digit[:, 1].numpy(),
        c=colors[digit],
        label=f"Digit {digit}",
        alpha=0.7,
        s=35
    )

plt.xlabel("Principal Component 1 (z1)")
plt.ylabel("Principal Component 2 (z2)")

plt.title("2D Visualization of MNIST using PCA")

plt.legend()
plt.grid(True)
save_figure("problem1_5_pca_latent_scatter.png")
plt.show()


# ============================================================
# 4. Final summary
# ============================================================

print("\n" + "=" * 60)
print("Problem 1. Final Results")
print("=" * 60)

print("\nTraining Reconstruction Errors")

for M, error in zip(COMPONENTS, train_errors):
    print(f"M={M:3d}: {error:.6f}")

print("\nTest Reconstruction Errors")

for M, error in zip(COMPONENTS, test_errors):
    print(f"M={M:3d}: {error:.6f}")

print("\nPCA implementation completed successfully!")
