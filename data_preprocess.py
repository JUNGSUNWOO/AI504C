
# ============================================================
# AI504c Midterm Project
# Step 1. Load and Preprocess MNIST Dataset
# ============================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# ------------------------------------------------------------
# 1. Set random seed
# ------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------------------------------------------
# 2. Load MNIST dataset
# ------------------------------------------------------------

DATA_PATH = "./data"

train_dataset = datasets.MNIST(
    root='./data',                  # 데이터셋을 저장할 로컬 하위 경로
    train=True,                     # 학습용 데이터셋 지정
    download=True,                  # 핵심 해결책: 로컬에 파일이 없으면 자동 다운로드
    transform=transforms.ToTensor() # (권장) PyTorch 모델 입력을 위한 텐서 변환
)

test_dataset = datasets.MNIST(
    root='./data',                  # 데이터셋을 저장할 로컬 하위 경로
    train=False,                     # 학습용 데이터셋 지정
    download=True,                  # 핵심 해결책: 로컬에 파일이 없으면 자동 다운로드
    transform=transforms.ToTensor() # (권장) PyTorch 모델 입력을 위한 텐서 변환
)

print("MNIST Dataset Loaded")
print("--------------------------------")
print(f"Training samples : {len(train_dataset)}")
print(f"Test samples     : {len(test_dataset)}")
print(f"Image shape      : {train_dataset.data.shape[1:]}")
print(f"Number of classes: {len(torch.unique(train_dataset.targets))}")

# ------------------------------------------------------------
# 3. Convert image data to float32 and normalize to [0, 1]
# ------------------------------------------------------------

# Original image shape: (N, 28, 28)
# Pixel range: [0, 255]

x_train_img = train_dataset.data.float() / 255.0
x_test_img = test_dataset.data.float() / 255.0

y_train = train_dataset.targets.long()
y_test = test_dataset.targets.long()

print("\nNormalized Data")
print("--------------------------------")
print(f"Training range: {x_train_img.min():.4f} ~ {x_train_img.max():.4f}")
print(f"Test range    : {x_test_img.min():.4f} ~ {x_test_img.max():.4f}")

# ------------------------------------------------------------
# 4. Flatten images (28 x 28 -> 784)
# ------------------------------------------------------------

x_train = x_train_img.reshape(-1, 784)
x_test = x_test_img.reshape(-1, 784)

print("\nFlattened Data")
print("--------------------------------")
print(f"Training shape: {x_train.shape}")
print(f"Test shape    : {x_test.shape}")

# ------------------------------------------------------------
# 5. PCA preprocessing
#    Centering and standardization using TRAINING statistics
# ------------------------------------------------------------

# Calculate mean and population standard deviation
train_mean = x_train.mean(dim=0)
train_std = x_train.std(dim=0, correction=0)

# Replace zero standard deviations with 1
train_scale = torch.where(
    train_std == 0,
    torch.ones_like(train_std),
    train_std
)

# Standardize training and test data
x_train_pca = (x_train - train_mean) / train_scale
x_test_pca = (x_test - train_mean) / train_scale

print("\nPCA Preprocessing")
print("--------------------------------")
print(f"Mean shape : {train_mean.shape}")
print(f"Scale shape: {train_scale.shape}")
print(f"Zero std features: {(train_std == 0).sum().item()}")
print(f"Training PCA shape: {x_train_pca.shape}")
print(f"Test PCA shape    : {x_test_pca.shape}")

# ------------------------------------------------------------
# 6. Fixed random sample indices
#    Reuse the same indices in later experiments
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)

# Five training images for PCA reconstruction comparison
sample_indices_5 = rng.choice(
    len(x_train),
    size=5,
    replace=False
)

# One training image for Problem 1.2
sample_index_1 = int(sample_indices_5[0])

# 100 samples for each digit: 0, 1, 9
class_indices = {}

for digit in [0, 1, 9]:

    candidates = torch.where(y_train == digit)[0].numpy()

    selected = rng.choice(
        candidates,
        size=100,
        replace=False
    )

    class_indices[digit] = selected

print("\nFixed Sample Indices")
print("--------------------------------")
print("Five sample indices:", sample_indices_5)
print("Single sample index :", sample_index_1)

for digit in [0, 1, 9]:
    print(f"Digit {digit}: {len(class_indices[digit])} samples")

# ------------------------------------------------------------
# 7. Visualize MNIST sample images
# ------------------------------------------------------------

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):

    ax.imshow(
        x_train_img[i].numpy(),
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.set_title(f"Label: {y_train[i].item()}")
    ax.axis("off")

plt.suptitle("MNIST Training Samples")
plt.tight_layout()
plt.close()

# ------------------------------------------------------------
# 8. Verify dataset consistency
# ------------------------------------------------------------

assert x_train.shape == (60000, 784)
assert x_test.shape == (10000, 784)

assert torch.isfinite(x_train_pca).all()
assert torch.isfinite(x_test_pca).all()

assert x_train_img.min() >= 0
assert x_train_img.max() <= 1

print("\nAll preprocessing checks passed!")

if __name__ == "__main__":
    import json
    from pathlib import Path
    Path("results").mkdir(exist_ok=True)
    indices = {"seed": SEED, "problem1_2_single_index": sample_index_1, "problem1_3_five_indices": sample_indices_5.tolist(), "class_indices": {str(k): v.tolist() for k, v in class_indices.items()}}
    Path("results/sample_indices.json").write_text(json.dumps(indices, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved results/sample_indices.json")


