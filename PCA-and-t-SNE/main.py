# ==========================================
# FEATURE ENGINEERING & DIMENSIONALITY REDUCTION
# MNIST DIGITS DATASET
# PCA and t-SNE using scikit-learn
# ==========================================

# ----------------------------
# 1. Import Libraries
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Set visualization style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,6)

# ----------------------------
# 2. Load MNIST Dataset
# ----------------------------
print("Downloading MNIST dataset...")

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# ----------------------------
# 3. Visualize Sample Images
# ----------------------------
def plot_sample_images(X, y):
    plt.figure(figsize=(10,6))
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(X[i].reshape(28,28), cmap='gray')
        plt.title(f"Label: {y[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_sample_images(X, y)

# ----------------------------
# 4. Feature Engineering
# ----------------------------
# Normalize pixel values
X = X / 255.0

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# For faster computation, use subset (important for t-SNE)
X_subset, _, y_subset, _ = train_test_split(
    X_scaled, y, train_size=10000, random_state=42, stratify=y
)

print("Subset shape:", X_subset.shape)

# ----------------------------
# 5. Principal Component Analysis (PCA)
# ----------------------------

print("Applying PCA...")

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_subset)

print("Original shape:", X_subset.shape)
print("Reduced shape after PCA:", X_pca.shape)

# Explained variance plot
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

# 2D PCA for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_subset)

plt.figure()
sns.scatterplot(
    x=X_pca_2d[:,0],
    y=X_pca_2d[:,1],
    hue=y_subset,
    palette="tab10",
    legend='full',
    s=15
)
plt.title("2D PCA Visualization")
plt.show()

# ----------------------------
# 6. t-SNE
# ----------------------------

print("Applying t-SNE (this may take a few minutes)...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    max_iter=1000,   # <-- changed here
    random_state=42
)

X_tsne = tsne.fit_transform(X_subset)

plt.figure()
sns.scatterplot(
    x=X_tsne[:,0],
    y=X_tsne[:,1],
    hue=y_subset,
    palette="tab10",
    legend='full',
    s=15
)
plt.title("t-SNE Visualization")
plt.show()

# ----------------------------
# 7. Comparison Summary
# ----------------------------
print("\n--- SUMMARY ---")
print("Original Dimensions:", X.shape[1])
print("After PCA (50 components):", X_pca.shape[1])
print("PCA retains approx:",
      round(np.sum(pca.explained_variance_ratio_)*100,2),
      "% variance")

print("\nObservation:")
print("- PCA captures global structure.")
print("- t-SNE preserves local neighborhood structure.")
print("- t-SNE shows clearer digit clustering.")

# ==========================================
# END OF ASSIGNMENT
# ==========================================