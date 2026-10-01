"""
Image Compression with SVD
===========================

Compress a grayscale image using the Singular Value Decomposition (SVD),
the same linear algebra that underlies Principal Component Analysis (PCA).

A grayscale image is just a matrix A (height x width). SVD factors it as

        A = U @ diag(S) @ Vt

where the singular values in S are sorted largest-first. Keeping only the
top-k singular values (and the matching columns of U / rows of Vt) gives the
best possible rank-k approximation of A (Eckart-Young theorem):

        A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]

Storing A_k costs k * (h + w + 1) numbers instead of h * w, so for small k
we keep a fraction of the data while the image stays recognizable.

Run:
    python3 compress.py                # uses photo.jpg
    python3 compress.py my_image.png   # uses your own image
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Number of singular values to keep for each reconstruction.
K_VALUES = [5, 20, 50, 100]

FIGURES_DIR = "figures"


def load_grayscale(path):
    """Load an image from disk and return it as a float matrix in [0, 255]."""
    img = Image.open(path).convert("L")  # "L" = 8-bit grayscale
    return np.asarray(img, dtype=np.float64)


def compress(matrix, k):
    """Return the best rank-k approximation of `matrix` via truncated SVD."""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return np.clip(approx, 0, 255)


def storage_ratio(shape, k):
    """Fraction of the original data needed to store a rank-k approximation."""
    h, w = shape
    original = h * w
    compressed = k * (h + w + 1)  # U columns + Vt rows + singular values
    return compressed / original


def singular_values(matrix):
    """Return just the singular values of `matrix`, largest first."""
    return np.linalg.svd(matrix, compute_uv=False)


def save_comparison(original, k_values, out_path):
    """Save a side-by-side grid: original + one reconstruction per k."""
    n = len(k_values) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))

    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], k_values):
        approx = compress(original, k)
        pct = storage_ratio(original.shape, k) * 100
        ax.imshow(approx, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"k = {k}\n~{pct:.0f}% of data")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_singular_values(matrix, out_path):
    """Save a log-scale plot of the singular value spectrum."""
    S = singular_values(matrix)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(1, len(S) + 1), S, color="#2b6cb0")
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("Singular values decay rapidly")
    ax.grid(True, which="both", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"

    if not os.path.exists(path):
        sys.exit(
            f"Image not found: {path}\n"
            "Pass an image path, or add photo.jpg to this folder."
        )

    original = load_grayscale(path)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Loaded {path}  ->  {original.shape[0]} x {original.shape[1]} grayscale")

    print("\n k  | storage | reconstruction error (RMSE)")
    print("----+---------+-----------------------------")
    for k in K_VALUES:
        approx = compress(original, k)
        rmse = np.sqrt(np.mean((original - approx) ** 2))
        pct = storage_ratio(original.shape, k) * 100
        print(f"{k:>3} | {pct:>5.1f}%  | {rmse:>8.2f}")

    comparison_path = os.path.join(FIGURES_DIR, "comparison.png")
    spectrum_path = os.path.join(FIGURES_DIR, "singular_values.png")

    save_comparison(original, K_VALUES, comparison_path)
    save_singular_values(original, spectrum_path)

    print(f"\nSaved {comparison_path}")
    print(f"Saved {spectrum_path}")


if __name__ == "__main__":
    main()
