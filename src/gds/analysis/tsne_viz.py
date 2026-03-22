"""t-SNE visualization of sample embeddings colored by difficulty.

Produces a scatter plot similar to the reference: easy samples (cluster centers)
vs difficult samples (cluster boundaries), with random baseline for comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings from model's penultimate layer.

    Returns (embeddings, labels, sample_ids).
    """
    model.eval()
    model.to(device)
    all_emb, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for x, y, sids in loader:
            x = x.to(device)
            emb = model.get_embeddings(x)
            all_emb.append(emb.cpu().numpy())
            all_labels.append(y.numpy())
            all_ids.append(sids.numpy())

    return (
        np.concatenate(all_emb, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_ids, axis=0),
    )


def compute_tsne(embeddings: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """Run t-SNE on embeddings. Returns 2D coordinates."""
    from sklearn.manifold import TSNE

    # Adjust perplexity if too high for the number of samples
    n = embeddings.shape[0]
    perp = min(perplexity, max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        max_iter=1000,
        method="barnes_hut",
    )
    return tsne.fit_transform(embeddings.astype(np.float64))


def categorize_samples(
    sample_ids: np.ndarray,
    scores: np.ndarray,
    n_per_category: int = 1000,
    rng_seed: int = 42,
) -> dict[str, np.ndarray]:
    """Categorize samples into easy, hard, and random subsets.

    Returns dict mapping category name -> array of indices into sample_ids.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(sample_ids)
    sorted_idx = np.argsort(scores)

    n_pick = min(n_per_category, n // 4)

    # Easy = lowest scores (never or rarely forgotten)
    easy_idx = sorted_idx[:n_pick]
    # Hard = highest scores (most frequently forgotten)
    hard_idx = sorted_idx[-n_pick:]
    # Random = random subset
    random_idx = rng.choice(n, size=n_pick, replace=False)

    return {
        "easy": easy_idx,
        "hard": hard_idx,
        "random": random_idx,
    }


def plot_tsne(
    coords_2d: np.ndarray,
    categories: dict[str, np.ndarray],
    title: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Create t-SNE scatter plot with color-coded difficulty categories."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {
        "random": ("gray", "o", 0.3, 15),
        "easy": ("orange", "x", 0.7, 25),
        "hard": ("purple", "x", 0.7, 25),
    }

    # Plot random first (background), then easy and hard on top
    for cat_name in ["random", "easy", "hard"]:
        if cat_name not in categories:
            continue
        idx = categories[cat_name]
        color, marker, alpha, size = colors[cat_name]
        ax.scatter(
            coords_2d[idx, 0],
            coords_2d[idx, 1],
            c=color,
            marker=marker,
            alpha=alpha,
            s=size,
            label=cat_name,
            linewidths=0.5,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.15)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE plot saved: {out_path}")
