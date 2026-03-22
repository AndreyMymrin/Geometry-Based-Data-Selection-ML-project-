
from __future__ import annotations
 
from typing import Any
 
import numpy as np
import pandas as pd
 
# def erank(X: np.ndarray, eps: float = 1e-12) -> float:
#     if X.ndim != 2:
#         raise ValueError(f"Expected 2-D matrix, got shape {X.shape}.")
#     if X.shape[0] < 2:
#         raise ValueError("ERank requires at least 2 rows (tokens).")
 
#     sigma = np.linalg.svd(X.astype(np.float64), compute_uv=False)
 
#     total = sigma.sum()
#     if total < eps:
#         return 1.0
 
#     p = sigma / total                          # normalise: p_k = σ_k / Σσ_i
#     p_safe = np.maximum(p, eps)               # avoid log(0)
#     H = -float(np.sum(p * np.log(p_safe)))    # Shannon entropy
#     return float(np.exp(H))
 

def _entropy_from_sigma(sigma: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of the normalised singular-value distribution."""
    total = sigma.sum()
    if total < eps:
        return 0.0                      
    p = sigma / total
    p_safe = np.maximum(p, eps)
    return float(-np.sum(p * np.log(p_safe)))


def erank(X: np.ndarray, eps: float = 1e-12) -> float:
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("ERank requires at least 2 rows (tokens).")

    X64 = X.astype(np.float64)
    n, d = X64.shape

    if n < d:
        G = X64 @ X64.T                              # (n, n), symmetric
        eigs = np.linalg.eigvalsh(G)                 # ascending order, O(n³)
        eigs = np.maximum(eigs, 0.0)                 # clamp numerical negatives
        sigma = np.sqrt(eigs)
    else:
        sigma = np.linalg.svd(X64, compute_uv=False)

    H = _entropy_from_sigma(sigma, eps)
    return float(np.exp(H))

def layerwise_erank(
    hidden_states: list[np.ndarray],
) -> np.ndarray:
    return np.array([erank(X) for X in hidden_states], dtype=np.float64)
 
 
def average_erank(hidden_states: list[np.ndarray]) -> float:
    scores = layerwise_erank(hidden_states)
    return float(scores.mean())
 
 
class EffectiveRankScorer:
    def __init__(self) -> None:
        self._last_metadata: dict[str, Any] = {}
 
    @property
    def name(self) -> str:
        return "paper_erank"
 
    def build_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)
 
    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "hidden_states" not in metadata:
            raise ValueError(
                "metadata['hidden_states'] is required. "
                "Expected a list of N per-sample hidden-state lists, "
                "each containing L arrays of shape (n_tokens, d_hidden)."
            )
 
        all_hidden: list[list[np.ndarray]] = metadata["hidden_states"]
        n_samples = len(sample_ids)
 
        if len(all_hidden) != n_samples:
            raise ValueError(
                f"hidden_states has {len(all_hidden)} entries "
                f"but sample_ids has {n_samples}."
            )
 
        scores = np.array(
            [average_erank(hs) for hs in all_hidden],
            dtype=np.float64,
        )
 
        ranks = np.argsort(np.argsort(-scores)) + 1
 
        n_layers = len(all_hidden[0]) if all_hidden else 0
        self._last_metadata = {
            "method":     self.name,
            "n_samples":  n_samples,
            "n_layers":   n_layers,
            "mean_score": float(scores.mean()),
            "min_score":  float(scores.min()),
            "max_score":  float(scores.max()),
        }
 
        return (
            pd.DataFrame(
                {
                    "sample_id": np.asarray(sample_ids, dtype=np.int64),
                    "label":     np.asarray(labels, dtype=np.int64),
                    "score":     scores,
                    "rank":      ranks.astype(int),
                    "method":    self.name,
                }
            )
            .sort_values("rank", kind="stable")
            .reset_index(drop=True)
        )
 
def rank_generators(
    generator_hidden_states: dict[str, list[list[np.ndarray]]],
) -> pd.DataFrame:
    """Rank generator models by average ERank across their outputs.
 
    Replicates the paper's Table 1 experiment for ERank:
    given hidden states extracted by a single tester model for texts
    from each generator, compute s̄^R(X_g) per generator and rank them.
 
    Parameters
    ----------
    generator_hidden_states : dict[str, list[list[np.ndarray]]]
        Keys are generator model names.
        Values are lists of per-text hidden states:
          outer list  → texts (e.g. 1 000 reviews)
          middle list → layers L
          inner array → shape (n_tokens, d_hidden)
 
    Returns
    -------
    pd.DataFrame
        Columns: generator, mean_erank, rank.
        Sorted by rank ascending (rank 1 = most human-like).
    """
    rows = []
    for generator_name, texts in generator_hidden_states.items():
        text_scores = [average_erank(hs) for hs in texts]
        rows.append(
            {
                "generator":   generator_name,
                "mean_erank":  float(np.mean(text_scores)),
                "std_erank":   float(np.std(text_scores)),
                "n_texts":     len(texts),
            }
        )
 
    df = pd.DataFrame(rows).sort_values("mean_erank", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)