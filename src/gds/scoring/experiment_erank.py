"""
paper_erank_optimized.py
========================
Drop-in replacement for paper_erank.py with one targeted algorithmic
improvement: the gram-matrix path for erank().

BACKGROUND
----------
The original implementation calls:

    sigma = np.linalg.svd(X, compute_uv=False)   # X is (n, d)

SVD of an (n, d) matrix costs O(min(n,d)^2 * max(n,d)).

In practice X = X^(l) contains one row per *token*, so n = n_tokens
(typically 20–512) and d = hidden_dim (typically 512–4096).
Because n << d is the overwhelmingly common regime, the SVD is doing
far more work than necessary.

THE FIX: GRAM MATRIX PATH
--------------------------
The singular values of X are the square roots of the eigenvalues of
the symmetric gram matrix G = X @ X.T  (shape n×n):

    σ_k(X)  =  √λ_k(G)

Building G costs O(n² d) — same leading term as SVD — but the
constant is much smaller (a single BLAS-3 matmul vs a full LAPACK
bidiagonal reduction).  Eigendecomposing the n×n gram matrix then
costs only O(n³), which is negligible when n << d.

Empirical speedups measured on this hardware (n=100, d=4096):
  n=50,  d=512:   15× faster than SVD path
  n=50,  d=4096:   8× faster
  n=100, d=512:   21× faster
  n=100, d=4096:  18× faster
  n=256, d=4096:  11× faster
  n=512, d=4096:   8× faster

Results are numerically identical to the SVD path (max diff < 1e-6).

The gram path is used whenever n < d.  When n >= d the SVD path is
used unchanged (it is never slower in that regime).

COMPARISON TO ID ESTIMATORS
-----------------------------
The paper also computes Intrinsic Dimensionality (ID) via MLE and
CorrInt.  Both require all O(n²) pairwise distances computed in R^d,
costing O(n² d) with *no* dimensionality reduction before the main
computation, and with large constants (a Python distance loop or
a dense (n, n, d) intermediate tensor).

Benchmarks at n=100, d=4096 (32 layers):
  ERank (gram path, per layer):  ~4 ms
  MLE   (per layer):             ~925 ms   → ~230× slower
  CorrInt (per layer):           ~927 ms   → ~232× slower

ERank is faster because:
  1. The gram matrix reduces the problem from O(n²d) with big constants
     to O(n²d) + O(n³) with small constants (pure BLAS-3).
  2. n³ is negligible relative to n²d when d >> n.
  3. MLE/CorrInt need explicit pairwise distances in R^d (no reduction),
     and CorrInt additionally sweeps over ε values.

WHAT DOES NOT HELP
------------------
- Batching all L layers into (L, n, d) and using a single matmul for
  the gram matrices: no speedup — BLAS already saturates threads per
  individual call.
- Batching eigvalsh across layers: no speedup (same reason).
- The entropy formula accounts for < 0.1 ms total and is already
  vectorised in both implementations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core mathematical primitives
# ---------------------------------------------------------------------------

def _entropy_from_sigma(sigma: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of the normalised singular-value distribution."""
    total = sigma.sum()
    if total < eps:
        return 0.0                      # degenerate → H = 0 → ERank = exp(0) = 1
    p = sigma / total
    p_safe = np.maximum(p, eps)
    return float(-np.sum(p * np.log(p_safe)))


def erank(X: np.ndarray, eps: float = 1e-12) -> float:
    """Effective Rank of a 2-D activation matrix.

    Implements:
        p_k  = σ_k / Σ_i σ_i
        ERank(X) = exp( −Σ_k p_k log p_k )

    as described in Roy & Vetterli (2007), cited by Yusupov et al. (2025).

    Optimisation: when n < d (tokens < hidden dim, the common case)
    the singular values are computed via the n×n gram matrix
    G = X @ X.T rather than a full SVD of X, yielding 8–21× speedups
    at no loss of precision.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    eps : float
        Numerical floor for log and normalisation.

    Returns
    -------
    float  —  ERank ∈ [1, min(n, d)]
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("ERank requires at least 2 rows (tokens).")

    X64 = X.astype(np.float64)
    n, d = X64.shape

    if n < d:
        # --- gram path: O(n² d) matmul + O(n³) eigh ---
        G = X64 @ X64.T                              # (n, n), symmetric
        eigs = np.linalg.eigvalsh(G)                 # ascending order, O(n³)
        eigs = np.maximum(eigs, 0.0)                 # clamp numerical negatives
        sigma = np.sqrt(eigs)
    else:
        # --- SVD path: O(d² n) for wide X or O(n² d) for tall ---
        sigma = np.linalg.svd(X64, compute_uv=False)

    H = _entropy_from_sigma(sigma, eps)
    return float(np.exp(H))


def layerwise_erank(hidden_states: list[np.ndarray]) -> np.ndarray:
    """Per-layer ERank for one text's hidden states.

    Parameters
    ----------
    hidden_states : list of np.ndarray, each shape (n_l, d_l)

    Returns
    -------
    np.ndarray, shape (L,), dtype float64
    """
    return np.array([erank(X) for X in hidden_states], dtype=np.float64)


def average_erank(hidden_states: list[np.ndarray]) -> float:
    """Mean ERank across layers — the quality score for one text.

    Implements Equation (1) from Yusupov et al. (2025):
        s̄^R(X_g) = (1/L) Σ_{l=1}^{L} R(X^(l)_g)
    """
    return float(layerwise_erank(hidden_states).mean())


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------

class PaperEffectiveRankScorer:
    """Reference-free text quality scorer (Yusupov et al., 2025).

    Each sample is scored by its mean ERank across all L layers of a
    tester model.  Higher score → more isotropic representations →
    more human-like, diverse text.

    metadata['hidden_states'] : list (N) of lists (L) of np.ndarray
        Each array has shape (n_tokens, d_hidden), extracted from MLP
        blocks (post-activation, pre-residual connection).
    """

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
            pd.DataFrame({
                "sample_id": np.asarray(sample_ids, dtype=np.int64),
                "label":     np.asarray(labels,     dtype=np.int64),
                "score":     scores,
                "rank":      ranks.astype(int),
                "method":    self.name,
            })
            .sort_values("rank", kind="stable")
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Convenience: rank a set of generators
# ---------------------------------------------------------------------------

def rank_generators(
    generator_hidden_states: dict[str, list[list[np.ndarray]]],
) -> pd.DataFrame:
    """Rank generator models by average ERank across their outputs."""
    rows = []
    for name, texts in generator_hidden_states.items():
        text_scores = [average_erank(hs) for hs in texts]
        rows.append({
            "generator":  name,
            "mean_erank": float(np.mean(text_scores)),
            "std_erank":  float(np.std(text_scores)),
            "n_texts":    len(texts),
        })
    df = pd.DataFrame(rows).sort_values("mean_erank", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Benchmark: original vs optimized vs ID estimators
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)

    # --- original SVD path (for comparison) ---------------------------------
    def erank_svd(X: np.ndarray, eps: float = 1e-12) -> float:
        sigma = np.linalg.svd(X.astype(np.float64), compute_uv=False)
        total = sigma.sum()
        if total < eps: return 1.0
        p = sigma / total
        return float(np.exp(-np.sum(p * np.log(np.maximum(p, eps)))))

    # --- MLE ID estimator (Levina & Bickel 2004) ----------------------------
    def id_mle(X: np.ndarray, k: int = 5) -> float:
        X64 = X.astype(np.float64)
        diffs = X64[:, None, :] - X64[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dists_k = np.sort(dists, axis=1)[:, :k]
        r_k = dists_k[:, -1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.log(r_k / dists_k[:, :-1])
        ratios = np.where(np.isfinite(ratios), ratios, 0.0)
        inv_id = ratios.mean(axis=1)
        valid = inv_id > 0
        return float(1.0 / inv_id[valid].mean()) if valid.any() else float("nan")

    # --- CorrInt ID estimator (Grassberger & Procaccia 1983) ----------------
    def id_corrint(X: np.ndarray, n_eps: int = 20) -> float:
        X64 = X.astype(np.float64)
        diffs = X64[:, None, :] - X64[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(-1))
        upper = np.triu(dists, k=1)
        flat = upper[upper > 0]
        if len(flat) < 2: return float("nan")
        eps_vals = np.logspace(
            np.log10(flat.min()), np.log10(flat.max() * 0.9), n_eps
        )
        C = np.array([np.sum(flat <= e) for e in eps_vals], dtype=float)
        C /= len(flat)
        valid = C > 0
        log_e, log_C = np.log(eps_vals[valid]), np.log(C[valid])
        if len(log_e) < 2: return float("nan")
        slope, _ = np.polyfit(log_e, log_C, 1)
        return float(slope)

    def time_fn(fn, X, reps=20):
        for _ in range(3): fn(X)
        t0 = time.perf_counter()
        for _ in range(reps): fn(X)
        return (time.perf_counter() - t0) / reps * 1000

    print("=" * 72)
    print("ERank gram-path vs SVD-path speedup  (n < d regime)")
    print("=" * 72)
    print(f"{'n':>5} {'d':>5}  SVD(ms)  Gram(ms)  speedup  match")
    print("-" * 52)
    for n, d in [(50, 512), (50, 4096), (100, 512), (100, 4096),
                 (256, 512), (256, 4096), (512, 512), (512, 4096)]:
        X = rng.standard_normal((n, d)).astype(np.float64)
        t_svd  = time_fn(erank_svd, X)
        t_gram = time_fn(erank, X)
        match  = abs(erank_svd(X) - erank(X)) < 1e-5
        print(f"{n:>5} {d:>5}  {t_svd:>6.2f}ms  {t_gram:>7.2f}ms  "
              f"{t_svd/t_gram:>5.1f}x   {match}")

    print()
    print("=" * 72)
    print("ERank (gram) vs ID estimators  —  per-layer cost")
    print("=" * 72)
    print(f"{'n':>5} {'d':>5}  ERank(ms)  MLE(ms)  CorrInt(ms)  "
          f"MLE/ERank  CI/ERank")
    print("-" * 68)
    for n, d in [(50, 512), (50, 4096), (100, 512), (100, 4096), (200, 512)]:
        X = rng.standard_normal((n, d)).astype(np.float64)
        t_er  = time_fn(erank, X, reps=30)
        t_mle = time_fn(id_mle, X, reps=10)
        t_ci  = time_fn(id_corrint, X, reps=10)
        print(f"{n:>5} {d:>5}  {t_er:>7.3f}ms  {t_mle:>6.2f}ms  "
              f"{t_ci:>9.2f}ms  {t_mle/t_er:>8.1f}x  {t_ci/t_er:>7.1f}x")

    print()
    print("=" * 72)
    print("Cost breakdown for one text through L=32 layers  (n=100, d=4096)")
    print("=" * 72)
    L, n, d = 32, 100, 4096
    hs = [rng.standard_normal((n, d)).astype(np.float64) for _ in range(L)]
    Gs = [X @ X.T for X in hs]
    all_eigs = np.array([np.linalg.eigvalsh(G) for G in Gs])
    reps = 30

    t0 = time.perf_counter()
    for _ in range(reps):
        [X @ X.T for X in hs]
    t_gram_build = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        [np.linalg.eigvalsh(G) for G in Gs]
    t_eigh = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        sigma = np.sqrt(np.maximum(all_eigs, 0.0))
        totals = np.where(sigma.sum(1, keepdims=True) < 1e-12, 1.0,
                          sigma.sum(1, keepdims=True))
        p = sigma / totals
        p_safe = np.maximum(p, 1e-12)
        np.exp(-np.sum(np.where(p > 1e-12, p * np.log(p_safe), 0.0), axis=1))
    t_entropy = (time.perf_counter() - t0) / reps * 1000

    total = t_gram_build + t_eigh + t_entropy
    print(f"  Gram build  X@X.T × {L}:  {t_gram_build:>7.2f} ms  "
          f"({100*t_gram_build/total:.0f}%)")
    print(f"  eigvalsh    n×n   × {L}:  {t_eigh:>7.2f} ms  "
          f"({100*t_eigh/total:.0f}%)")
    print(f"  Entropy formula (vec):  {t_entropy:>7.2f} ms  "
          f"({100*t_entropy/total:.0f}%)")
    print(f"  ─────────────────────────────────")
    print(f"  Total                   {total:>7.2f} ms")
    print()
    print("  Note: batching all L gram builds into one (L,n,d)@(L,d,n)")
    print("  call gives no speedup — BLAS already saturates threads")
    print("  on individual (n,d)@(d,n) calls at these sizes.")