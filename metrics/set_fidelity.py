"""
Diversity-Calibrated Set Fidelity (DCSF).
MMD-based distributional distance with diversity penalty.
Measures: does the generated set match the real distribution without mode collapse?
"""
import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional


def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel: str = "rbf",
                gamma: Optional[float] = None) -> float:
    """Compute Maximum Mean Discrepancy between two sets of embeddings.

    Args:
        X: generated embeddings [N, D]
        Y: reference embeddings [M, D]
        kernel: 'rbf' or 'linear'
        gamma: RBF kernel bandwidth (auto if None)

    Returns:
        MMD^2 value (lower = more similar distributions)
    """
    if kernel == "linear":
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T
    elif kernel == "rbf":
        if gamma is None:
            # Median heuristic
            all_dists = cdist(np.vstack([X, Y]), np.vstack([X, Y]), metric="sqeuclidean")
            gamma = 1.0 / np.median(all_dists[all_dists > 0])

        K_XX = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
        K_YY = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
        K_XY = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    n = len(X)
    m = len(Y)

    # Unbiased MMD^2 estimate
    mmd2 = (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) \
         + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) \
         - 2 * K_XY.sum() / (n * m)

    return float(max(mmd2, 0.0))  # clamp numerical noise


def compute_diversity(embeddings: np.ndarray) -> float:
    """Compute within-set diversity as mean pairwise distance."""
    if len(embeddings) < 2:
        return 0.0

    dists = cdist(embeddings, embeddings, metric="cosine")
    # Upper triangle (exclude diagonal)
    n = len(embeddings)
    upper_tri = dists[np.triu_indices(n, k=1)]
    return float(upper_tri.mean())


def diversity_calibrated_set_fidelity(
    gen_embeddings: np.ndarray,
    ref_embeddings: np.ndarray,
    alpha: float = 0.5,
    kernel: str = "rbf",
) -> dict:
    """Compute Diversity-Calibrated Set Fidelity.

    DCSF = MMD(gen, ref) + alpha * max(0, div_ref - div_gen)

    The penalty term fires when generated images are less diverse
    than reference images (mode collapse).

    Args:
        gen_embeddings: [N, D] generated image embeddings
        ref_embeddings: [M, D] reference image embeddings
        alpha: weight for diversity penalty
        kernel: MMD kernel type

    Returns:
        Dict with DCSF score and components
    """
    mmd = compute_mmd(gen_embeddings, ref_embeddings, kernel=kernel)
    div_gen = compute_diversity(gen_embeddings)
    div_ref = compute_diversity(ref_embeddings)

    # Diversity gap: penalize when gen is less diverse than ref
    div_gap = max(0.0, div_ref - div_gen)
    dcsf = mmd + alpha * div_gap

    return {
        "dcsf": dcsf,             # lower = better (composite score)
        "mmd": mmd,               # lower = closer distributions
        "diversity_gen": div_gen,  # higher = more diverse generation
        "diversity_ref": div_ref,  # reference diversity
        "diversity_gap": div_gap,  # positive = mode collapse indicator
    }


def main():
    """Quick test with dummy embeddings."""
    np.random.seed(42)

    # Similar distributions
    ref = np.random.randn(10, 128)
    gen_good = ref + np.random.randn(10, 128) * 0.1
    gen_collapsed = np.tile(ref[0], (10, 1)) + np.random.randn(10, 128) * 0.01

    result_good = diversity_calibrated_set_fidelity(gen_good, ref)
    result_bad = diversity_calibrated_set_fidelity(gen_collapsed, ref)

    print("Good generation:")
    print(f"  DCSF: {result_good['dcsf']:.4f}, MMD: {result_good['mmd']:.4f}, "
          f"DivGap: {result_good['diversity_gap']:.4f}")

    print("Mode-collapsed generation:")
    print(f"  DCSF: {result_bad['dcsf']:.4f}, MMD: {result_bad['mmd']:.4f}, "
          f"DivGap: {result_bad['diversity_gap']:.4f}")

    assert result_bad["dcsf"] > result_good["dcsf"], "Mode collapse should score worse"
    print("\nTest passed!")


if __name__ == "__main__":
    main()
