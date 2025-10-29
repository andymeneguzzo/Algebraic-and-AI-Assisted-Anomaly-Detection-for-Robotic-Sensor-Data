import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse.linalg import svds
from numpy.linalg import norm


def robust_pca(X, lam=None, mu=None, max_iter=1000, tol=1e-7, verbose=False):
    n, d = X.shape
    normX = norm(X, ord='fro')
    lam = lam or 1.0 / np.sqrt(max(n, d))
    mu = mu or (n * d) / (4.0 * norm(X, 1) + 1e-8)

    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    errors = []

    for i in range(max_iter):
        # Low-rank update via SVD thresholding
        U, sigma, Vt = svds(X - S + (1 / mu) * Y, k=min(n, d) - 1)
        sigma_thresh = np.maximum(sigma - 1 / mu, 0)
        L = (U @ np.diag(sigma_thresh) @ Vt)

        # Sparse update via soft thresholding
        S = np.sign(X - L + (1 / mu) * Y) * np.maximum(
            np.abs(X - L + (1 / mu) * Y) - lam / mu, 0
        )

        # Dual update
        Z = X - L - S
        Y += mu * Z

        err = norm(Z, 'fro') / (normX + 1e-8)
        errors.append(err)
        if verbose and i % 50 == 0:
            print(f"Iter {i}, error={err:.6f}")
        if err < tol:
            break

    return L, S, errors


def robust_pca_scores(X):
    L, S, errors = robust_pca(X)
    anomaly_score = np.mean(np.abs(S), axis=1)
    return anomaly_score, L, S, errors


def detect_all(datasets: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, df in datasets.items():
        X = df.select_dtypes(include=np.number).dropna().values
        n, d = X.shape
        print(f"[INFO] Running Robust PCA on {name} ({n}x{d})...")
        scores, L, S, errors = robust_pca_scores(X)
        print(f"[INFO] Converged after {len(errors)} iterations.")
        results[name] = scores

        pd.DataFrame({"robustpca_score": scores}).to_csv(
            output_dir / f"{name}_robustpca.csv", index=False
        )
        print(f"[INFO] Saved → {output_dir}/{name}_robustpca.csv")

    return results
