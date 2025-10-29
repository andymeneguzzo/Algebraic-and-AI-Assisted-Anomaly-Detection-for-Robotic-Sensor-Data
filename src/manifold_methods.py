import numpy as np
import pandas as pd
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def pseudo_inverse_isomap(iso, X_embedded, X_train, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(iso.embedding_)
    distances, indices = nn.kneighbors(X_embedded)

    X_recon = np.zeros_like(X_train)
    for i in range(len(X_embedded)):
        neigh_idx = indices[i, 1:]
        neigh_dist = distances[i, 1:]
        weights = np.exp(-neigh_dist / (np.mean(neigh_dist) + 1e-8))
        weights /= weights.sum()
        X_recon[i] = np.sum(X_train[neigh_idx] * weights[:, None], axis=0)
    return X_recon


def manifold_reconstruction_error(X, n_neighbors=10, n_components=2):
    iso = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_emb = iso.fit_transform(X)
    X_rec = pseudo_inverse_isomap(iso, X_emb, X)
    rec_error = np.mean((X - X_rec) ** 2, axis=1)
    return rec_error, X_emb, X_rec


def detect_all(datasets: dict, output_dir: str | Path, n_neighbors=10, n_components=2):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, df in datasets.items():
        X = df.select_dtypes(include=np.number).dropna().values
        n, d = X.shape
        print(f"[INFO] Running Isomap manifold detection on {name} ({n}x{d})...")
        try:
            score, X_emb, X_rec = manifold_reconstruction_error(X, n_neighbors, n_components)
            results[name] = score
            pd.DataFrame({"manifold_score": score}).to_csv(
                output_dir / f"{name}_manifold.csv", index=False
            )
            print(f"[INFO] Saved → {output_dir}/{name}_manifold.csv")
        except Exception as e:
            print(f"[WARNING] Isomap failed on {name}: {e}")
            results[name] = np.full(n, np.nan)

    return results
