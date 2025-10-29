import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 42
np.random.seed(SEED)

"""PCA"""

def choose_pca_components(X, var_ratio=0.95):
    pca = PCA(n_components=min(X.shape)).fit(X)
    csum = np.cumsum(pca.explained_variance_ratio_)
    n = int(np.searchsorted(csum, var_ratio) + 1)
    return max(1, min(n, X.shape[1]))

def pca_reconstruction_scores(X, var_ratio=0.95):
    n_components = choose_pca_components(X, var_ratio=var_ratio)
    pca = PCA(n_components=n_components, random_state=SEED)
    Z = pca.fit_transform(X)
    X_hat = pca.inverse_transform(Z)

    q_residual = np.mean((X - X_hat)**2, axis=1)

    eigvals = pca.explained_variance_
    eigvals_inv = np.where(eigvals > 1e-12, 1.0 / eigvals, 0.0)
    T2 = np.sum((Z**2) * eigvals_inv, axis=1)

    return q_residual, T2, pca


"""Kernel PCA"""

def median_pairwise_sigma(X, max_samples=2000):
    from scipy.spatial.distance import pdist
    if X.shape[0] > max_samples:
        idx = np.random.RandomState(SEED).choice(X.shape[0], max_samples, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    dists = pdist(Xs, metric="euclidean")
    median = np.median(dists)
    return median if median > 0 else 1.0

def kpca_reconstruction_scores(X, var_ratio=0.95, gamma=None):
    n_components = choose_pca_components(X, var_ratio)
    n_components = max(2, min(n_components, min(50, X.shape[1])))

    if gamma is None:
        sigma = median_pairwise_sigma(X)
        gamma = 1.0 / (2.0 * (sigma**2))

    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        fit_inverse_transform=True,
        n_jobs=-1,
        random_state=SEED
    )
    Z = kpca.fit_transform(X)
    try:
        X_hat = kpca.inverse_transform(Z)
        kpca_rec = np.mean((X - X_hat)**2, axis=1)
    except Exception:
        # Fallback: latent energy as monotonic proxy
        kpca_rec = np.sum(Z**2, axis=1)
        print("[WARNING] KPCA inverse transform failed — using latent energy proxy.")
    return kpca_rec, kpca


"""Autoencoder"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except Exception:
    HAS_TF = False
    print("[WARNING] TensorFlow not found — Autoencoder will be skipped.")

def build_autoencoder(input_dim, bottleneck=None, dropout=0.0):
    if not HAS_TF:
        raise ImportError("TensorFlow is not installed.")
    if bottleneck is None:
        bottleneck = max(2, min(16, input_dim // 2))
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(max(64, input_dim*2), activation="relu")(inp)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(max(32, input_dim), activation="relu")(x)
    z = layers.Dense(bottleneck, activation="linear", name="bottleneck")(x)
    x = layers.Dense(max(32, input_dim), activation="relu")(z)
    x = layers.Dense(max(64, input_dim*2), activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    ae = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae

def autoencoder_reconstruction_scores(X, epochs=200, batch_size=None, bottleneck=None):
    if not HAS_TF:
        return np.full(X.shape[0], np.nan)

    n, d = X.shape
    if batch_size is None:
        batch_size = min(256, max(32, n // 20))

    ae = build_autoencoder(d, bottleneck=bottleneck)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=SEED, shuffle=True)

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    ae.fit(X_train, X_train, validation_data=(X_val, X_val),
           epochs=epochs, batch_size=batch_size, verbose=True, callbacks=callbacks)

    X_hat = ae.predict(X, verbose=0)
    ae_rec = np.mean((X - X_hat)**2, axis=1)
    return ae_rec


"""Batch detection pipeline"""

def detect_all(datasets: dict, output_dir: str | Path, var_ratio=0.95):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, df in datasets.items():
        num_df = df.select_dtypes(include=np.number).dropna()
        X = num_df.values
        n, d = X.shape
        print(f"[INFO] Running PCA/KPCA/AE detection on {name} ({n}x{d})")

        # PCA
        pca_q, pca_t2, pca_model = pca_reconstruction_scores(X, var_ratio)
        # KPCA
        kpca_rec, kpca_model = kpca_reconstruction_scores(X, var_ratio)
        # Autoencoder
        ae_rec = autoencoder_reconstruction_scores(X, bottleneck=max(2, min(16, d // 2)))

        scores_df = pd.DataFrame({
            "pca_q_residual": pca_q,
            "pca_hotelling_t2": pca_t2,
            "kpca_recon": kpca_rec,
            "ae_recon": ae_rec
        })
        out_csv = output_dir / f"{name}_pca_kpca_ae.csv"
        scores_df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved → {out_csv}")

        all_results[name] = scores_df

    return all_results