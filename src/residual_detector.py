import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

def residual_scores(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    residual_matrix = np.zeros_like(X)

    for i in range(d):
        X_i = np.delete(X, i, axis=1)
        y_i = X[:, i]
        model = LinearRegression()
        model.fit(X_i, y_i)
        y_pred = model.predict(X_i)
        residual_matrix[:, i] = y_i - y_pred

    return np.mean(np.abs(residual_matrix), axis=1)

def detect_all(datasets: dict) -> dict:
    residual_scores_dict = {}
    for name, df in datasets.items():
        num_df = df.select_dtypes(include=np.number).dropna()
        X = num_df.values
        n, d = X.shape
        print(f"[INFO] Residual-based detection on {name} ({n}x{d})")
        residual_scores_dict[name] = residual_scores(X)
    return residual_scores_dict

def save_scores(scores: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, score in scores.items():
        df_out = pd.DataFrame({"residual_score": score})
        out_path = output_dir / f"{name}_residual.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Saved residual scores → {out_path}")