import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from scipy.stats import chi2
from pathlib import Path

def mahalanobis_scores(X: np.ndarray) -> np.ndarray:
    emp_cov = EmpiricalCovariance().fit(X)
    return emp_cov.mahalanobis(X)

def detect_all(datasets: dict, p_threshold: float = 0.99) -> dict:
    covariance_scores_dict = {}
    for name, df in datasets.items():
        num_df = df.select_dtypes(include=np.number).dropna()
        X = num_df.values
        n, d = X.shape
        mahal_sq = mahalanobis_scores(X)
        covariance_scores_dict[name] = mahal_sq

        # Chi-square thresholding
        thresh = chi2.ppf(p_threshold, df=d)
        anomaly_mask = mahal_sq > thresh
        ratio = anomaly_mask.mean()
        print(f"[INFO] {name}: anomalies above χ²({p_threshold*100:.0f}%) = {ratio:.2%}")

    return covariance_scores_dict

def save_scores(scores: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, score in scores.items():
        df_out = pd.DataFrame({"mahalanobis_d2": score})
        out_path = output_dir / f"{name}_covariance.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Saved covariance scores → {out_path}")