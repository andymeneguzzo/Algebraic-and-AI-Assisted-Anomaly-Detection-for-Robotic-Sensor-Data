import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def load_datasets(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    dataframes = {}
    for f in files:
        df = pd.read_csv(f)
        key = f.stem.replace("_data", "")
        dataframes[key] = df
        print(f"[INFO] Loaded {key}: {df.shape[0]} samples x {df.shape[1]} features")
    return dataframes

def summarize_datasets(datasets: dict) -> pd.DataFrame:
    summary = []
    for name, df in datasets.items():
        summary.append({
            "Dataset": name,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Nan %": df.isna().mean().mean() * 100,
            "Numeric cols": df.select_dtypes(include=np.number).shape[1],
            "Categorical cols": df.select_dtypes(exclude=np.number).shape[1],
        })
    summary_df = pd.DataFrame(summary)
    print(summary_df)
    return summary_df

def clean_columns(datasets: dict) -> dict:
    cleaned = {}
    for name, df in datasets.items():
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        cleaned[name] = df
    print("[INFO] Column names standardized.")
    return cleaned

def save_cleaned(datasets: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        out_path = output_dir / f"{name}_clean.csv"
        df.to_csv(out_path, index=False)
        print(f"[INFO] Saved cleaned: {out_path}")
    print("[INFO] All cleaned datasets saved.")

def normalize_datasets(datasets: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalers = {}
    for name, df in datasets.items():
        num_df = df.select_dtypes(include=np.number)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(num_df)
        normalized_df = pd.DataFrame(scaled_data, columns=num_df.columns)
        out_path = output_dir / f"{name}_norm.csv"
        normalized_df.to_csv(out_path, index=False)
        scalers[name] = scaler
        print(f"[INFO] {name} normalized → {out_path}")
    print("[INFO] All datasets normalized and saved.")
    return scalers

def outlier_ratio(df: pd.DataFrame) -> float:
    from scipy.stats import zscore
    num_df = df.select_dtypes(include=np.number)
    z = np.abs(zscore(num_df, nan_policy='omit'))
    return float((z > 3).mean().mean())


def check_outliers(datasets: dict):
    for name, df in datasets.items():
        ratio = outlier_ratio(df)
        print(f"[INFO] {name} — Outlier ratio (|z| > 3): {ratio:.2%}")