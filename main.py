
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src import preprocessing, residual_detector, covariance_detector, pca_based_methods, robust_pca, manifold_methods, evaluation_report

def run_preprocessing():
    data_dir = Path("./data")
    clean_dir = data_dir / "cleaned"
    norm_dir = data_dir / "normalized"

    # 1. Load
    datasets = preprocessing.load_datasets(data_dir)

    # 2. Summarize
    preprocessing.summarize_datasets(datasets)

    # 3. Clean names
    cleaned = preprocessing.clean_columns(datasets)

    # 4. Save cleaned
    preprocessing.save_cleaned(cleaned, clean_dir)

    # 5. Check outliers
    preprocessing.check_outliers(cleaned)

    # 6. Normalize and save
    preprocessing.normalize_datasets(cleaned, norm_dir)

def run_residual_covariance_detection():
    norm_dir = Path("./data/normalized")
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    # Load normalized data
    datasets = {f.stem.replace("_norm", ""): pd.read_csv(f) for f in norm_dir.glob("*.csv")}
    print(f"[INFO] Loaded {len(datasets)} normalized datasets for detection.")

    # Residual-based
    residual_scores = residual_detector.detect_all(datasets)
    residual_detector.save_scores(residual_scores, result_dir)

    # Covariance-based
    covariance_scores = covariance_detector.detect_all(datasets)
    covariance_detector.save_scores(covariance_scores, result_dir)

    # Merge both into single CSVs
    merge_residual_covariance_results(result_dir, residual_scores, covariance_scores)

def merge_residual_covariance_results(result_dir: Path, residual_scores: dict, covariance_scores: dict):
    for name in residual_scores.keys():
        df_out = pd.DataFrame({
            "residual_score": residual_scores[name],
            "mahalanobis_d2": covariance_scores[name]
        })
        out_path = result_dir / f"{name}_residual_cov.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Saved merged results → {out_path}")

def run_pca_kpca_autoencoder_detection():
    norm_dir = Path("./data/normalized")
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    datasets = {f.stem.replace("_norm", ""): pd.read_csv(f) for f in norm_dir.glob("*.csv")}
    print(f"[INFO] Loaded {len(datasets)} normalized datasets for PCA/KPCA/AE detection.")

    results = pca_based_methods.detect_all(datasets, output_dir=result_dir)
    print(f"[INFO] PCA/KPCA/AE detection completed on {len(results)} datasets.")

def run_robustpca_manifold_detection():
    norm_dir = Path("./data/normalized")
    result_dir = Path("./results")
    result_dir.mkdir(exist_ok=True)

    datasets = {f.stem.replace("_norm", ""): pd.read_csv(f) for f in norm_dir.glob("*.csv")}
    print(f"[INFO] Loaded {len(datasets)} normalized datasets for Robust PCA + Manifold detection.")

    robust_scores = robust_pca.detect_all(datasets, output_dir=result_dir)
    manifold_scores = manifold_methods.detect_all(datasets, output_dir=result_dir)

    # Merge into joint files
    for name in datasets.keys():
        df_out = pd.DataFrame({
            "robustpca_score": robust_scores[name],
            "manifold_score": manifold_scores[name]
        })
        out_path = result_dir / f"{name}_robustpca_manifold.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] Saved → {out_path}")

    print("[INFO] Robust PCA + Manifold detection completed.")

def run_evaluation_report():
    evaluation_report.run_evaluation(results_dir="./results", ai_report_dir="./ai_report")



def main():
    parser = argparse.ArgumentParser(description="Algebraic Anomaly Detection Pipeline")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["preprocess", "detect_res_cov", "detect_pca_autoenc", "detect_robust_manifold", "evaluate_report"], help="Pipeline stage to execute.")
    args = parser.parse_args()

    if args.stage == "preprocess":
        run_preprocessing()
    elif args.stage == "detect_res_cov":
        run_residual_covariance_detection()
    elif args.stage == "detect_pca_autoenc":
        run_pca_kpca_autoencoder_detection()
    elif args.stage == "detect_robust_manifold":
        run_robustpca_manifold_detection()
    elif args.stage == "evaluate_report":
        run_evaluation_report()
    else:
        print("Unknown stage. Available: preprocess, detect, detect_pca_autoenc, detect_robust_manifold, evaluate_report")


if __name__ == "__main__":
    main()
