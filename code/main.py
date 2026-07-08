# main.py

from preprocessing import extract_and_split_dataset
from feature_extraction import extract_features

from mlmodel_training import (
    train_rf_balanced,
    evaluate_rf_once,
    train_svm,
    evaluate_svm_once
)

from ga_feature_selection import (
    run_ga,
    apply_feature_mask
)

import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier


def main(dataset_input, dataset_name, force_cpu=False):

    print("\n[Main] Starting pipeline...")

    os.makedirs("results", exist_ok=True)

    # -------------------------------
    # SPLIT DATASET (always ok to run once)
    # -------------------------------
    paths = extract_and_split_dataset(
        dataset_input,
        'data_processed_' + dataset_name
    )

    # -------------------------------
    # FEATURE FILE PATHS
    # -------------------------------
    train_feat_file = f"results/{dataset_name}_features_train.npz"
    val_feat_file   = f"results/{dataset_name}_features_val.npz"
    test_feat_file  = f"results/{dataset_name}_features_test.npz"

    # -------------------------------
    # FEATURE EXTRACTION (SKIP IF EXISTS)
    # -------------------------------
    print("\n[Step] Feature Extraction")

    if os.path.exists(train_feat_file) and os.path.exists(val_feat_file) and os.path.exists(test_feat_file):
        print("[Feature] Already exists. Skipping extraction.")
    else:
        print("[Feature] Extracting features...")
        extract_features(paths['train'], train_feat_file)
        extract_features(paths['validate'], val_feat_file)
        extract_features(paths['test'], test_feat_file)

    # -------------------------------
    # BASELINE RF
    # -------------------------------
    rf_path = f"results/{dataset_name}_baseline_rf.pkl"

    print("\n[Step] Baseline RF")

    if os.path.exists(rf_path):
        print("[RF] Loading saved model...")
        baseline_rf_model = joblib.load(rf_path)
        baseline_rf_train = None
    else:
        print("[RF] Training model...")
        baseline_rf_model, baseline_rf_train = train_rf_balanced(train_feat_file)
        joblib.dump(baseline_rf_model, rf_path)

    # -------------------------------
    # BASELINE SVM
    # -------------------------------
    svm_path = f"results/{dataset_name}_baseline_svm.pkl"

    print("\n[Step] Baseline SVM")

    if os.path.exists(svm_path):
        print("[SVM] Loading saved model...")
        baseline_svm_model = joblib.load(svm_path)
        baseline_svm_train = None
    else:
        print("[SVM] Training model...")
        baseline_svm_model, baseline_svm_train = train_svm(train_feat_file)
        joblib.dump(baseline_svm_model, svm_path)

    # -------------------------------
    # GA MASK
    # -------------------------------
    ga_path = f"results/{dataset_name}_ga_mask.npz"

    print("\n[Step] GA Feature Selection")

    if os.path.exists(ga_path):
        print("[GA] Loading saved GA mask...")
        best_mask_full = np.load(ga_path)["mask"]
    else:
        print("[GA] Running GA (this will take time)...")

        d_train = np.load(train_feat_file)
        X_train = d_train["X"]
        y_train = d_train["y"]

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)
        importances = rf.feature_importances_

        top_N = 200
        top_idx = np.argsort(importances)[::-1][:top_N]

        X_filtered = X_train[:, top_idx]

        best_mask_small = run_ga(
            X_filtered,
            y_train,
            pop_size=20,
            n_gens=10,
            mutation_rate=0.02,
            crossover_rate=0.8
        )

        best_mask_full = np.zeros(X_train.shape[1], dtype=np.int8)
        best_mask_full[top_idx] = best_mask_small

        np.savez(ga_path, mask=best_mask_full)

    # -------------------------------
    # APPLY MASK
    # -------------------------------
    print("\n[Step] Applying feature mask")

    for split, feat_file in zip(
        ['train', 'val', 'test'],
        [train_feat_file, val_feat_file, test_feat_file]
    ):
        out_file = f"results/{dataset_name}_features_{split}_selected.npz"

        if not os.path.exists(out_file):
            apply_feature_mask(feat_file, out_file, best_mask_full)
        else:
            print(f"[Mask] Already exists: {out_file}")

    # -------------------------------
    # SELECTED RF
    # -------------------------------
    sel_rf_path = f"results/{dataset_name}_selected_rf.pkl"

    print("\n[Step] Selected RF")

    if os.path.exists(sel_rf_path):
        print("[RF-SEL] Loading saved model...")
        selected_rf_model = joblib.load(sel_rf_path)
    else:
        selected_rf_model, _ = train_rf_balanced(
            f"results/{dataset_name}_features_train_selected.npz"
        )
        joblib.dump(selected_rf_model, sel_rf_path)

    # -------------------------------
    # SELECTED SVM
    # -------------------------------
    sel_svm_path = f"results/{dataset_name}_selected_svm.pkl"

    print("\n[Step] Selected SVM")

    if os.path.exists(sel_svm_path):
        print("[SVM-SEL] Loading saved model...")
        selected_svm_model = joblib.load(sel_svm_path)
    else:
        selected_svm_model, _ = train_svm(
            f"results/{dataset_name}_features_train_selected.npz"
        )
        joblib.dump(selected_svm_model, sel_svm_path)

    print("\n[Main] PIPELINE COMPLETED SUCCESSFULLY ✅")


if __name__ == "__main__":

    if len(sys.argv) == 2:
        dataset_name = input("Enter dataset name: ").strip()
        main(sys.argv[1], dataset_name)
    else:
        print("Usage: python main.py <dataset_path>")