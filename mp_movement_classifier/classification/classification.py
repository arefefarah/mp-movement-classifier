# Python
from __future__ import annotations

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
from mp_movement_classifier.utils import config

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from mp_movement_classifier.utils.utils import (
    config,
    load_model_with_full_state,
    plot_learn_curve,
    plot_mp,
    plot_reconstructions,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,
    set_figures_directory,
)


# Globals (populated in main)
data_dir: Optional[Path] = None
model_path: Optional[Path] = None


def prepare_weights_for_classification(model, num_segments, num_signals, num_MPs=20):
    """
    Returns:
        X: Feature matrix with shape [num_segments, num_signals * num_MPs]
    """
    X = np.zeros((num_segments, num_signals * num_MPs))

    for seg_idx in range(num_segments):
        for signal_idx in range(num_signals):
            for mp_idx in range(num_MPs):
                feature_idx = signal_idx * num_MPs + mp_idx
                X[seg_idx, feature_idx] = model.weights[seg_idx][signal_idx, mp_idx].item()

    return X

def save_classification_report(report_str: str, out_dir: str, filename: str = "classification_report.txt"):
    path = os.path.join(out_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    return path


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    filename: str = "confusion_matrix.png",
    labels: Optional[List[str]] = None,
) -> Path:
    """
    Plot and save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)

    out_path = Path(out_dir) / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_and_save_feature_importance(
    model,
    feature_names: Optional[List[str]],
    topn: int,
    out_dir: Path,
    filename: str = "feature_importance.png",
) -> Optional[Path]:
    """
    Plot and save top-N feature importances. Supports:
    - feature_importances_ (tree-based models)
    - coef_ (linear models), using mean absolute value across classes if needed
    """
    importances = None

    if hasattr(model, "feature_importances_"):
        try:
            importances = np.asarray(model.feature_importances_)
        except Exception:
            importances = None
    elif hasattr(model, "coef_"):
        try:
            coef = np.asarray(model.coef_)
            if coef.ndim == 1:
                importances = np.abs(coef)
            else:
                # Multi-class: average absolute weights across classes
                importances = np.mean(np.abs(coef), axis=0)
        except Exception:
            importances = None

    if importances is None:
        # Nothing to plot
        return None

    n_features = importances.shape[0]
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"f{i}" for i in range(n_features)]

    idx_sorted = np.argsort(importances)[::-1]
    topn = max(1, min(int(topn), n_features))
    top_idx = idx_sorted[:topn]

    fig, ax = plt.subplots(figsize=(10, max(4, int(topn * 0.3))))
    sns.barplot(
        x=importances[top_idx],
        y=[feature_names[i] for i in top_idx],
        ax=ax,
        orient="h",
        color="#4C72B0",
    )
    ax.set_title(f"Top-{topn} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    out_path = Path(out_dir) / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compare_classifiers(X_scaled, y,out_dir: Path,
    filename: str = "classifier_comparison.png"):
    """
    Compare different classification algorithms
    """
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(random_state=42),
        'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
    }

    results = {}

    for name, clf in classifiers.items():
        # Perform 5-fold cross-validation
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        results[name] = scores
        print(f"{name}: Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

    plt.figure(figsize=(10, 6))
    plt.boxplot(list(results.values()), labels=list(results.keys()))
    plt.title('Classifier Comparison')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    out_path = Path(out_dir) / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[info] Classifier Comparison saved to: {out_path}")

    return results


def main():
    global data_dir, model_path

    num_MPs = 20
    cutoff_freq = 3.0
    model_name = f"mp_model_{num_MPs}_cutoff_{cutoff_freq}"
    model_dir = os.path.join(config.SAVING_DIR, model_name)
    model_file = os.path.join(model_dir, f"mp_model_{num_MPs}_PC_init_cutoff_{cutoff_freq}")
    out_dir = os.path.join(model_dir, "classification")
    model_path = model_file

    folder_path = "../../data/bvh_files"
    bvh_data, motion_ids = read_bvh_files(folder_path)

    # Process data according to paper specifications
    processed_data, segment_motion_ids = process_bvh_data(bvh_data, motion_ids)

    # based on TMP code: the format of data=list(segment_data[signals,time])
    num_segments = len(processed_data)
    num_signals = processed_data[0].shape[0]

    model = load_model_with_full_state(
        model_path,
        num_segments=num_segments,
        num_signals=num_signals
    )

    segment_lengths = np.array([segment.shape[1] for segment in processed_data])
    recon_data = model.predict(segment_lengths, as_numpy=True)

    # feature space
    X= prepare_weights_for_classification(model, num_segments, num_signals, num_MPs)
    y = np.array(segment_motion_ids)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    print(f"Unique motion IDs: {np.unique(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    # Build and save report
    # Use sorted unique labels for stable ordering
    unique_labels = np.unique(y_test)
    report = classification_report(y_test, y_pred, labels=unique_labels)
    report_path = save_classification_report(report, str(out_dir))
    print(f"[info] Classification report saved to: {report_path}")

    # Confusion matrix
    label_names = [str(lbl) for lbl in unique_labels]
    cm_path = plot_and_save_confusion_matrix(y_test, y_pred, out_dir, labels=label_names)
    print(f"[info] Confusion matrix saved to: {cm_path}")

    # Feature importances
    fi_path = plot_and_save_feature_importance(
        model=clf,
        feature_names=None,
        topn=10,
        out_dir=out_dir,
    )
    if fi_path:
        print(f"[info] Feature importance plot saved to: {fi_path}")
    else:
        print("[info] Model does not expose feature importances/coefs; skipping importance plot.")

    # Compare classifiers
    X_all_scaled = scaler.fit_transform(X)
    classifier_results = compare_classifiers(X_all_scaled, y,out_dir=out_dir)


if __name__ == "__main__":
    main()
