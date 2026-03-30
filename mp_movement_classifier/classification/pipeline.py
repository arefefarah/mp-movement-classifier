from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Reuse existing utilities and plots from the legacy classification module
from mp_movement_classifier.classification.classification_utils import (
    analyze_feature_pca,
    visualize_with_tsne,
    calculate_rdm,
    plot_and_save_feature_importance,
    save_classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def _row_normalized_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix normalized by true-class (rows sum to 1).
    If a row has zero support, it remains zeros (to avoid division by zero).
    """
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    return cm_norm


def _plot_confusion_matrix_percent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    labels: Optional[List[str]] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> Path:
    """Plot confusion matrix with percentage annotations and fixed color range.
    The matrix is row-normalized so each row sums to 1. Values shown as percentages.
    """
    cm_norm = _row_normalized_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    annot = np.where(cm_norm == 0, "", cm_norm.round(2).astype(str))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_classification_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str | Path,
    feature_names: Optional[List[str]] = None,
    feature_structure: Optional[Dict[str, Any]] = None,
    primary_classifier: str = 'linear_svc',
    also_run_random_forest: bool = False,
    fixed_cm_vmin: float = 0.0,
    fixed_cm_vmax: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Unified classification pipeline for TMP, Autoencoder, and Legendre features.

    Parameters
    - X, y: feature matrix and labels
    - out_dir: directory to save artifacts
    - feature_names: optional names for features (used in importance plots)
    - feature_structure: optional metadata for plots (kept for compatibility)
    - primary_classifier: 'linear_svc' or 'random_forest'
    - also_run_random_forest: if True, train RF in addition to the primary
    - fixed_cm_vmin/vmax: fixed color range for confusion matrices [0..1]
    - seed: RNG seed

    Returns: dict with metrics, artifacts paths, and analysis outputs.
    """
    np.random.seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Optional unsupervised analyses (PCA, t-SNE, RDM) for interpretability
    pca_info = analyze_feature_pca(X=X, y=y, out_dir=out_dir, feature_names=feature_names)
    tsne_model = visualize_with_tsne(X=X, y=y, out_dir=out_dir)
    rdm_info = calculate_rdm(X=X, y=y, out_dir=out_dir)

    # 2) Train/test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: Dict[str, Any] = {
        'artifacts': {},
        'models': {},
        'reports': {},
        'pca': pca_info,
        'tsne': tsne_model,
        'rdm': rdm_info,
        'train_indices': None,  # not returned from sklearn directly
        'test_indices': None,
    }

    # 3) Primary classifier
    def _train_linear_svc() -> Dict[str, Any]:
        clf = LinearSVC(C=1.0, penalty='l2', dual=True, random_state=seed)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        rep_str = classification_report(y_test, y_pred, labels=np.unique(y_test))
        report_path = save_classification_report(rep_str, str(out_dir), filename='classification_report_linear_svc.txt')
        # Confusion matrix (% based)
        labels_sorted = [str(lbl) for lbl in sorted(np.unique(y))]
        cm_path = _plot_confusion_matrix_percent(
            y_true=y_test, y_pred=y_pred,
            out_path=out_dir / 'confusion_matrix_linear_svc.png',
            labels=labels_sorted,
            vmin=fixed_cm_vmin, vmax=fixed_cm_vmax,
        )
        # Feature importance (coef_)
        fi_path = plot_and_save_feature_importance(
            model=clf, feature_names=feature_names, topn=min(20, X.shape[1]), out_dir=out_dir,
            filename='feature_importance_linear_svc.png',
        )
        return {
            'model': clf,
            'y_pred': y_pred,
            'report_str': rep_str,
            'report_path': report_path,
            'confusion_matrix_path': cm_path,
            'feature_importance_path': fi_path,
        }

    def _train_random_forest() -> Dict[str, Any]:
        rf = RandomForestClassifier(n_estimators=200, random_state=seed)
        rf.fit(X_train, y_train)  # RF can work on unscaled features
        y_pred = rf.predict(X_test)
        rep_str = classification_report(y_test, y_pred, labels=np.unique(y_test))
        report_path = save_classification_report(rep_str, str(out_dir), filename='classification_report_random_forest.txt')
        labels_sorted = [str(lbl) for lbl in sorted(np.unique(y))]
        cm_path = _plot_confusion_matrix_percent(
            y_true=y_test, y_pred=y_pred,
            out_path=out_dir / 'confusion_matrix_random_forest.png',
            labels=labels_sorted,
            vmin=fixed_cm_vmin, vmax=fixed_cm_vmax,
        )
        fi_path = plot_and_save_feature_importance(
            model=rf, feature_names=feature_names, topn=min(20, X.shape[1]), out_dir=out_dir,
            filename='feature_importance_random_forest.png',
        )
        return {
            'model': rf,
            'y_pred': y_pred,
            'report_str': rep_str,
            'report_path': report_path,
            'confusion_matrix_path': cm_path,
            'feature_importance_path': fi_path,
        }

    if primary_classifier == 'linear_svc':
        primary_res = _train_linear_svc()
        results['models']['linear_svc'] = primary_res['model']
        results['reports']['linear_svc'] = primary_res
    elif primary_classifier == 'random_forest':
        primary_res = _train_random_forest()
        results['models']['random_forest'] = primary_res['model']
        results['reports']['random_forest'] = primary_res
    else:
        raise ValueError(f"Unsupported primary_classifier: {primary_classifier}")

    # 4) Optional secondary model
    if also_run_random_forest and primary_classifier != 'random_forest':
        rf_res = _train_random_forest()
        results['models']['random_forest'] = rf_res['model']
        results['reports']['random_forest'] = rf_res

    return results
