from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Reuse existing utilities and plots from the legacy classification module
from mp_movement_classifier.classification.utils import (
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


# def _plot_confusion_matrix_percent(
#         y_true: np.ndarray,
#         y_pred: np.ndarray,
#         out_path: Path,
#         labels: Optional[List[str]] = None,
#         vmin: float = 0.0,
#         vmax: float = 1.0,
# ) -> Path:
#     """Plot confusion matrix with percentage annotations and fixed color range.
#     The matrix is row-normalized so each row sums to 1. Values shown as percentages.
#     """
#     cm_norm = _row_normalized_confusion_matrix(y_true, y_pred)
#
#     fig, ax = plt.subplots(figsize=(9, 7))
#     annot = np.where(cm_norm == 0, "", cm_norm.round(2).astype(str))
#     sns.heatmap(
#         cm_norm,
#         annot=annot,
#         fmt="",
#         cmap="Blues",
#         cbar=True,
#         vmin=vmin,
#         vmax=vmax,
#         ax=ax,
#     )
#
#     ax.set_title("Confusion Matrix")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#
#     if labels is not None:
#         ax.set_xticklabels(labels, rotation=45, ha="right")
#         ax.set_yticklabels(labels, rotation=0)
#
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
#     plt.close(fig)
#     return out_path

def _plot_confusion_matrix_percent(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        out_path: Path,
        labels: Optional[List[str]] = None,
        vmin: float = 0.0,
        vmax: float = 1.0,
) -> Path:
    cm_norm = _row_normalized_confusion_matrix(y_true, y_pred)

    # ── 0.  Central font-size config ────────────────────────────────────────
    FONT = dict(
        annot   = 14,   # numbers inside each cell
        tick    = 13,   # class names on both axes
        label   = 14,   # "Predicted" / "True"
        title   = 16,   # "Confusion Matrix"
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    annot = np.where(cm_norm == 0, "", cm_norm.round(2).astype(str))

    # ── 1.  Cell annotation font ─────────────────────────────────────────────
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        annot_kws={"size": FONT["annot"]},   # ← cell values
    )

    # ── 2.  Tick label font ───────────────────────────────────────────────────
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT["tick"])
        ax.set_yticklabels(labels, rotation=0,              fontsize=FONT["tick"])
    else:
        # no explicit labels — still resize whatever seaborn put there
        ax.tick_params(axis="both", labelsize=FONT["tick"])

    # ── 3.  Axis label & title font ───────────────────────────────────────────
    ax.set_title("Confusion Matrix",  fontsize=FONT["title"])
    ax.set_xlabel("Predicted",        fontsize=FONT["label"])
    ax.set_ylabel("True",             fontsize=FONT["label"])
    ax.collections[0].colorbar.ax.tick_params(labelsize=FONT["tick"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def _perform_svc_cross_validation(
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        cv_folds: int = 5
) -> Dict[str, Any]:

    # Initialize the SVC classifier
    clf = LinearSVC(C=1.0, penalty='l2', dual=True, random_state=seed)

    # Scale features for SVC
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified K-Fold to maintain class distribution
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Perform cross-validation with multiple metrics
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(clf, X_scaled, y, cv=cv, scoring=scoring,
                                return_train_score=True, n_jobs=-1)

    # Calculate statistics
    cv_stats = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']

        cv_stats[f'{metric}_test_mean'] = np.mean(test_scores)
        cv_stats[f'{metric}_test_std'] = np.std(test_scores)
        cv_stats[f'{metric}_train_mean'] = np.mean(train_scores)
        cv_stats[f'{metric}_train_std'] = np.std(train_scores)
        cv_stats[f'{metric}_test_scores'] = test_scores
        cv_stats[f'{metric}_train_scores'] = train_scores

        # Calculate generalization gap (train - test)
        cv_stats[f'{metric}_generalization_gap'] = np.mean(train_scores) - np.mean(test_scores)

    return cv_stats


def _save_cross_validation_results(cv_results: Dict[str, Any], out_path: Path) -> Path:
    """Save cross-validation results to a text file."""
    with open(out_path, 'w') as f:
        f.write("Linear SVC Cross-Validation Results\n")
        f.write("=" * 40 + "\n\n")

        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for metric in metrics:
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Test:  {cv_results[f'{metric}_test_mean']:.4f} ± {cv_results[f'{metric}_test_std']:.4f}\n")
            f.write(f"  Train: {cv_results[f'{metric}_train_mean']:.4f} ± {cv_results[f'{metric}_train_std']:.4f}\n")
            f.write(f"  Generalization Gap: {cv_results[f'{metric}_generalization_gap']:.4f}\n")
            f.write(f"  Test scores: {[f'{score:.4f}' for score in cv_results[f'{metric}_test_scores']]}\n\n")

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
        cv_folds: int = 5,
        perform_cv: bool = True,
) -> Dict[str, Any]:
    """

    Parameters
    - X, y: feature matrix and labels
    - out_dir: directory to save artifacts
    - feature_names: optional names for features (used in importance plots)
    - feature_structure: optional metadata for plots (kept for compatibility)
    - primary_classifier: 'linear_svc' or 'random_forest'
    - also_run_random_forest: if True, train RF in addition to the primary
    - fixed_cm_vmin/vmax: fixed color range for confusion matrices [0..1]
    - seed: RNG seed
    - cv_folds: number of folds for cross-validation (default: 5)
    - perform_cv: whether to perform cross-validation for SVC (default: True)

    Returns: dict with metrics, artifacts paths, analysis outputs, and CV results.
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
        'cross_validation': {},
        'pca': pca_info,
        'tsne': tsne_model,
        'rdm': rdm_info,
        'train_indices': None,  # not returned from sklearn directly
        'test_indices': None,
    }

    # 3) Cross-validation for SVC only (performed on the entire dataset)
    if perform_cv:
        print(f"Performing {cv_folds}-fold cross-validation for LinearSVC...")

        cv_results = _perform_svc_cross_validation(X, y, seed, cv_folds)
        results['cross_validation']['linear_svc'] = cv_results

        # Save CV results
        cv_report_path = _save_cross_validation_results(
            cv_results,
            out_dir / 'cross_validation_linear_svc.txt'
        )
        results['cross_validation']['linear_svc_report_path'] = cv_report_path

        # Print CV summary
        print(f"LinearSVC Cross-Validation Results:")
        print(f"Accuracy: {cv_results['accuracy_test_mean']:.4f} ± {cv_results['accuracy_test_std']:.4f}")
        print(f"F1-Score: {cv_results['f1_macro_test_mean']:.4f} ± {cv_results['f1_macro_test_std']:.4f}")
        print(f"Generalization Gap (Accuracy): {cv_results['accuracy_generalization_gap']:.4f}")

    # 4) Primary classifier (traditional train/test evaluation)
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
        report_path = save_classification_report(rep_str, str(out_dir),
                                                 filename='classification_report_random_forest.txt')
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

    # 5) Optional secondary model
    if also_run_random_forest and primary_classifier != 'random_forest':
        rf_res = _train_random_forest()
        results['models']['random_forest'] = rf_res['model']
        results['reports']['random_forest'] = rf_res

    return results