from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
        annot=14,  # numbers inside each cell
        tick=13,  # class names on both axes
        label=14,  # "Predicted" / "True"
        title=16,  # "Confusion Matrix"
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
        annot_kws={"size": FONT["annot"]},  # ← cell values
    )

    # ── 2.  Tick label font ───────────────────────────────────────────────────
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT["tick"])
        ax.set_yticklabels(labels, rotation=0, fontsize=FONT["tick"])
    else:
        # no explicit labels — still resize whatever seaborn put there
        ax.tick_params(axis="both", labelsize=FONT["tick"])

    # ── 3.  Axis label & title font ───────────────────────────────────────────
    ax.set_title("Confusion Matrix", fontsize=FONT["title"])
    ax.set_xlabel("Predicted", fontsize=FONT["label"])
    ax.set_ylabel("True", fontsize=FONT["label"])
    ax.collections[0].colorbar.ax.tick_params(labelsize=FONT["tick"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out_path


def _perform_cross_validation(
        X: np.ndarray,
        y: np.ndarray,
        classifier,
        classifier_name: str,
        seed: int = 42,
        cv_folds: int = 5,
        scale_data: bool = True
) -> Dict[str, Any]:
    """
    Generalized cross-validation function that works with any classifier.
    Replaces the original _perform_svc_cross_validation to work with all classifiers.

    Args:
        X: Feature matrix
        y: Target labels
        classifier: Sklearn classifier instance
        classifier_name: Name for reporting
        seed: Random seed
        cv_folds: Number of CV folds
        scale_data: Whether to scale the data
    """
    # Scale features if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()

    # Stratified K-Fold to maintain class distribution
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Perform cross-validation with multiple metrics
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(classifier, X_scaled, y, cv=cv, scoring=scoring,
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

    cv_stats['classifier_name'] = classifier_name
    return cv_stats


# Keep the original function name for backward compatibility, but make it call the generalized version
def _perform_svc_cross_validation(
        X: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Original SVC cross-validation function, now calls the generalized version.
    Kept for backward compatibility.
    """
    clf = LinearSVC(C=1.0, penalty='l2', dual=True, random_state=seed)
    return _perform_cross_validation(X, y, clf, "Linear SVC", seed, cv_folds, scale_data=True)


def _save_cross_validation_results(cv_results: Dict[str, Any], out_path: Path, classifier_name: str = None) -> Path:
    """Save cross-validation results to a text file."""
    # Use classifier_name from cv_results if not provided
    if classifier_name is None:
        classifier_name = cv_results.get('classifier_name', 'Classifier')

    with open(out_path, 'w') as f:
        f.write(f"{classifier_name} Cross-Validation Results\n")
        f.write("=" * 40 + "\n\n")

        metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        for metric in metrics:
            f.write(f"{metric.upper()}:\n")
            f.write(f"  Test:  {cv_results[f'{metric}_test_mean']:.4f} ± {cv_results[f'{metric}_test_std']:.4f}\n")
            f.write(f"  Train: {cv_results[f'{metric}_train_mean']:.4f} ± {cv_results[f'{metric}_train_std']:.4f}\n")
            f.write(f"  Generalization Gap: {cv_results[f'{metric}_generalization_gap']:.4f}\n")
            f.write(f"  Test scores: {[f'{score:.4f}' for score in cv_results[f'{metric}_test_scores']]}\n\n")

    return out_path

from matplotlib.ticker import FormatStrFormatter


def _plot_classifier_comparison(cv_results: Dict[str, Dict[str, Any]], out_dir: Path) -> Path:
    """
    Create a comparison boxplot of accuracy across all classifiers focusing on upper values.
    """
    classifiers = list(cv_results.keys())

    name_mapping = {
        'linear_svc': 'Linear SVC',
        'random_forest': 'Random Forest',
        'logistic_regression': 'Logistic Regression',
        'mlp': 'MLP'
    }
    display_names = [name_mapping.get(clf, clf) for clf in classifiers]

    # Extract all CV fold scores
    all_scores = [cv_results[clf]['accuracy_test_scores'] for clf in classifiers]
    all_flat_scores = [s for scores in all_scores for s in scores]
    max_score = max(all_flat_scores)

    y_min = 0.8
    y_max = 1.0

    # --- COMPACTNESS CHANGE 1: smaller figure ---
    fig, ax = plt.subplots(figsize=(4.8, 4.0))

    # --- COMPACTNESS CHANGE 2: tighter box positions and narrower widths ---
    n = len(classifiers)
    positions = np.arange(1, n + 1) * 0.7   # compress horizontal spacing
    bp = ax.boxplot(all_scores, positions=positions, widths=0.35,
                    labels=display_names, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.8),
                    meanprops=dict(color='black', linewidth=2, linestyle='--'),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:n]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # --- COMPACTNESS CHANGE 3: tighter annotations (smaller bbox pad, smaller offset) ---
    for i, clf in enumerate(classifiers):
        mean_acc = cv_results[clf]['accuracy_test_mean']
        std_acc = cv_results[clf]['accuracy_test_std']
        ax.text(positions[i], max_score + (y_max - y_min) * 0.005,
                f'{mean_acc:.3f}±{std_acc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classifier', fontsize=14, fontweight='bold')
    ax.set_title('Cross-Validation Accuracy Comparison', fontsize=16, fontweight='bold')

    # --- COMPACTNESS CHANGE 4: tight x-lim, less headroom above y_max ---
    ax.set_xlim(positions[0] - 0.35, positions[-1] + 0.35)
    ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.02)

    # --- Y-AXIS FORMAT: 2 decimals (1.00 instead of 1.000) ---
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    out_path = out_dir / 'classifier_accuracy_comparison.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

    return out_path
#
# def _plot_classifier_comparison(cv_results: Dict[str, Dict[str, Any]], out_dir: Path) -> Path:
#     """
#     Create a comparison boxplot of accuracy across all classifiers focusing on upper values.
#     """
#     classifiers = list(cv_results.keys())
#
#     # Create a more readable classifier name mapping
#     name_mapping = {
#         'linear_svc': 'Linear SVC',
#         'random_forest': 'Random Forest',
#         'logistic_regression': 'Logistic Regression',
#         'mlp': 'MLP'
#     }
#
#     display_names = [name_mapping.get(clf, clf) for clf in classifiers]
#
#     # Extract all CV fold scores for each classifier
#     all_scores = []
#     for clf in classifiers:
#         scores = cv_results[clf]['accuracy_test_scores']
#         all_scores.append(scores)
#
#     # Calculate dynamic y-axis range focusing on upper values
#     all_flat_scores = [score for scores in all_scores for score in scores]
#     min_score = min(all_flat_scores)
#     max_score = max(all_flat_scores)
#
#     # Set lower bound to be around 0.7 or 10% below the minimum score, whichever is higher
#     y_min = 0.8 #max(0.7, min_score - 0.05)
#     y_max = 1 #min(1.0, max_score + 0.02)
#
#     fig, ax = plt.subplots(figsize=(6, 5))
#
#     # Create boxplot
#     bp = ax.boxplot(all_scores, labels=display_names, patch_artist=True,
#                     showmeans=True, meanline=True,
#                     boxprops=dict(facecolor='lightblue', alpha=0.8),
#                     meanprops=dict(color='black', linewidth=2, linestyle='--'),
#                     whiskerprops=dict(color='black', linewidth=1.5),
#                     capprops=dict(color='black', linewidth=1.5),
#                     flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.7))
#
#     # Color the boxes differently
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(classifiers)]
#     for patch, color in zip(bp['boxes'], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.7)
#
#     # Add mean value annotations above each box
#     for i, clf in enumerate(classifiers):
#         mean_acc = cv_results[clf]['accuracy_test_mean']
#         std_acc = cv_results[clf]['accuracy_test_std']
#         ax.text(i + 1, max_score + (y_max - y_min) * 0.01,
#                 f'{mean_acc:.3f}±{std_acc:.3f}',
#                 ha='center', va='bottom', fontsize=11, fontweight='bold',
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
#
#     ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
#     ax.set_xlabel('Classifier', fontsize=14, fontweight='bold')
#     ax.set_title('Cross-Validation Accuracy Comparison', fontsize=16, fontweight='bold')
#
#     # Set focused y-axis range
#     ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.05)
#
#     # Add grid for better readability
#     ax.grid(True, alpha=0.3, axis='y', linestyle='--')
#     ax.set_axisbelow(True)
#
#     # Rotate x-axis labels if needed
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#
#     # Save the plot
#     out_path = out_dir / 'classifier_accuracy_comparison.png'
#     fig.savefig(out_path, dpi=150, bbox_inches='tight')
#     fig.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
#     plt.close(fig)
#
#     return out_path


def run_classification_pipeline(
        X: np.ndarray,
        y: np.ndarray,
        out_dir: str | Path,
        feature_names: Optional[List[str]] = None,
        feature_structure: Optional[Dict[str, Any]] = None,
        primary_classifier: str = 'random_forest',
        run_all_classifiers: bool = False,
        fixed_cm_vmin: float = 0.0,
        fixed_cm_vmax: float = 1.0,
        seed: int = 42,
        cv_folds: int = 5,
        perform_cv: bool = True,
) -> Dict[str, Any]:
    """
    Enhanced classification pipeline with support for multiple classifiers.

    Parameters
    - X, y: feature matrix and labels
    - out_dir: directory to save artifacts
    - feature_names: optional names for features (used in importance plots)
    - feature_structure: optional metadata for plots (kept for compatibility)
    - primary_classifier: 'linear_svc', 'random_forest', 'logistic_regression', or 'mlp'
    - run_all_classifiers: if True, run all four classifiers and generate comparison
    - fixed_cm_vmin/vmax: fixed color range for confusion matrices [0..1]
    - seed: RNG seed
    - cv_folds: number of folds for cross-validation (default: 5)
    - perform_cv: whether to perform cross-validation (default: True)

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

    # Define all available classifiers
    def _get_classifier_config():
        return {
            'linear_svc': {
                'model': LinearSVC(C=1.0, penalty='l2', dual=True, random_state=seed),
                'scale_data': True,
                'use_scaled_train': True,
                'name': 'Linear SVC'
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=200, random_state=seed),
                'scale_data': False,  # RF doesn't need scaling
                'use_scaled_train': False,
                'name': 'Random Forest'
            },
            'logistic_regression': {
                'model': LogisticRegression(C=1.0, random_state=seed, max_iter=1000),
                'scale_data': True,
                'use_scaled_train': True,
                'name': 'Logistic Regression'
            },
            'mlp': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=seed,
                                       max_iter=500, early_stopping=True, validation_fraction=0.1),
                'scale_data': True,
                'use_scaled_train': True,
                'name': 'MLP'
            }
        }

    classifier_configs = _get_classifier_config()

    # Determine which classifiers to run
    if run_all_classifiers:
        classifiers_to_run = list(classifier_configs.keys())
    else:
        classifiers_to_run = [primary_classifier]

    # 3) Cross-validation for all selected classifiers
    if perform_cv:
        print(f"Performing {cv_folds}-fold cross-validation for {len(classifiers_to_run)} classifier(s)...")

        for clf_name in classifiers_to_run:
            if clf_name not in classifier_configs:
                raise ValueError(f"Unknown classifier: {clf_name}")

            config = classifier_configs[clf_name]
            print(f"Running CV for {config['name']}...")

            # Use the generalized cross-validation function
            cv_results = _perform_cross_validation(
                X, y, config['model'], config['name'], seed, cv_folds, config['scale_data']
            )
            results['cross_validation'][clf_name] = cv_results

            # Save CV results - updated function signature
            cv_report_path = _save_cross_validation_results(
                cv_results,
                out_dir / f'cross_validation_{clf_name}.txt',
                config['name']
            )
            results['cross_validation'][f'{clf_name}_report_path'] = cv_report_path

            # Print CV summary
            print(f"{config['name']} Cross-Validation Results:")
            print(f"Accuracy: {cv_results['accuracy_test_mean']:.4f} ± {cv_results['accuracy_test_std']:.4f}")
            print(f"F1-Score: {cv_results['f1_macro_test_mean']:.4f} ± {cv_results['f1_macro_test_std']:.4f}")
            print(f"Generalization Gap (Accuracy): {cv_results['accuracy_generalization_gap']:.4f}")
            print("-" * 50)

        # Create comparison plot if multiple classifiers
        if len(classifiers_to_run) > 1:
            cv_comparison_data = {k: v for k, v in results['cross_validation'].items()
                                  if not k.endswith('_report_path')}
            comparison_plot_path = _plot_classifier_comparison(cv_comparison_data, out_dir)
            results['cross_validation']['comparison_plot_path'] = comparison_plot_path

    # 4) Train/test evaluation for all selected classifiers
    def _train_classifier(clf_name: str) -> Dict[str, Any]:
        config = classifier_configs[clf_name]
        clf = config['model']

        # Choose appropriate training data
        if config['use_scaled_train']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test

        # Train the model
        clf.fit(X_train_use, y_train)
        y_pred = clf.predict(X_test_use)

        # Generate reports and plots
        rep_str = classification_report(y_test, y_pred, labels=np.unique(y_test))
        report_path = save_classification_report(rep_str, str(out_dir),
                                                 filename=f'classification_report_{clf_name}.txt')

        # Confusion matrix
        labels_sorted = [str(lbl) for lbl in sorted(np.unique(y))]
        cm_path = _plot_confusion_matrix_percent(
            y_true=y_test, y_pred=y_pred,
            out_path=out_dir / f'confusion_matrix_{clf_name}.png',
            labels=labels_sorted,
            vmin=fixed_cm_vmin, vmax=fixed_cm_vmax,
        )

        # Feature importance (if available)
        fi_path = None
        try:
            fi_path = plot_and_save_feature_importance(
                model=clf, feature_names=feature_names, topn=min(20, X.shape[1]), out_dir=out_dir,
                filename=f'feature_importance_{clf_name}.png',
            )
        except (AttributeError, TypeError):
            # Some models (like MLP) don't have easily interpretable feature importance
            print(f"Feature importance not available for {config['name']}")

        return {
            'model': clf,
            'y_pred': y_pred,
            'report_str': rep_str,
            'report_path': report_path,
            'confusion_matrix_path': cm_path,
            'feature_importance_path': fi_path,
            'test_accuracy': accuracy_score(y_test, y_pred)
        }

    # Train all selected classifiers
    for clf_name in classifiers_to_run:
        print(f"Training {classifier_configs[clf_name]['name']}...")
        clf_results = _train_classifier(clf_name)
        results['models'][clf_name] = clf_results['model']
        results['reports'][clf_name] = clf_results
        print(f"Test accuracy: {clf_results['test_accuracy']:.4f}")

    return results
