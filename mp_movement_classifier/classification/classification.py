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
from sklearn.decomposition import PCA  # <-- ADDED THIS
import seaborn as sns
from mp_movement_classifier.utils import config

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,

)
from mp_movement_classifier.utils.plotting import (
    plot_learn_curve, plot_mp,
    plot_reconstructions,
    set_figures_directory
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


def compare_classifiers(X_scaled, y, out_dir: Path,
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


# ============================================================================
# NEW FUNCTION: PCA ANALYSIS
# ============================================================================
def analyze_feature_pca(
        X: np.ndarray,
        out_dir: Path,
        max_components: int = None,
        feature_names: list = None
) -> dict:
    """
    Comprehensive PCA analysis on feature matrix.

    This function:
    1. Runs PCA on the feature matrix
    2. Computes and plots variance explained vs number of components
    3. Analyzes feature covariance/correlation to identify repetitive features
    4. Saves all plots to the output directory

    Args:
        X: Feature matrix [n_samples, n_features]
        out_dir: Directory to save plots
        max_components: Maximum number of components to analyze (default: n_features)
        feature_names: Optional list of feature names for labeling

    Returns:
        dict: Dictionary containing PCA results and statistics
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples, n_features = X.shape

    # Set max_components if not specified
    if max_components is None:
        max_components = min(n_samples, n_features)
    else:
        max_components = min(max_components, n_samples, n_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=max_components)
    X_pca = pca.fit_transform(X_scaled)

    # Get variance explained
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Print key statistics
    print(f"\nPCA Results:")
    print(f"  - Total components: {len(explained_variance_ratio)}")
    print(f"  - Variance explained by PC1: {explained_variance_ratio[0]:.4f}")
    print(f"  - Cumulative variance (first 10 PCs): {cumulative_variance[min(9, len(cumulative_variance) - 1)]:.4f}")

    # Find number of components for different variance thresholds
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_comp = np.argmax(cumulative_variance >= threshold) + 1
        if cumulative_variance[-1] >= threshold:
            print(f"  - Components for {threshold:.0%} variance: {n_comp}")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Find highly correlated feature pairs
    high_corr_threshold = 0.8
    high_corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if abs(corr_matrix[i, j]) > high_corr_threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    print(f"\nFeature Correlation Analysis:")
    print(f"  - Highly correlated pairs (|r| > {high_corr_threshold}): {len(high_corr_pairs)}")
    if high_corr_pairs:
        print(f"  - Top 5 correlated pairs:")
        sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]
        for i, j, corr in sorted_pairs:
            feat_i = feature_names[i] if feature_names else f"f{i}"
            feat_j = feature_names[j] if feature_names else f"f{j}"
            print(f"    * {feat_i} <-> {feat_j}: r={corr:.3f}")

    # Compute average correlation (excluding diagonal)
    mask = ~np.eye(n_features, dtype=bool)
    avg_abs_corr = np.mean(np.abs(corr_matrix[mask]))
    print(f"  - Average absolute correlation: {avg_abs_corr:.4f}")

    # --- Plot 1: Variance Explained (Individual and Cumulative) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Individual variance
    components_range = np.arange(1, len(explained_variance_ratio) + 1)
    ax1.bar(components_range, explained_variance_ratio, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Variance Explained Ratio', fontsize=12)
    ax1.set_title('Individual Variance Explained by Each PC', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, min(50, len(explained_variance_ratio) + 0.5))

    # Cumulative variance
    ax2.plot(components_range, cumulative_variance, 'o-', linewidth=2, markersize=4, color='darkred')
    ax2.axhline(y=0.90, color='green', linestyle='--', label='90% variance', linewidth=1.5)
    ax2.axhline(y=0.95, color='orange', linestyle='--', label='95% variance', linewidth=1.5)
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, min(50, len(explained_variance_ratio)))
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    variance_plot_path = out_dir / "pca_variance_explained.png"
    plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {variance_plot_path}")

    # --- Plot 2: Scree Plot (Eigenvalues) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    eigenvalues = pca.explained_variance_
    ax.plot(components_range, eigenvalues, 'o-', linewidth=2, markersize=6, color='darkblue')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Eigenvalue', fontsize=12)
    ax.set_title('Scree Plot - Eigenvalues per Component', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, min(30, len(eigenvalues) + 0.5))

    plt.tight_layout()
    scree_plot_path = out_dir / "pca_scree_plot.png"
    plt.savefig(scree_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {scree_plot_path}")

    # --- Plot 3: Feature Correlation Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a subset if there are too many features
    if n_features > 50:
        # Show first 50 features
        corr_subset = corr_matrix[:50, :50]
        title_suffix = " (First 50 Features)"
    else:
        corr_subset = corr_matrix
        title_suffix = ""

    sns.heatmap(
        corr_subset,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    ax.set_title(f'Feature Correlation Matrix{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)

    plt.tight_layout()
    corr_heatmap_path = out_dir / "feature_correlation_heatmap.png"
    plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {corr_heatmap_path}")

    # --- Plot 4: PCA Component Loadings (First 3 PCs) ---
    n_components_to_show = min(3, pca.n_components_)
    fig, axes = plt.subplots(n_components_to_show, 1, figsize=(12, 4 * n_components_to_show))

    if n_components_to_show == 1:
        axes = [axes]

    for i in range(n_components_to_show):
        loadings = pca.components_[i]
        feature_indices = np.arange(n_features)

        axes[i].bar(feature_indices, loadings, alpha=0.7, color='teal')
        axes[i].set_xlabel('Feature Index', fontsize=11)
        axes[i].set_ylabel('Loading', fontsize=11)
        axes[i].set_title(
            f'PC{i + 1} Loadings (explains {explained_variance_ratio[i]:.2%} variance)',
            fontsize=12,
            fontweight='bold'
        )
        axes[i].grid(True, alpha=0.3, linestyle='--', axis='y')
        axes[i].axhline(y=0, color='black', linewidth=0.8)

        # Highlight top loadings
        top_k = 5
        top_indices = np.argsort(np.abs(loadings))[-top_k:]
        for idx in top_indices:
            feat_name = feature_names[idx] if feature_names else f"f{idx}"
            axes[i].text(idx, loadings[idx], f"{feat_name}",
                         rotation=90, va='bottom' if loadings[idx] > 0 else 'top',
                         fontsize=8, color='red')

    plt.tight_layout()
    loadings_plot_path = out_dir / "pca_component_loadings.png"
    plt.savefig(loadings_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {loadings_plot_path}")

    # --- Plot 5: 2D PCA Projection ---
    if pca.n_components_ >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=50, c='steelblue')
        ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance)', fontsize=12)
        ax.set_title('2D PCA Projection', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        projection_path = out_dir / "pca_2d_projection.png"
        plt.savefig(projection_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {projection_path}")

    results = {
        'pca_model': pca,
        'scaler': scaler,
        'X_pca': X_pca,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'eigenvalues': eigenvalues,
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'avg_abs_correlation': avg_abs_corr,
        'n_features': n_features,
        'n_components': pca.n_components_,
    }

    # Save numerical results to text file
    results_text_path = out_dir / "pca_analysis_summary.txt"
    with open(results_text_path, 'w') as f:
        f.write("PCA ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Feature matrix shape: {X.shape}\n")
        f.write(f"Number of components analyzed: {pca.n_components_}\n\n")

        f.write("VARIANCE EXPLAINED:\n")
        f.write("-" * 60 + "\n")
        for i in range(min(10, len(explained_variance_ratio))):
            f.write(f"  PC{i + 1}: {explained_variance_ratio[i]:.4f} "
                    f"(cumulative: {cumulative_variance[i]:.4f})\n")
        f.write("\n")

        for threshold in [0.80, 0.90, 0.95, 0.99]:
            n_comp = np.argmax(cumulative_variance >= threshold) + 1
            if cumulative_variance[-1] >= threshold:
                f.write(f"Components needed for {threshold:.0%} variance: {n_comp}\n")
        f.write("\n")

        f.write("FEATURE CORRELATION:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Highly correlated pairs (|r| > {high_corr_threshold}): {len(high_corr_pairs)}\n")
        f.write(f"Average absolute correlation: {avg_abs_corr:.4f}\n")
        f.write(f"Max correlation: {np.max(corr_matrix[mask]):.4f}\n")
        f.write(f"Min correlation: {np.min(corr_matrix[mask]):.4f}\n")

        if high_corr_pairs:
            f.write("\nTop 10 correlated pairs:\n")
            sorted_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]
            for i, j, corr in sorted_pairs:
                feat_i = feature_names[i] if feature_names else f"f{i}"
                feat_j = feature_names[j] if feature_names else f"f{j}"
                f.write(f"  {feat_i} <-> {feat_j}: r={corr:.3f}\n")

    print(f"  ✓ Saved: {results_text_path}")

    return results


def main():
    global data_dir, model_path

    num_MPs = 20
    cutoff_freq = 6.0
    tpoints = 30
    model_name = f"mp_model_{num_MPs}_cutoff_{cutoff_freq}"
    model_dir = os.path.join(config.SAVING_DIR, f"mp_model_{num_MPs}_cutoff_{cutoff_freq}_tpoints_{tpoints}")
    model_file = os.path.join(model_dir, f"mp_model_{num_MPs}_PC_init_cutoff_{cutoff_freq}_tpoints_{tpoints}")
    out_dir = os.path.join(model_dir, "classification_PCA_20Componants")
    model_path = model_file

    folder_path = "../../data/bvh_files"
    bvh_data, motion_ids = read_bvh_files(folder_path)

    # Process data according to paper specifications
    processed_data, segment_motion_ids = process_bvh_data(
        data_dir = folder_path,
        motion_ids = motion_ids,
        cutoff_freq= cutoff_freq,
    )
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
    X = prepare_weights_for_classification(model, num_segments, num_signals, num_MPs)
    y = np.array(segment_motion_ids)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    print(f"Unique motion IDs: {np.unique(y)}")

    # ============================================================
    # NEW: PCA ANALYSIS ON FEATURE MATRIX
    # ============================================================
    print("\n" + "=" * 70)
    print("PERFORMING PCA ANALYSIS ON FEATURE MATRIX")
    print("=" * 70)

    # Create feature names for better interpretability
    feature_names = []
    for signal_idx in range(num_signals):
        for mp_idx in range(num_MPs):
            feature_names.append(f"signal_{signal_idx}_mp_{mp_idx}")

    # Run comprehensive PCA analysis
    pca_results = analyze_feature_pca(
        X=X,
        out_dir=Path(out_dir),
        max_components=400,  # Analyze all possible components
        feature_names=feature_names
    )

    # OPTIONAL: Use PCA features for classification
    # Uncomment the following lines if you want to classify using reduced features

    # # Get number of components for 95% variance
    # n_components_95 = np.argmax(pca_results['cumulative_variance'] >= 0.95) + 1
    n_components_95 =400
    # print(f"\n[INFO] Using {n_components_95} PCA components for 95% variance")
    #
    # # Replace original features with PCA features
    X = pca_results['X_pca'][:, :n_components_95]
    # print(f"[INFO] Reduced feature matrix shape: {X.shape}")

    # ============================================================
    # CONTINUE WITH ORIGINAL CLASSIFICATION
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    # Build and save report
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
    classifier_results = compare_classifiers(X_all_scaled, y, out_dir=out_dir)


if __name__ == "__main__":
    main()



# from __future__ import annotations
#
# import os
# import sys
# import json
# import pickle
# from pathlib import Path
# from typing import Optional, Tuple, List
#
# import numpy as np
# import pandas as pd
# import matplotlib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import cross_val_score
# import seaborn as sns
# from mp_movement_classifier.utils import config
#
# # Use non-interactive backend for headless environments
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt  # noqa: E402
# import seaborn as sns  # noqa: E402
#
#
# from mp_movement_classifier.utils.utils import (
#     load_model_with_full_state,
#     process_bvh_data,
#     read_bvh_files,
#     save_model_with_full_state,
#
# )
# from mp_movement_classifier.utils.plotting import (
#     plot_learn_curve, plot_mp,
#     plot_reconstructions,
#     set_figures_directory
# )
#
# # Globals (populated in main)
# data_dir: Optional[Path] = None
# model_path: Optional[Path] = None
#
#
# def prepare_weights_for_classification(model, num_segments, num_signals, num_MPs=20):
#     """
#     Returns:
#         X: Feature matrix with shape [num_segments, num_signals * num_MPs]
#     """
#     X = np.zeros((num_segments, num_signals * num_MPs))
#
#     for seg_idx in range(num_segments):
#         for signal_idx in range(num_signals):
#             for mp_idx in range(num_MPs):
#                 feature_idx = signal_idx * num_MPs + mp_idx
#                 X[seg_idx, feature_idx] = model.weights[seg_idx][signal_idx, mp_idx].item()
#
#     return X
#
# def save_classification_report(report_str: str, out_dir: str, filename: str = "classification_report.txt"):
#     path = os.path.join(out_dir, filename)
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#
#     with open(path, "w", encoding="utf-8") as f:
#         f.write(report_str)
#     return path
#
#
# def plot_and_save_confusion_matrix(
#     y_true: np.ndarray,
#     y_pred: np.ndarray,
#     out_dir: Path,
#     filename: str = "confusion_matrix.png",
#     labels: Optional[List[str]] = None,
# ) -> Path:
#     """
#     Plot and save a confusion matrix heatmap.
#     """
#     cm = confusion_matrix(y_true, y_pred)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
#
#     ax.set_title("Confusion Matrix")
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#
#     if labels is not None:
#         ax.set_xticklabels(labels, rotation=45, ha="right")
#         ax.set_yticklabels(labels, rotation=0)
#
#     out_path = Path(out_dir) / filename
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#     return out_path
#
#
# def plot_and_save_feature_importance(
#     model,
#     feature_names: Optional[List[str]],
#     topn: int,
#     out_dir: Path,
#     filename: str = "feature_importance.png",
# ) -> Optional[Path]:
#     """
#     Plot and save top-N feature importances. Supports:
#     - feature_importances_ (tree-based models)
#     - coef_ (linear models), using mean absolute value across classes if needed
#     """
#     importances = None
#
#     if hasattr(model, "feature_importances_"):
#         try:
#             importances = np.asarray(model.feature_importances_)
#         except Exception:
#             importances = None
#     elif hasattr(model, "coef_"):
#         try:
#             coef = np.asarray(model.coef_)
#             if coef.ndim == 1:
#                 importances = np.abs(coef)
#             else:
#                 # Multi-class: average absolute weights across classes
#                 importances = np.mean(np.abs(coef), axis=0)
#         except Exception:
#             importances = None
#
#     if importances is None:
#         # Nothing to plot
#         return None
#
#     n_features = importances.shape[0]
#     if feature_names is None or len(feature_names) != n_features:
#         feature_names = [f"f{i}" for i in range(n_features)]
#
#     idx_sorted = np.argsort(importances)[::-1]
#     topn = max(1, min(int(topn), n_features))
#     top_idx = idx_sorted[:topn]
#
#     fig, ax = plt.subplots(figsize=(10, max(4, int(topn * 0.3))))
#     sns.barplot(
#         x=importances[top_idx],
#         y=[feature_names[i] for i in top_idx],
#         ax=ax,
#         orient="h",
#         color="#4C72B0",
#     )
#     ax.set_title(f"Top-{topn} Feature Importances")
#     ax.set_xlabel("Importance")
#     ax.set_ylabel("Feature")
#     fig.tight_layout()
#
#     out_path = Path(out_dir) / filename
#     fig.savefig(out_path, dpi=150)
#     plt.close(fig)
#     return out_path
#
#
# def compare_classifiers(X_scaled, y,out_dir: Path,
#     filename: str = "classifier_comparison.png"):
#     """
#     Compare different classification algorithms
#     """
#     classifiers = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#         'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#         'SVM': SVC(random_state=42),
#         'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
#     }
#
#     results = {}
#
#     for name, clf in classifiers.items():
#         # Perform 5-fold cross-validation
#         scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
#         results[name] = scores
#         print(f"{name}: Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")
#
#     plt.figure(figsize=(10, 6))
#     plt.boxplot(list(results.values()), labels=list(results.keys()))
#     plt.title('Classifier Comparison')
#     plt.ylabel('Accuracy')
#     plt.grid(True, alpha=0.3)
#     out_path = Path(out_dir) / filename
#     plt.savefig(out_path, dpi=150)
#     plt.close()
#     print(f"[info] Classifier Comparison saved to: {out_path}")
#
#     return results
#
#
# def main():
#     global data_dir, model_path
#
#     num_MPs = 20
#     cutoff_freq = 3
#     tpoints = 30
#     model_name = f"mp_model_{num_MPs}_cutoff_{cutoff_freq}"
#     # model_dir = os.path.join(config.SAVING_DIR, model_name)
#     model_dir = os.path.join(config.SAVING_DIR, f"new_seg_mp_model_{num_MPs}_cutoff_{cutoff_freq}_tpoints_{tpoints}")
#     model_file = os.path.join(model_dir, f"mp_model_{num_MPs}_PC_init_cutoff_{cutoff_freq}_tpoints_{tpoints}")
#     out_dir = os.path.join(model_dir, "classification")
#     model_path = model_file
#
#     folder_path = "../../data/bvh_files"
#     bvh_data, motion_ids = read_bvh_files(folder_path)
#
#     # Process data according to paper specifications
#     processed_data, segment_motion_ids = process_bvh_data(bvh_data, motion_ids)
#
#     # based on TMP code: the format of data=list(segment_data[signals,time])
#     num_segments = len(processed_data)
#     num_signals = processed_data[0].shape[0]
#
#     model = load_model_with_full_state(
#         model_path,
#         num_segments=num_segments,
#         num_signals=num_signals
#     )
#
#     segment_lengths = np.array([segment.shape[1] for segment in processed_data])
#     recon_data = model.predict(segment_lengths, as_numpy=True)
#
#     # feature space
#     X= prepare_weights_for_classification(model, num_segments, num_signals, num_MPs)
#     y = np.array(segment_motion_ids)
#     print(f"Feature matrix shape: {X.shape}")
#     print(f"Label array shape: {y.shape}")
#     print(f"Unique motion IDs: {np.unique(y)}")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
#
#     scaler = StandardScaler()
#     scaler.fit_transform(X_train)
#     X_train_scaled = scaler.transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X_train_scaled, y_train)
#
#     y_pred = clf.predict(X_test_scaled)
#
#     unique_labels = np.unique(y_test)
#     report = classification_report(y_test, y_pred, labels=unique_labels)
#     report_path = save_classification_report(report, str(out_dir))
#     print(f"[info] Classification report saved to: {report_path}")
#
#     # Confusion matrix
#     label_names = [str(lbl) for lbl in unique_labels]
#     cm_path = plot_and_save_confusion_matrix(y_test, y_pred, out_dir, labels=label_names)
#     print(f"[info] Confusion matrix saved to: {cm_path}")
#
#     # Feature importances
#     fi_path = plot_and_save_feature_importance(
#         model=clf,
#         feature_names=None,
#         topn=10,
#         out_dir=out_dir,
#     )
#     if fi_path:
#         print(f"[info] Feature importance plot saved to: {fi_path}")
#     else:
#         print("[info] Model does not expose feature importances/coefs; skipping importance plot.")
#
#     # Compare classifiers
#     X_all_scaled = scaler.fit_transform(X)
#     classifier_results = compare_classifiers(X_all_scaled, y,out_dir=out_dir)
#
#
# if __name__ == "__main__":
#     main()
