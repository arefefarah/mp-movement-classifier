from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.metrics import classification_report, confusion_matrix

MOTION_NAMES = {
        0: 'hand clapping',
        1: 'jogging',
        2: 'walking',
        3: 'cross-legged sitting',
        4: 'vertical jumping',
        5: 'crawling',
        6: 'hand waving',
        7: 'running in spot',
        8: 'checking watch',
        9: 'kicking',
        10: 'taking photo',
        11: 'cross arms',
        12: 'jumping jacks',
        13: 'scratching head',
        14: 'throw & catch',
        17: 'sitting down',
        18: 'sideways',
        20: 'stretching',
    }
# --- Extracted helpers (single source of truth) ---

def prepare_weights_for_classification(model, num_segments, num_signals, num_MPs=20):
    """
    Build TMP feature matrix from learned weights.

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


def save_classification_report(report_str: str, out_dir: str, filename: str = "classification_report.txt") -> str:
    path = os.path.join(out_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(report_str)
    return path


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


def compare_classifiers(X_scaled, y, out_dir: Path, filename: str = "classifier_comparison.png"):
    """
    Compare different classification algorithms via 5-fold CV on the provided (scaled) features.
    """
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': LinearSVC(C=1.0, penalty='l2', dual=True),
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


def analyze_feature_pca(
    X: np.ndarray,
    y: np.array,
    out_dir: Path,
    max_components: int = None,
    feature_names: list = None,
):
    """
    Perform PCA analysis on features X and create several plots:
    - 2D scatter (PC1 vs PC2) colored by y
    - Variance explained bar + cumulative curves
    - Optional feature loadings heatmap if feature_names provided
    Returns a dict with PCA model and arrays (e.g., transformed data, explained variance ratios).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_components = min(max_components or X.shape[1], X.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # --- categorical legend setup (move imports/dict to top of file if reused) ---

    classes = np.unique(y)
    n_classes = len(classes)
    cmap = cm.get_cmap('hsv', n_classes)
    colors = [cmap(i) for i in range(n_classes)]
    class_names = {c: MOTION_NAMES.get(int(c), str(c)) for c in classes}

    # --- scatter PC1 vs PC2 ---
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for i, c in enumerate(classes):
        mask = (y == c)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   s=14, alpha=0.8, color=colors[i],
                   label=class_names[c], linewidths=0)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=11)
    ax.set_title('PCA of Features', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / 'pca_scatter.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.savefig(plot_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    # # Scatter PC1 vs PC2
    # fig, ax = plt.subplots(figsize=(3.8, 3.4))
    # scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='hsv', alpha=0.8, s=14)
    # cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Motion Type', fontsize=10)
    # cbar.ax.tick_params(labelsize=9)
    # ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})', fontsize=11)
    # ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})', fontsize=11)
    # ax.set_title('PCA of Features', fontsize=13, fontweight='bold')
    # ax.tick_params(axis='both', labelsize=10)
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plot_path = out_dir / 'pca_scatter.png'
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.savefig(plot_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    # plt.close()

    # Variance explained plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    upto = min(80, len(pca.explained_variance_ratio_))
    ax1.bar(range(1, upto + 1), pca.explained_variance_ratio_[:upto], alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Variance Explained by Each PC')
    ax1.grid(True, alpha=0.3)

    cumsum = np.cumsum(pca.explained_variance_ratio_[:upto])
    ax2.plot(range(1, upto + 1), cumsum, marker='o', linewidth=2, markersize=5)
    ax2.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(y=0.95, color='g', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = out_dir / 'pca_variance_explained.png'
    plt.savefig(out_dir / 'pca_variance_explained.png', dpi=150)
    plt.savefig(plot_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close()

    # Optional: feature loadings heatmap
    if feature_names is not None and len(feature_names) == X.shape[1]:
        try:
            loadings = pca.components_  # shape (n_components, n_features)
            upto_load = min(10, loadings.shape[0])
            fig, ax = plt.subplots(figsize=(12, 0.4 * upto_load * len(feature_names) / max(1, X.shape[1] // 20) + 3))
            sns.heatmap(loadings[:upto_load, :], cmap='RdBu_r', center=0.0, ax=ax)
            ax.set_title('Top PCA Component Loadings (first 10 PCs)')
            ax.set_xlabel('Features')
            ax.set_ylabel('PC Index')
            plt.tight_layout()
            plt.savefig(out_dir / 'pca_feature_loadings.png', dpi=150)
            plt.close()
        except Exception as e:
            print(f"[warn] Could not generate loadings heatmap: {e}")

    return {
        'pca_model': pca,
        'X_pca': X_pca,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
    }


def visualize_with_tsne(X, y, out_dir):
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X)

    # fig, ax = plt.subplots(figsize=(3.8, 3.4))
    # scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='hsv', alpha=0.8, s=14)
    # cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Motion Type', fontsize=10)
    # cbar.ax.tick_params(labelsize=9)
    # ax.set_xlabel('t-SNE 1', fontsize=11)
    # ax.set_ylabel('t-SNE 2', fontsize=11)
    # ax.set_title('t-SNE of Features', fontsize=13, fontweight='bold')
    # ax.tick_params(axis='both', labelsize=10)
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # out_path = Path(out_dir) / 'tsne_visualization.png'
    # plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    # plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    # plt.close()
    # return tsne

    classes = np.unique(y)
    n_classes = len(classes)
    cmap = cm.get_cmap('hsv', n_classes)
    colors = [cmap(i) for i in range(n_classes)]
    class_names = {c: MOTION_NAMES.get(int(c), str(c)) for c in classes}

    # --- plot ---
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for i, c in enumerate(classes):
        mask = (y == c)
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   s=14, alpha=0.8, color=colors[i],
                   label=class_names[c], linewidths=0)

    # f'PC1 ({pca.explained_variance_ratio_[0]:.2f})'
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('t-SNE of Features', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(out_dir) / 'tsne_visualization.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out_path.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close()
    return tsne

from matplotlib.lines import Line2D

def save_motion_legend(
    out_path: Path,
    motion_ids: list,
    ncol: int = 2,
    marker_size: int = 9,
    fontsize: int = 10,
):
    """Render a standalone legend mapping motion IDs to color swatches + names.
    """
    cmap_name = 'hsv'
    motion_names = MOTION_NAMES
    cmap = cm.get_cmap(cmap_name, len(motion_ids))
    handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=cmap(i), markeredgecolor='none',
               markersize=marker_size,
               label=motion_names.get(int(c), str(c)))
        for i, c in enumerate(motion_ids)
    ]

    # Size the legend canvas to its contents — no axes, no extras.
    n_rows = -(-len(motion_ids) // ncol)  # ceil
    fig = plt.figure(figsize=(2.6 * ncol, 0.28 * n_rows + 0.2))
    fig.legend(
        handles=handles, loc='center', frameon=False,
        ncol=ncol, fontsize=fontsize,
        handletextpad=0.5, columnspacing=1.2, labelspacing=0.4,
    )
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(Path(out_path).with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    plt.close()


def calculate_rdm(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    filename: str = "rdm_heatmap.png",
):
    """
    Compute a Representational Dissimilarity Matrix (RDM) between class means
    using Euclidean distance and save a heatmap.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.unique(y)
    # Compute class means
    class_means = []
    for lbl in labels:
        class_means.append(X[y == lbl].mean(axis=0))
    class_means = np.stack(class_means, axis=0)

    # Pairwise Euclidean distances between class means
    from scipy.spatial.distance import pdist, squareform
    dists = squareform(pdist(class_means, metric='euclidean'))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dists, cmap='magma', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Euclidean Distance', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_title('Representational Dissimilarity Matrix (Class Means)')
    ax.set_xlabel('Class')
    ax.set_ylabel('Class')
    plt.tight_layout()
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        'labels': labels,
        'class_means': class_means,
        'dist_matrix': dists,
        'figure_path': out_path,
    }


def compute_classification_aic(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    cv_folds: int = 5,
) -> dict:
    """
    Compute AIC for a representation using cross-validated log-likelihood from Random Forest.

    AIC = 2k - 2*ln(L_cv)
      k       = number of input features (representation dimensionality)
      ln(L_cv) = sum of log P(true_class | features) over all samples,
                 estimated by k-fold CV (each sample appears in exactly one test fold)

    Returns dict with keys: aic, k, log_likelihood, n_samples, cv_folds,
                            fold_log_likelihoods
    """
    k = X.shape[1]
    n_samples = X.shape[0]
    eps = 1e-15  # floor to avoid log(0)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_log_likelihoods = []
    total_log_likelihood = 0.0

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        clf_fold = RandomForestClassifier(n_estimators=200, random_state=seed)
        clf_fold.fit(X[train_idx], y[train_idx])

        proba = clf_fold.predict_proba(X[test_idx])       # shape (n_test, n_classes)
        classes = clf_fold.classes_

        # Map true labels -> column index in proba
        class_to_idx = {c: i for i, c in enumerate(classes)}
        col_indices = np.array([class_to_idx[label] for label in y[test_idx]])

        # P(true class) for each test sample, clipped away from 0
        true_probs = np.clip(proba[np.arange(len(test_idx)), col_indices], eps, 1.0)
        fold_ll = float(np.sum(np.log(true_probs)))
        fold_log_likelihoods.append(fold_ll)
        total_log_likelihood += fold_ll

    aic = 2.0 * k - 2.0 * total_log_likelihood

    return {
        'aic': aic,
        'k': k,
        'log_likelihood': total_log_likelihood,
        'n_samples': n_samples,
        'cv_folds': cv_folds,
        'fold_log_likelihoods': fold_log_likelihoods,
    }
