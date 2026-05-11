"""
Generalized LDA Analysis & Visualization for Movement Feature Representations
===============================================================================

Works with ANY method that produces (X, y) feature matrices:
  - TMP weights
  - Legendre polynomial coefficients
  - Autoencoder latent representations
  - Any other method with the same structure

Usage:
    from movement_representation_analysis import (
        run_lda_analysis,
        compare_methods
    )

    # Single method analysis
    results_tmp = run_lda_analysis(
        X=X_tmp, y=y_tmp,
        out_dir='./results/tmp_analysis',
        method_name='TMP Weights',
        feature_structure={'n_signals': 51, 'n_features_per_signal': 5},
        # feature_structure is optional — used for heatmap layout
    )

    # Compare multiple methods
    compare_methods(
        methods={
            'TMP Weights': {'X': X_tmp, 'y': y_tmp},
            'Legendre Coefficients': {'X': X_leg, 'y': y_leg},
            'Autoencoder Latent': {'X': X_ae, 'y': y_ae},
        },
        out_dir='./results/comparison'
    )
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway, skew, kurtosis as kurt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

matplotlib.use('Agg')


def _save_png_svg(path):
    """
    Save the current matplotlib figure as both PNG (300 dpi) and SVG so the
    same image can be used for screen previews and vector-based paper
    figures. ``path`` may be a str or Path with either suffix.
    """
    p = Path(path)
    png = p.with_suffix('.png')
    svg = p.with_suffix('.svg')
    plt.savefig(png, dpi=300, bbox_inches='tight')
    plt.savefig(svg, bbox_inches='tight', facecolor='white')


# =============================================================================
# CORE: LDA Analysis (method-agnostic)
# =============================================================================

def perform_lda_analysis(X, y, out_dir, method_name='Features', n_components=None):
    """
    Full LDA pipeline. Projects into discriminant space and generates all
    LDA-related visualizations.

    Parameters
    ----------
    X           : ndarray (n_samples, n_features)
    y           : ndarray (n_samples,) — class labels (int or str)
    out_dir     : str or Path
    method_name : str — human-readable name for plot titles
    n_components: int or None

    Returns
    -------
    dict with lda model, projections, centroids, distances, cv scores
    """
    # Save scatter / distribution / heatmap figures next to the other
    # analyses produced by ``run_lda_analysis`` (mahalanobis, fisher,
    # confusion RDM, distributional RDM). The orchestrator already passes
    # a method-specific directory, so nesting another ``lda_analysis``
    # folder here only created an awkward double-subdir layout.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y)
    n_classes = len(classes)

    if n_components is None:
        n_components = min(n_classes - 1, X.shape[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, y)

    print("=" * 70)
    print(f"LDA ANALYSIS — {method_name}")
    print("=" * 70)
    print(f"  Classes: {n_classes}  |  Features: {X.shape[1]}  |  Samples: {X.shape[0]}")
    print(f"  Discriminant components: {n_components}")
    print(f"  Explained var (first 5): {lda.explained_variance_ratio_[:5].round(4)}")
    print(f"  Cumulative   (first 3): {np.cumsum(lda.explained_variance_ratio_[:3]).round(4)}")

    # Class centroids in LDA space
    centroids = {}
    for cls in classes:
        centroids[cls] = X_lda[y == cls].mean(axis=0)

    centroid_matrix = np.array([centroids[c] for c in classes])
    centroid_distances = squareform(pdist(centroid_matrix, metric='euclidean'))

    # Cross-val
    cv_scores = cross_val_score(LinearDiscriminantAnalysis(), X_scaled, y, cv=5, scoring='accuracy')
    print(f"  LDA 5-fold CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    results = {
        'lda': lda, 'scaler': scaler, 'X_lda': X_lda, 'y': y,
        'classes': classes, 'centroids': centroids,
        'centroid_matrix': centroid_matrix,
        'centroid_distances': centroid_distances,
        'cv_scores': cv_scores, 'method_name': method_name,
    }

    _plot_lda_2d(results, out_dir)
    if n_components >= 3:
        _plot_lda_3d(results, out_dir)
    _plot_lda_explained_variance(results, out_dir)
    _plot_lda_centroid_rdm(results, out_dir)
    _plot_lda_class_separation(results, out_dir)
    _plot_lda_scalings_heatmap(results, out_dir)

    return results


def _cmap_for(n):
    return plt.cm.tab20 if n > 10 else plt.cm.tab10


def _plot_lda_2d(r, d):
    """
    LD1 vs LD2 scatter, styled to match
    ``classification.utils.analyze_feature_pca``'s ``pca_scatter`` so the
    two figures sit next to each other cleanly in the manuscript.

    Matches:
      - ``figsize=(3.8, 3.4)``
      - ``hsv`` colormap discretized by ``n_classes``
      - per-class scatter loop with ``s=14, alpha=0.8, linewidths=0``
      - axis label format ``f'LD1 ({evr:.2f})'`` (decimal, two places)
      - title fontsize 13 bold, axis labels fontsize 11, ticks fontsize 10
      - no inline legend (paired with the separate ``motion_legend.png``)
    """
    from matplotlib import cm
    try:
        from mp_movement_classifier.classification.utils import MOTION_NAMES
    except Exception:
        MOTION_NAMES = {}

    X_lda, y, classes, lda = r['X_lda'], r['y'], r['classes'], r['lda']

    n_classes = len(classes)
    cmap = cm.get_cmap('hsv', n_classes)
    colors = [cmap(i) for i in range(n_classes)]
    class_names = {c: MOTION_NAMES.get(int(c), str(c)) for c in classes}

    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for i, c in enumerate(classes):
        mask = (y == c)
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                   s=14, alpha=0.8, color=colors[i],
                   label=class_names[c], linewidths=0)

    evr = lda.explained_variance_ratio_
    ax.set_xlabel(f'LD1 ({evr[0]:.2f})', fontsize=11)
    ax.set_ylabel(f'LD2 ({evr[1]:.2f})', fontsize=11)
    ax.set_title('LDA of Features', fontsize=13, fontweight='bold')
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_png_svg(d / 'lda_2d_scatter.png')
    plt.close()


def _plot_lda_3d(r, d):
    X_lda, y, classes, lda, mn = r['X_lda'], r['y'], r['classes'], r['lda'], r['method_name']
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = _cmap_for(len(classes))(np.linspace(0, 1, len(classes)))
    for i, cls in enumerate(classes):
        m = y == cls
        ax.scatter(X_lda[m, 0], X_lda[m, 1], X_lda[m, 2], c=[colors[i]], alpha=0.4, s=30, label=f'{cls}')
    evr = lda.explained_variance_ratio_
    ax.set_xlabel(f'LD1 ({evr[0]:.1%})', fontweight='bold')
    ax.set_ylabel(f'LD2 ({evr[1]:.1%})', fontweight='bold')
    ax.set_zlabel(f'LD3 ({evr[2]:.1%})', fontweight='bold')
    ax.set_title(f'LDA 3D — {mn}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    _save_png_svg(d / 'lda_3d_scatter.png')
    plt.close()


def _plot_lda_explained_variance(r, d):
    evr = r['lda'].explained_variance_ratio_
    mn = r['method_name']
    n = len(evr)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.bar(range(1, n+1), evr, alpha=0.7, edgecolor='black')
    a1.set_xlabel('Linear Discriminant', fontweight='bold')
    a1.set_ylabel('Between-Class Variance Ratio', fontweight='bold')
    a1.set_title(f'Variance per LD — {mn}', fontweight='bold')
    a1.grid(True, alpha=0.3)
    cum = np.cumsum(evr)
    a2.plot(range(1, n+1), cum, 'o-', lw=2, ms=7)
    a2.axhline(0.90, color='r', ls='--', lw=2, label='90%')
    a2.axhline(0.95, color='g', ls='--', lw=2, label='95%')
    a2.set_xlabel('Number of LDs', fontweight='bold')
    a2.set_ylabel('Cumulative Variance', fontweight='bold')
    a2.set_title(f'Cumulative — {mn}', fontweight='bold')
    a2.grid(True, alpha=0.3); a2.legend()
    plt.tight_layout()
    _save_png_svg(d / 'lda_explained_variance.png')
    plt.close()


def _plot_lda_centroid_rdm(r, d):
    dist, classes, mn = r['centroid_distances'], r['classes'], r['method_name']
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dist, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax, label='Euclidean Distance in LDA Space')
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            c = 'white' if dist[i, j] > dist.max() * 0.5 else 'black'
            ax.text(j, i, f'{dist[i,j]:.1f}', ha='center', va='center', color=c, fontsize=7)
    ax.set_title(f'RDM (LDA Centroid Distance) — {mn}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Motion ID', fontweight='bold'); ax.set_ylabel('Motion ID', fontweight='bold')
    plt.tight_layout()
    _save_png_svg(d / 'lda_centroid_rdm.png')
    plt.close()


def _plot_lda_class_separation(r, d):
    X_lda, y, classes, mn = r['X_lda'], r['y'], r['classes'], r['method_name']
    n_lds = min(3, X_lda.shape[1])
    fig, axes = plt.subplots(1, n_lds, figsize=(6 * n_lds, 6))
    if n_lds == 1: axes = [axes]
    colors = _cmap_for(len(classes))(np.linspace(0, 1, len(classes)))
    for idx, ax in enumerate(axes):
        data = [X_lda[y == cls, idx] for cls in classes]
        bp = ax.boxplot(data, tick_labels=classes, patch_artist=True)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_xlabel('Motion ID', fontweight='bold')
        ax.set_ylabel(f'LD{idx+1}', fontweight='bold')
        ax.set_title(f'LD{idx+1} Distributions — {mn}', fontweight='bold')
        ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    _save_png_svg(d / 'lda_class_distributions.png')
    plt.close()


def _plot_lda_scalings_heatmap(r, d):
    lda, mn = r['lda'], r['method_name']
    scalings = lda.scalings_
    n_show = min(5, scalings.shape[1])
    fig, ax = plt.subplots(figsize=(8, max(6, scalings.shape[0] * 0.08)))
    sns.heatmap(scalings[:, :n_show], cmap='RdBu_r', center=0, ax=ax,
                xticklabels=[f'LD{i+1}' for i in range(n_show)], yticklabels=False)
    ax.set_xlabel('Linear Discriminant', fontweight='bold')
    ax.set_ylabel('Feature Index', fontweight='bold')
    ax.set_title(f'LDA Scalings — {mn}', fontweight='bold')
    plt.tight_layout()
    _save_png_svg(d / 'lda_scalings_heatmap.png')
    plt.close()


# =============================================================================
# Mahalanobis-Distance RDM
# =============================================================================

def compute_mahalanobis_rdm(X, y, out_dir, method_name='Features'):
    """
    Pairwise Mahalanobis distance between class distributions.
    Uses pooled within-class covariance (same as LDA internally).
    """
    out_dir = Path(out_dir) / 'mahalanobis_rdm'
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y)
    n_classes = len(classes)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    means = {cls: X_s[y == cls].mean(axis=0) for cls in classes}

    n_f = X_s.shape[1]
    S_w = np.zeros((n_f, n_f))
    for cls in classes:
        diff = X_s[y == cls] - means[cls]
        S_w += diff.T @ diff
    S_w /= (X_s.shape[0] - n_classes)
    S_w += np.eye(n_f) * 1e-4

    try:
        S_w_inv = np.linalg.inv(S_w)
    except np.linalg.LinAlgError:
        S_w_inv = np.linalg.pinv(S_w)

    dist = np.zeros((n_classes, n_classes))
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i < j:
                delta = means[ci] - means[cj]
                d = np.sqrt(delta @ S_w_inv @ delta)
                dist[i, j] = dist[j, i] = d

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dist, cmap='magma', aspect='auto')
    plt.colorbar(im, ax=ax, label='Mahalanobis Distance')
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    for i in range(n_classes):
        for j in range(n_classes):
            c = 'white' if dist[i, j] > dist.max() * 0.5 else 'black'
            ax.text(j, i, f'{dist[i,j]:.1f}', ha='center', va='center', color=c, fontsize=7)
    ax.set_title(f'Mahalanobis RDM — {method_name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Motion ID', fontweight='bold'); ax.set_ylabel('Motion ID', fontweight='bold')
    plt.tight_layout()
    _save_png_svg(out_dir / 'mahalanobis_rdm.png')
    plt.close()

    return dist, classes


# =============================================================================
# Fisher Discriminant Ratio per Feature
# =============================================================================

def compute_fisher_ratios(X, y, out_dir, method_name='Features', feature_structure=None):
    """
    Per-feature Fisher discriminant ratio + ANOVA p-values.

    Parameters
    ----------
    feature_structure : dict or None
        If provided, reshapes FDR into a heatmap.
        Keys: 'n_signals' (e.g. n_joints), 'n_features_per_signal' (e.g. n_coefficients or n_weights)
        Labels (optional): 'signal_label' (default 'Signal'), 'feature_label' (default 'Feature')
    """
    out_dir = Path(out_dir) / 'fisher_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y)
    n_f = X.shape[1]
    grand_mean = X.mean(axis=0)

    var_b = np.zeros(n_f)
    var_w = np.zeros(n_f)
    for cls in classes:
        Xc = X[y == cls]
        n_k = len(Xc)
        var_b += n_k * (Xc.mean(axis=0) - grand_mean) ** 2
        var_w += Xc.var(axis=0) * n_k
    var_b /= X.shape[0]
    var_w /= X.shape[0]
    fdr = np.where(var_w > 1e-10, var_b / var_w, 0)

    # Bar plot
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(n_f), fdr, alpha=0.7, width=1.0)
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_ylabel('Fisher Discriminant Ratio', fontweight='bold')
    ax.set_title(f'Feature Discriminability — {method_name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_png_svg(out_dir / 'fisher_ratio_bar.png')
    plt.close()

    # Heatmap if structure known. We only need the signal × feature layout —
    # autoencoder latents pass ``feature_structure={'n_features': D}`` which
    # has neither key, so skip the heatmap gracefully instead of crashing
    # the whole LDA pipeline.
    n_sig = feature_structure.get('n_signals') if feature_structure else None
    n_fpersig = feature_structure.get('n_features_per_signal') if feature_structure else None
    if n_sig is not None and n_fpersig is not None:
        sig_label = feature_structure.get('signal_label', 'Signal')
        feat_label = feature_structure.get('feature_label', 'Feature Index')

        if n_sig * n_fpersig == n_f:
            fdr_r = fdr.reshape(n_sig, n_fpersig)
            fig, ax = plt.subplots(figsize=(max(6, n_fpersig * 0.8), max(6, n_sig * 0.15)))
            sns.heatmap(fdr_r, cmap='YlOrRd', ax=ax,
                        annot=(n_fpersig <= 12), fmt='.2f' if n_fpersig <= 12 else '',
                        xticklabels=[str(i) for i in range(n_fpersig)],
                        yticklabels=[str(i) for i in range(n_sig)])
            ax.set_xlabel(feat_label, fontweight='bold')
            ax.set_ylabel(sig_label, fontweight='bold')
            ax.set_title(f'Fisher Ratio Heatmap — {method_name}', fontsize=13, fontweight='bold')
            plt.tight_layout()
            _save_png_svg(out_dir / 'fisher_ratio_heatmap.png')
            plt.close()

    # ANOVA
    p_values = np.zeros(n_f)
    for j in range(n_f):
        groups = [X[y == cls, j] for cls in classes]
        _, p_values[j] = f_oneway(*groups)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(n_f), -np.log10(p_values + 1e-300), alpha=0.7, width=1.0, color='coral')
    ax.axhline(-np.log10(0.05 / n_f), color='red', ls='--', lw=2,
               label=f'Bonferroni (α=0.05/{n_f})')
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_ylabel('-log10(p)', fontweight='bold')
    ax.set_title(f'ANOVA per Feature — {method_name}', fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_png_svg(out_dir / 'anova_pvalues.png')
    plt.close()

    return fdr, p_values


# =============================================================================
# Confusion-Matrix Based RDM
# =============================================================================

def compute_confusion_rdm(X, y, out_dir, method_name='Features', n_repeats=10):
    """
    Dissimilarity(i,j) = 1 − avg confusion rate between classes i and j.
    Directly reflects what the classifier exploits.
    """
    out_dir = Path(out_dir) / 'confusion_rdm'
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y)
    n_c = len(classes)
    cm_acc = np.zeros((n_c, n_c))

    for seed in range(n_repeats):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
        clf = LinearSVC(C=1.0, dual=True, max_iter=5000)
        clf.fit(Xtr_s, ytr)
        yp = clf.predict(Xte_s)
        cm = confusion_matrix(yte, yp, labels=classes)
        cm_acc += cm.astype(float) / cm.sum(axis=1, keepdims=True)

    cm_avg = cm_acc / n_repeats
    confusion_sym = 0.5 * (cm_avg + cm_avg.T)
    dissimilarity = 1.0 - confusion_sym

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))

    im1 = axes[0].imshow(cm_avg, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[0], label='Confusion Rate')
    axes[0].set_xticks(range(n_c)); axes[0].set_yticks(range(n_c))
    axes[0].set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
    axes[0].set_yticklabels(classes, fontsize=7)
    axes[0].set_title(f'Avg Confusion Matrix — {method_name}', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    im2 = axes[1].imshow(dissimilarity, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1], label='Dissimilarity (1 − confusion)')
    axes[1].set_xticks(range(n_c)); axes[1].set_yticks(range(n_c))
    axes[1].set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
    axes[1].set_yticklabels(classes, fontsize=7)
    axes[1].set_title(f'Confusion-Based RDM — {method_name}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    _save_png_svg(out_dir / 'confusion_based_rdm.png')
    plt.close()

    return dissimilarity, cm_avg, classes


# =============================================================================
# Distributional RDM (7 statistics instead of just mean)
# =============================================================================

def compute_distributional_rdm(X, y, out_dir, method_name='Features'):
    """
    Represent each class by [mean, median, std, skewness, kurtosis, P10, P90]
    then compute RDM. Captures what averaging alone misses.
    """
    out_dir = Path(out_dir) / 'distributional_rdm'
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(y)
    n_c = len(classes)

    rich = []
    for cls in classes:
        Xc = X[y == cls]
        rich.append(np.concatenate([
            Xc.mean(0), np.median(Xc, 0), Xc.std(0),
            skew(Xc, 0), kurt(Xc, 0),
            np.percentile(Xc, 10, 0), np.percentile(Xc, 90, 0),
        ]))
    rich = np.array(rich)
    rich_s = StandardScaler().fit_transform(rich)

    rdm_corr = 1 - np.corrcoef(rich_s)
    rdm_eucl = squareform(pdist(rich_s, metric='euclidean'))

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, mat, cmap, label, title in [
        (axes[0], rdm_corr, 'RdYlBu_r', '1 − Pearson r', 'Correlation-Based'),
        (axes[1], rdm_eucl, 'viridis', 'Euclidean Distance', 'Euclidean'),
    ]:
        im = ax.imshow(mat, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xticks(range(n_c)); ax.set_yticks(range(n_c))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_title(f'{title} Distributional RDM — {method_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save_png_svg(out_dir / 'distributional_rdm.png')
    plt.close()

    return rdm_corr, rdm_eucl, classes

#
# # =============================================================================
# # Variance Decomposition (diagnoses why mean-RDM fails)
# # =============================================================================
#
# def analyze_variance_structure(X, y, out_dir, method_name='Features'):
#     """Between-class / within-class ratio in raw vs LDA space."""
#     out_dir = Path(out_dir) / 'variance_analysis'
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     classes = np.unique(y)
#     gm = X.mean(0)
#     N = X.shape[0]
#
#     Sb, Sw = 0.0, 0.0
#     for cls in classes:
#         Xc = X[y == cls]
#         cm = Xc.mean(0)
#         Sb += len(Xc) * np.sum((cm - gm) ** 2)
#         Sw += np.sum((Xc - cm) ** 2)
#     St = np.sum((X - gm) ** 2)
#     ratio_raw = Sb / Sw
#
#     # LDA space
#     Xs = StandardScaler().fit_transform(X)
#     lda = LinearDiscriminantAnalysis()
#     Xl = lda.fit_transform(Xs, y)
#     gm_l = Xl.mean(0)
#     Sb_l, Sw_l = 0.0, 0.0
#     for cls in classes:
#         Xc = Xl[y == cls]; cm = Xc.mean(0)
#         Sb_l += len(Xc) * np.sum((cm - gm_l) ** 2)
#         Sw_l += np.sum((Xc - cm) ** 2)
#     ratio_lda = Sb_l / Sw_l
#
#     print(f"\n  VARIANCE — {method_name}")
#     print(f"    Raw  B/W ratio: {ratio_raw:.4f}   (between/total = {Sb/St:.4f})")
#     print(f"    LDA  B/W ratio: {ratio_lda:.4f}")
#     print(f"    Improvement:    {ratio_lda/ratio_raw:.1f}×")
#
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     labels = ['Between-class', 'Within-class']
#     for ax, vals, title in [
#         (axes[0], [Sb / St, Sw / St], f'Raw Space (B/W = {ratio_raw:.3f})'),
#         (axes[1], [Sb_l / (Sb_l+Sw_l), Sw_l / (Sb_l+Sw_l)], f'LDA Space (B/W = {ratio_lda:.1f})'),
#     ]:
#         ax.bar(labels, vals, color=['#2196F3', '#FF9800'], edgecolor='black', alpha=0.8)
#         ax.set_title(title, fontsize=13, fontweight='bold')
#         ax.set_ylabel('Proportion of Total Variance', fontweight='bold')
#         ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
#     fig.suptitle(f'Variance Decomposition — {method_name}\n'
#                  f'(explains low RDM values on averaged features)',
#                  fontsize=14, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(out_dir / 'variance_decomposition.png', dpi=300, bbox_inches='tight')
#     plt.close()
#
#     return {'raw_between': Sb, 'raw_within': Sw, 'raw_total': St, 'raw_ratio': ratio_raw,
#             'lda_between': Sb_l, 'lda_within': Sw_l, 'lda_ratio': ratio_lda}


# =============================================================================
# Hierarchical Clustering Dendrogram
# =============================================================================

# def plot_hierarchical_clustering(X, y, out_dir, method_name='Features', space='lda'):
#     out_dir = Path(out_dir) / 'hierarchical_clustering'
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     classes = np.unique(y)
#     Xs = StandardScaler().fit_transform(X)
#
#     if space == 'lda':
#         Xp = LinearDiscriminantAnalysis().fit_transform(Xs, y)
#     else:
#         Xp = Xs
#
#     centroids = np.array([Xp[y == cls].mean(0) for cls in classes])
#     Z = linkage(centroids, method='ward')
#
#     fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.5), 8))
#     dendrogram(Z, labels=[str(c) for c in classes], ax=ax, leaf_rotation=45, leaf_font_size=9)
#     ax.set_ylabel('Ward Distance', fontweight='bold')
#     ax.set_title(f'Hierarchical Clustering ({space.upper()} space) — {method_name}',
#                  fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3, axis='y')
#     plt.tight_layout()
#     plt.savefig(out_dir / f'dendrogram_{space}.png', dpi=300, bbox_inches='tight')
#     plt.close()
#

# =============================================================================
# MASTER: Single-method analysis
# =============================================================================

def run_lda_analysis(X, y, out_dir, method_name='Features', feature_structure=None):
    """
    Run all analyses for a single method.

    Parameters
    ----------
    X                : ndarray (n_samples, n_features)
    y                : ndarray (n_samples,) — class labels
    out_dir          : str or Path
    method_name      : str — e.g. 'TMP Weights', 'Legendre Coefficients', 'Autoencoder Latent'
    feature_structure: dict or None
        Optional layout info for Fisher heatmap.
        Example for TMP:      {'n_signals': 51, 'n_features_per_signal': 5,
                                'signal_label': 'Joint', 'feature_label': 'Primitive Weight'}
        Example for Legendre: {'n_signals': 51, 'n_features_per_signal': 2,
                                'signal_label': 'Joint', 'feature_label': 'Polynomial Degree'}
        Example for AE:        None  (latent dims have no inherent structure)

    Returns
    -------
    dict of all results
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'█' * 70}")
    print(f"  ANALYSIS: {method_name}")
    print(f"  X shape: {X.shape}  |  Classes: {len(np.unique(y))}")
    print(f"{'█' * 70}\n")

    lda_res = perform_lda_analysis(X, y, out_dir, method_name)
    maha_rdm, _ = compute_mahalanobis_rdm(X, y, out_dir, method_name)
    fdr, pvals = compute_fisher_ratios(X, y, out_dir, method_name, feature_structure)
    conf_rdm, conf_mat, _ = compute_confusion_rdm(X, y, out_dir, method_name)
    dist_corr, dist_eucl, _ = compute_distributional_rdm(X, y, out_dir, method_name)
    # var_res = analyze_variance_structure(X, y, out_dir, method_name)
    # plot_hierarchical_clustering(X, y, out_dir, method_name, 'lda')
    # plot_hierarchical_clustering(X, y, out_dir, method_name, 'raw')

    print(f"\n{'=' * 70}")
    print(f"  DONE: {method_name} — outputs in {out_dir}")
    print(f"{'=' * 70}\n")

    return {
        'method_name': method_name,
        'lda': lda_res,
        'mahalanobis_rdm': maha_rdm,
        'fisher_ratios': fdr,
        'confusion_rdm': conf_rdm,
        'distributional_rdm_corr': dist_corr,
        'distributional_rdm_eucl': dist_eucl,
        # 'variance': var_res,
    }


# =============================================================================
# COMPARISON: Side-by-side across methods
# =============================================================================

def compare_methods(methods, out_dir):
    """
    Compare multiple methods side-by-side.

    Parameters
    ----------
    methods : dict
        { 'Method Name': {'X': ndarray, 'y': ndarray}, ... }
        All methods must share the same y labels.
    out_dir : str or Path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(methods.keys())
    n_methods = len(names)

    print(f"\n{'█' * 70}")
    print(f"  CROSS-METHOD COMPARISON: {', '.join(names)}")
    print(f"{'█' * 70}\n")

    # --- 1. LDA projections side-by-side ---
    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 7))
    if n_methods == 1: axes = [axes]

    lda_results = {}
    for idx, (name, data) in enumerate(methods.items()):
        X, y = data['X'], data['y']
        classes = np.unique(y)
        Xs = StandardScaler().fit_transform(X)
        lda = LinearDiscriminantAnalysis(n_components=min(len(classes)-1, X.shape[1]))
        Xl = lda.fit_transform(Xs, y)
        lda_results[name] = {'X_lda': Xl, 'y': y, 'lda': lda, 'classes': classes}

        ax = axes[idx]
        colors = _cmap_for(len(classes))(np.linspace(0, 1, len(classes)))
        for ci, cls in enumerate(classes):
            m = y == cls
            ax.scatter(Xl[m, 0], Xl[m, 1], c=[colors[ci]], alpha=0.4, s=25, label=f'{cls}')
        evr = lda.explained_variance_ratio_
        ax.set_xlabel(f'LD1 ({evr[0]:.1%})', fontweight='bold')
        ax.set_ylabel(f'LD2 ({evr[1]:.1%})', fontweight='bold')
        ax.set_title(f'{name}\n(dim={X.shape[1]})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if idx == n_methods - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2, title='Motion')

    fig.suptitle('LDA Projection Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_lda_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. LDA centroid RDMs side-by-side ---
    fig, axes = plt.subplots(1, n_methods, figsize=(9 * n_methods, 8))
    if n_methods == 1: axes = [axes]

    for idx, (name, lr) in enumerate(lda_results.items()):
        Xl, y, classes = lr['X_lda'], lr['y'], lr['classes']
        centroids = np.array([Xl[y == c].mean(0) for c in classes])
        dist = squareform(pdist(centroids, 'euclidean'))

        ax = axes[idx]
        im = ax.imshow(dist, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        for i in range(len(classes)):
            for j in range(len(classes)):
                c = 'white' if dist[i, j] > dist.max() * 0.5 else 'black'
                ax.text(j, i, f'{dist[i,j]:.1f}', ha='center', va='center', color=c, fontsize=6)

    fig.suptitle('LDA Centroid RDM Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_lda_rdm.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 3. Variance decomposition comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    raw_ratios = []
    lda_ratios = []
    for name, data in methods.items():
        X, y = data['X'], data['y']
        classes = np.unique(y)
        gm = X.mean(0)
        Sb, Sw = 0.0, 0.0
        for cls in classes:
            Xc = X[y == cls]; cm = Xc.mean(0)
            Sb += len(Xc) * np.sum((cm - gm) ** 2)
            Sw += np.sum((Xc - cm) ** 2)
        raw_ratios.append(Sb / Sw)

        Xs = StandardScaler().fit_transform(X)
        Xl = LinearDiscriminantAnalysis().fit_transform(Xs, y)
        gm_l = Xl.mean(0)
        Sb_l, Sw_l = 0.0, 0.0
        for cls in classes:
            Xc = Xl[y == cls]; cm = Xc.mean(0)
            Sb_l += len(Xc) * np.sum((cm - gm_l) ** 2)
            Sw_l += np.sum((Xc - cm) ** 2)
        lda_ratios.append(Sb_l / Sw_l)

    x_pos = np.arange(n_methods)
    w = 0.35
    ax.bar(x_pos - w/2, raw_ratios, w, label='Raw Feature Space', color='#FF9800', edgecolor='black')
    ax.bar(x_pos + w/2, lda_ratios, w, label='LDA Space', color='#2196F3', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Between / Within Class Variance Ratio', fontweight='bold')
    ax.set_title('Class Separability: Raw vs LDA Space', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    # Add value annotations
    for i, (r, l) in enumerate(zip(raw_ratios, lda_ratios)):
        ax.text(i - w/2, r + 0.01, f'{r:.3f}', ha='center', fontsize=9)
        ax.text(i + w/2, l + 0.01, f'{l:.2f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_variance_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 4. Classification accuracy comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cv_means = []
    cv_stds = []
    for name, data in methods.items():
        X, y = data['X'], data['y']
        Xs = StandardScaler().fit_transform(X)
        # LDA classifier
        cv_lda = cross_val_score(LinearDiscriminantAnalysis(), Xs, y, cv=5, scoring='accuracy')
        # LinearSVC classifier
        cv_svc = cross_val_score(LinearSVC(C=1.0, dual=True, max_iter=5000), Xs, y, cv=5, scoring='accuracy')
        cv_means.append([cv_lda.mean(), cv_svc.mean()])
        cv_stds.append([cv_lda.std(), cv_svc.std()])

    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)
    x_pos = np.arange(n_methods)
    w = 0.3
    ax.bar(x_pos - w/2, cv_means[:, 0], w, yerr=cv_stds[:, 0], capsize=4,
           label='LDA', color='#4CAF50', edgecolor='black')
    ax.bar(x_pos + w/2, cv_means[:, 1], w, yerr=cv_stds[:, 1], capsize=4,
           label='LinearSVC', color='#9C27B0', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('5-Fold CV Accuracy', fontweight='bold')
    ax.set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    for i in range(n_methods):
        ax.text(i - w/2, cv_means[i, 0] + cv_stds[i, 0] + 0.01,
                f'{cv_means[i, 0]:.3f}', ha='center', fontsize=9)
        ax.text(i + w/2, cv_means[i, 1] + cv_stds[i, 1] + 0.01,
                f'{cv_means[i, 1]:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_classification.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 5. LDA explained variance comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, lr in lda_results.items():
        evr = lr['lda'].explained_variance_ratio_
        cum = np.cumsum(evr)
        ax.plot(range(1, len(cum)+1), cum, 'o-', lw=2, ms=6, label=name)
    ax.axhline(0.95, color='gray', ls='--', lw=1.5, label='95% threshold')
    ax.set_xlabel('Number of Linear Discriminants', fontweight='bold')
    ax.set_ylabel('Cumulative Between-Class Variance', fontweight='bold')
    ax.set_title('LDA Dimensionality Comparison', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_lda_cumvar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 6. Confusion-based RDM comparison ---
    fig, axes = plt.subplots(1, n_methods, figsize=(9 * n_methods, 8))
    if n_methods == 1: axes = [axes]

    for idx, (name, data) in enumerate(methods.items()):
        X, y = data['X'], data['y']
        classes = np.unique(y)
        n_c = len(classes)
        cm_acc = np.zeros((n_c, n_c))
        for seed in range(10):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
            clf = LinearSVC(C=1.0, dual=True, max_iter=5000)
            clf.fit(Xtr_s, ytr)
            yp = clf.predict(Xte_s)
            cm = confusion_matrix(yte, yp, labels=classes)
            cm_acc += cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_avg = cm_acc / 10
        dissim = 1.0 - 0.5 * (cm_avg + cm_avg.T)

        ax = axes[idx]
        im = ax.imshow(dissim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(n_c)); ax.set_yticks(range(n_c))
        ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')

    fig.suptitle('Confusion-Based RDM Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_confusion_rdm.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Comparison outputs saved to: {out_dir}")
    print(f"  Files: comparison_lda_scatter.png, comparison_lda_rdm.png,")
    print(f"         comparison_variance_ratio.png, comparison_classification.png,")
    print(f"         comparison_lda_cumvar.png, comparison_confusion_rdm.png")

    return lda_results


# =============================================================================
# Usage Example
# =============================================================================

# if __name__ == "__main__":
    # print("""
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  USAGE EXAMPLES                                                     ║
    # ╠═══════════════════════════════════════════════════════════════════════╣
    # ║                                                                     ║
    # ║  from movement_representation_analysis import (                     ║
    # ║      run_full_analysis, compare_methods                             ║
    # ║  )                                                                  ║
    # ║                                                                     ║
    # ║  # --- Per-method analysis ---                                      ║
    # ║                                                                     ║
    # ║  res_tmp = run_full_analysis(                                       ║
    # ║      X_tmp, y_tmp,                                                  ║
    # ║      out_dir='./results/tmp_analysis',                              ║
    # ║      method_name='TMP Weights',                                     ║
    # ║      feature_structure={                                            ║
    # ║          'n_signals': 51,                                           ║
    # ║          'n_features_per_signal': 5,                                ║
    # ║          'signal_label': 'Joint Coordinate',                        ║
    # ║          'feature_label': 'Primitive Weight',                       ║
    # ║      }                                                              ║
    # ║  )                                                                  ║
    # ║                                                                     ║
    # ║  res_leg = run_full_analysis(                                       ║
    # ║      X_leg, y_leg,                                                  ║
    # ║      out_dir='./results/legendre_analysis',                         ║
    # ║      method_name='Legendre Coefficients',                           ║
    # ║      feature_structure={                                            ║
    # ║          'n_signals': 51,                                           ║
    # ║          'n_features_per_signal': max_degree + 1,                   ║
    # ║          'signal_label': 'Joint Coordinate',                        ║
    # ║          'feature_label': 'Polynomial Degree',                      ║
    # ║      }                                                              ║
    # ║  )                                                                  ║
    # ║                                                                     ║
    # ║  res_ae = run_full_analysis(                                        ║
    # ║      X_ae, y_ae,                                                    ║
    # ║      out_dir='./results/autoencoder_analysis',                      ║
    # ║      method_name='Autoencoder Latent',                              ║
    # ║      feature_structure=None,  # latent dims have no structure       ║
    # ║  )                                                                  ║
    # ║                                                                     ║
    # ║  # --- Cross-method comparison ---                                  ║
    # ║                                                                     ║
    # ║  compare_methods(                                                   ║
    # ║      methods={                                                      ║
    # ║          'TMP Weights':           {'X': X_tmp, 'y': y_tmp},         ║
    # ║          'Legendre Coefficients': {'X': X_leg, 'y': y_leg},         ║
    # ║          'Autoencoder Latent':    {'X': X_ae,  'y': y_ae},          ║
    # ║      },                                                             ║
    # ║      out_dir='./results/method_comparison'                          ║
    # ║  )                                                                  ║
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # """)

    # --- Quick demo with synthetic data ---
    # np.random.seed(42)
    # n_per_class, n_classes = 80, 12
    #
    # def _make_data(n_features, separation=0.3, noise=2.0):
    #     X, y = [], []
    #     for c in range(n_classes):
    #         shift = np.random.randn(n_features) * separation
    #         X.append(np.random.randn(n_per_class, n_features) * noise + shift)
    #         y.extend([c] * n_per_class)
    #     return np.vstack(X), np.array(y)
    #
    # X_tmp, y_tmp = _make_data(255, separation=0.5)   # 51 joints × 5 primitives
    # X_leg, y_leg = _make_data(102, separation=0.3)    # 51 joints × 2 coefficients
    # X_ae, y_ae   = _make_data(64,  separation=0.4)    # 64 latent dims
    #
    # print("Running per-method analyses...")
    # run_full_analysis(X_tmp, y_tmp, '/tmp/demo/tmp',
    #                   method_name='TMP Weights',
    #                   feature_structure={'n_signals': 51, 'n_features_per_signal': 5,
    #                                      'signal_label': 'Joint', 'feature_label': 'Primitive'})
    #
    # run_full_analysis(X_leg, y_leg, '/tmp/demo/legendre',
    #                   method_name='Legendre Coefficients',
    #                   feature_structure={'n_signals': 51, 'n_features_per_signal': 2,
    #                                      'signal_label': 'Joint', 'feature_label': 'Degree'})
    #
    # run_full_analysis(X_ae, y_ae, '/tmp/demo/autoencoder',
    #                   method_name='Autoencoder Latent')
    #
    # print("\nRunning cross-method comparison...")
    # compare_methods(
    #     methods={
    #         'TMP Weights':           {'X': X_tmp, 'y': y_tmp},
    #         'Legendre Coefficients': {'X': X_leg, 'y': y_leg},
    #         'Autoencoder Latent':    {'X': X_ae,  'y': y_ae},
    #     },
    #     out_dir='/tmp/demo/comparison'
    # )