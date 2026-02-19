"""
Posture-Removal Experiment: Fair Comparison of TMP vs Legendre vs Autoencoder
=============================================================================

Core idea: Position-data Legendre degree-0 encodes the TIME-AVERAGED POSTURE,
which is hugely discriminative because different activities have different
average body configurations. This gives Legendre an unfair advantage -
it "wins" by ignoring dynamics entirely.

This script:
1. Runs Legendre analysis on RAW data (posture + dynamics) across degrees 1-9
2. Runs Legendre analysis on MEAN-SUBTRACTED data (dynamics only) across degrees 1-9
3. Loads TMP weights for comparison
4. Generates comprehensive comparison figures

Usage:
    from posture_removal_experiment import run_posture_removal_experiment

    run_posture_removal_experiment(
        processed_segments=processed_segments,
        segment_motion_ids=segment_motion_ids,
        out_dir='./results/posture_experiment',
        tmp_weights=tmp_weights_X,  # or None
    )
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import special
from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# =========================================================================
# Legendre basis functions
# =========================================================================

def shifted_legendre_polynomial(degree, r):
    x = 2 * r - 1
    return special.eval_legendre(degree, x)


def generate_legendre_basis(max_degree, time_points):
    basis = np.zeros((len(time_points), max_degree + 1))
    for i in range(max_degree + 1):
        basis[:, i] = shifted_legendre_polynomial(i, time_points)
    return basis


def fit_legendre_polynomials(data, max_degree):
    coefficients = []
    for segment in data:
        joints, time_steps = segment.shape
        time_normalized = np.linspace(0, 1, time_steps)
        basis = generate_legendre_basis(max_degree, time_normalized)
        segment_coeffs = np.zeros((joints, max_degree + 1))
        for j in range(joints):
            segment_coeffs[j] = np.linalg.lstsq(basis, segment[j], rcond=None)[0]
        coefficients.append(segment_coeffs)
    return coefficients


# =========================================================================
# Mean subtraction (the core of Option A)
# =========================================================================

def subtract_segment_means(processed_segments):
    """
    Remove per-segment, per-joint temporal mean.

    For each segment [signals, time], subtract the time-average of each signal.
    After this, degree-0 Legendre coefficient ~ 0 for all joints,
    so the representation only captures DYNAMICS, not posture.
    """
    mean_subtracted = []
    segment_means = []
    for seg in processed_segments:
        seg_mean = seg.mean(axis=1, keepdims=True)  # [signals, 1]
        mean_subtracted.append(seg - seg_mean)
        segment_means.append(seg_mean.squeeze())
    return mean_subtracted, segment_means


# =========================================================================
# Classification helper
# =========================================================================

def classify_and_evaluate(X, y, method_name='', n_cv_folds=5, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LinearSVC(C=1.0, dual=True, max_iter=5000)
    clf.fit(X_train_s, y_train)

    train_acc = clf.score(X_train_s, y_train)
    test_acc = clf.score(X_test_s, y_test)

    cv_scores = cross_val_score(
        LinearSVC(C=1.0, dual=True, max_iter=5000),
        X_train_s, y_train, cv=n_cv_folds, scoring='accuracy'
    )

    return {
        'method': method_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
    }


# =========================================================================
# Degree-0 analysis: What does it actually encode?
# =========================================================================

def analyze_degree0_content(processed_segments, segment_motion_ids, out_dir):
    """
    Show that degree-0 coefficients on raw position data
    ARE the average posture, and that this alone separates classes.
    """
    out_dir = Path(out_dir) / 'degree0_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = np.unique(segment_motion_ids)
    n_classes = len(classes)

    # Compute per-class average posture
    class_means = {}
    for cls in classes:
        mask = segment_motion_ids == cls
        segs = [processed_segments[i] for i in range(len(processed_segments)) if mask[i]]
        postures = np.array([seg.mean(axis=1) for seg in segs])
        class_means[cls] = postures.mean(axis=0)

    # Plot: average posture profile per class
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap = plt.cm.tab20(np.linspace(0, 1, n_classes))
    for i, cls in enumerate(classes):
        ax.plot(class_means[cls], color=cmap[i], alpha=0.8, linewidth=1.5, label=f'Motion {cls}')

    ax.set_xlabel('Joint Coordinate Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Position Value', fontsize=12, fontweight='bold')
    ax.set_title('Average Posture Per Activity\n(= Legendre degree-0 coefficient per joint)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / 'average_posture_per_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Classify using ONLY degree-0 (= segment means)
    X_posture = np.array([seg.mean(axis=1) for seg in processed_segments])
    result_posture = classify_and_evaluate(
        X_posture, segment_motion_ids, method_name='Posture Only (degree-0)'
    )

    print(f"\n  POSTURE-ONLY classification (just time-averaged position):")
    print(f"    Accuracy: {result_posture['test_acc']:.4f}")
    print(f"    CV:       {result_posture['cv_mean']:.4f} +/- {result_posture['cv_std']:.4f}")
    print(f"    Features: {result_posture['n_features']}")
    print(f"    --> This is what Legendre degree-0 captures!")

    return result_posture


# =========================================================================
# Degree sweep
# =========================================================================

def run_degree_sweep(processed_segments, segment_motion_ids, max_degrees=None, label='Raw'):
    if max_degrees is None:
        max_degrees = list(range(1, 10))

    results = []
    for deg in max_degrees:
        coeffs = fit_legendre_polynomials(processed_segments, deg)
        X = np.array([c.flatten() for c in coeffs])
        y = segment_motion_ids

        res = classify_and_evaluate(X, y, method_name=f'{label} deg={deg}')
        res['degree'] = deg
        results.append(res)

        print(f"    {label} degree={deg}: "
              f"test={res['test_acc']:.4f}, "
              f"CV={res['cv_mean']:.4f}+/-{res['cv_std']:.4f}, "
              f"dim={res['n_features']}")

    return results


# =========================================================================
# Main experiment
# =========================================================================

def run_posture_removal_experiment(
        processed_segments,
        segment_motion_ids,
        out_dir,
        tmp_weights=None,
        ae_latents=None,
        num_signals=None,
        max_degrees=None,
):
    """
    Full posture-removal experiment.

    Parameters
    ----------
    processed_segments : list of ndarray [signals, time]
    segment_motion_ids : ndarray of class labels
    out_dir : str
    tmp_weights : ndarray (n_segments, n_features) or None
    ae_latents : ndarray (n_segments, n_features) or None
    num_signals : int or None
    max_degrees : list of int or None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_degrees is None:
        max_degrees = list(range(1, 10))

    if num_signals is None:
        num_signals = processed_segments[0].shape[0]

    segment_motion_ids = np.array(segment_motion_ids)

    print("=" * 70)
    print("  POSTURE-REMOVAL EXPERIMENT")
    print("=" * 70)
    print(f"  Segments: {len(processed_segments)}")
    print(f"  Signals:  {num_signals}")
    print(f"  Classes:  {len(np.unique(segment_motion_ids))}")
    print(f"  Degrees:  {max_degrees}")
    print("=" * 70)

    # Step 1: Analyze what degree-0 actually is
    print("\n[1/5] Analyzing degree-0 content...")
    posture_result = analyze_degree0_content(processed_segments, segment_motion_ids, out_dir)

    # Step 2: Mean-subtract
    print("\n[2/5] Mean-subtracting segments...")
    ms_segments, seg_means = subtract_segment_means(processed_segments)

    test_coeffs = fit_legendre_polynomials(ms_segments[:5], 1)
    deg0_magnitude = np.mean([np.abs(c[:, 0]).mean() for c in test_coeffs])
    print(f"  Mean degree-0 magnitude after subtraction: {deg0_magnitude:.6f} (should be ~0)")

    # Step 3: Degree sweep on raw data
    print("\n[3/5] Degree sweep on RAW data (posture + dynamics)...")
    raw_results = run_degree_sweep(processed_segments, segment_motion_ids, max_degrees, label='Raw')

    # Step 4: Degree sweep on mean-subtracted data
    print("\n[4/5] Degree sweep on MEAN-SUBTRACTED data (dynamics only)...")
    ms_results = run_degree_sweep(ms_segments, segment_motion_ids, max_degrees, label='MeanSub')

    # Step 5: Collect TMP / AE results
    print("\n[5/5] Collecting comparison method results...")
    comparison_results = {}

    if tmp_weights is not None:
        tmp_res = classify_and_evaluate(tmp_weights, segment_motion_ids, method_name='TMP Weights')
        comparison_results['TMP Weights'] = tmp_res
        print(f"  TMP Weights: test={tmp_res['test_acc']:.4f}, "
              f"CV={tmp_res['cv_mean']:.4f}+/-{tmp_res['cv_std']:.4f}, dim={tmp_res['n_features']}")

    if ae_latents is not None:
        ae_res = classify_and_evaluate(ae_latents, segment_motion_ids, method_name='Autoencoder')
        comparison_results['Autoencoder'] = ae_res
        print(f"  Autoencoder: test={ae_res['test_acc']:.4f}, "
              f"CV={ae_res['cv_mean']:.4f}+/-{ae_res['cv_std']:.4f}, dim={ae_res['n_features']}")

    # Generate all figures
    print("\nGenerating figures...")
    _plot_main_comparison(raw_results, ms_results, comparison_results, posture_result, out_dir)
    _plot_accuracy_vs_dimension(raw_results, ms_results, comparison_results, out_dir)
    _plot_information_decomposition(raw_results, ms_results, posture_result, out_dir)
    _plot_generalization_gap_comparison(raw_results, ms_results, out_dir)
    _plot_lda_comparison(processed_segments, ms_segments, segment_motion_ids,
                         tmp_weights, out_dir)

    _print_summary_table(raw_results, ms_results, comparison_results, posture_result)
    _save_results(raw_results, ms_results, comparison_results, posture_result, out_dir)

    print(f"\n  All outputs saved to: {out_dir}")
    return raw_results, ms_results, comparison_results


# =========================================================================
# Plotting
# =========================================================================

def _plot_main_comparison(raw_results, ms_results, comp_results, posture_res, out_dir):
    """Key figure: Raw vs Mean-Subtracted accuracy with TMP reference."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    degrees = [r['degree'] for r in raw_results]

    # Left: Test accuracy
    ax = axes[0]
    raw_test = [r['test_acc'] for r in raw_results]
    ms_test = [r['test_acc'] for r in ms_results]

    ax.plot(degrees, raw_test, 'o-', color='#E53935', lw=2.5, ms=9,
            label='Legendre (raw = posture + dynamics)', zorder=5)
    ax.plot(degrees, ms_test, 's-', color='#1E88E5', lw=2.5, ms=9,
            label='Legendre (mean-subtracted = dynamics only)', zorder=5)
    ax.axhline(posture_res['test_acc'], color='#FF9800', ls=':', lw=2.5,
               label=f"Posture only (degree-0) = {posture_res['test_acc']:.3f}", zorder=4)

    colors_comp = {'TMP Weights': '#4CAF50', 'Autoencoder': '#9C27B0'}
    for name, res in comp_results.items():
        ax.axhline(res['test_acc'], color=colors_comp.get(name, 'gray'), ls='--', lw=2.5,
                   label=f"{name} = {res['test_acc']:.3f} (dim={res['n_features']})", zorder=4)

    ax.set_xlabel('Legendre Max Degree', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Classification: Raw vs Dynamics-Only', fontsize=14, fontweight='bold')
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(bottom=min(min(raw_test), min(ms_test)) - 0.03)

    # Right: CV accuracy
    ax = axes[1]
    raw_cv = [r['cv_mean'] for r in raw_results]
    raw_cv_std = [r['cv_std'] for r in raw_results]
    ms_cv = [r['cv_mean'] for r in ms_results]
    ms_cv_std = [r['cv_std'] for r in ms_results]

    ax.errorbar(degrees, raw_cv, yerr=raw_cv_std, fmt='o-', color='#E53935',
                lw=2.5, ms=9, capsize=4, label='Legendre (raw)', zorder=5)
    ax.errorbar(degrees, ms_cv, yerr=ms_cv_std, fmt='s-', color='#1E88E5',
                lw=2.5, ms=9, capsize=4, label='Legendre (mean-subtracted)', zorder=5)

    for name, res in comp_results.items():
        ax.axhline(res['cv_mean'], color=colors_comp.get(name, 'gray'), ls='--', lw=2.5,
                   label=f"{name} CV = {res['cv_mean']:.3f}+/-{res['cv_std']:.3f}", zorder=4)

    ax.set_xlabel('Legendre Max Degree', fontsize=13, fontweight='bold')
    ax.set_ylabel('5-Fold CV Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Cross-Validation: Raw vs Dynamics-Only', fontsize=14, fontweight='bold')
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    fig.suptitle('The Posture Bias: Removing Mean Position Reveals True Dynamic Discriminability',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / 'main_comparison_raw_vs_meansub.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: main_comparison_raw_vs_meansub.png")


def _plot_accuracy_vs_dimension(raw_results, ms_results, comp_results, out_dir):
    """Accuracy vs feature dimensionality."""
    fig, ax = plt.subplots(figsize=(12, 7))

    raw_dims = [r['n_features'] for r in raw_results]
    raw_acc = [r['test_acc'] for r in raw_results]
    raw_degs = [r['degree'] for r in raw_results]

    ax.plot(raw_dims, raw_acc, 'o-', color='#E53935', lw=2.5, ms=10,
            label='Legendre (raw)', zorder=5)
    for d, dim, acc in zip(raw_degs, raw_dims, raw_acc):
        ax.annotate(f'd={d}', (dim, acc), fontsize=8, fontweight='bold',
                    xytext=(5, 8), textcoords='offset points')

    ms_dims = [r['n_features'] for r in ms_results]
    ms_acc = [r['test_acc'] for r in ms_results]
    ms_degs = [r['degree'] for r in ms_results]

    ax.plot(ms_dims, ms_acc, 's-', color='#1E88E5', lw=2.5, ms=10,
            label='Legendre (dynamics only)', zorder=5)
    for d, dim, acc in zip(ms_degs, ms_dims, ms_acc):
        ax.annotate(f'd={d}', (dim, acc), fontsize=8, fontweight='bold',
                    xytext=(5, -15), textcoords='offset points')

    markers = {'TMP Weights': ('*', '#4CAF50', 300), 'Autoencoder': ('D', '#9C27B0', 150)}
    for name, res in comp_results.items():
        marker, color, size = markers.get(name, ('o', 'gray', 100))
        ax.scatter(res['n_features'], res['test_acc'], marker=marker, c=color,
                   s=size, edgecolors='black', linewidths=2, zorder=10,
                   label=f"{name} (dim={res['n_features']}, acc={res['test_acc']:.3f})")

    ax.set_xlabel('Feature Dimensionality', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Feature Efficiency: Accuracy vs Dimensionality\n'
                 '(Higher accuracy at lower dimension = more efficient)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / 'accuracy_vs_dimension.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: accuracy_vs_dimension.png")


def _plot_information_decomposition(raw_results, ms_results, posture_res, out_dir):
    """Bar chart: how much accuracy comes from posture vs dynamics."""
    fig, ax = plt.subplots(figsize=(14, 7))

    degrees = [r['degree'] for r in raw_results]
    raw_acc = np.array([r['test_acc'] for r in raw_results])
    ms_acc = np.array([r['test_acc'] for r in ms_results])
    posture_acc = posture_res['test_acc']

    x = np.arange(len(degrees))
    width = 0.35

    ax.bar(x - width / 2, raw_acc, width, color='#E53935', alpha=0.8,
           edgecolor='black', label='Raw Legendre (total)')
    ax.bar(x + width / 2, ms_acc, width, color='#1E88E5', alpha=0.8,
           edgecolor='black', label='Mean-subtracted (dynamics only)')
    ax.axhline(posture_acc, color='#FF9800', ls='--', lw=2.5,
               label=f'Posture alone = {posture_acc:.3f}')

    for i, (ra, ma) in enumerate(zip(raw_acc, ms_acc)):
        gap = ra - ma
        if abs(gap) > 0.005:
            ax.annotate(f'{gap:+.2f}',
                        xy=(x[i], max(ra, ma) + 0.005),
                        ha='center', fontsize=8, color='#333', fontweight='bold')

    ax.set_xlabel('Legendre Max Degree', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Information Decomposition: Posture vs Dynamics\n'
                 'Gap between red and blue = contribution of posture information',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(degrees)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=max(0.5, min(min(raw_acc), min(ms_acc)) - 0.05))
    plt.tight_layout()
    plt.savefig(out_dir / 'information_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: information_decomposition.png")


def _plot_generalization_gap_comparison(raw_results, ms_results, out_dir):
    """Compare overfitting between raw and mean-subtracted."""
    fig, ax = plt.subplots(figsize=(12, 6))

    degrees = [r['degree'] for r in raw_results]
    raw_gap = [r['train_acc'] - r['test_acc'] for r in raw_results]
    ms_gap = [r['train_acc'] - r['test_acc'] for r in ms_results]

    ax.plot(degrees, raw_gap, 'o-', color='#E53935', lw=2.5, ms=9,
            label='Raw Legendre')
    ax.plot(degrees, ms_gap, 's-', color='#1E88E5', lw=2.5, ms=9,
            label='Mean-subtracted Legendre')

    ax.set_xlabel('Max Degree', fontsize=13, fontweight='bold')
    ax.set_ylabel('Train - Test Accuracy (overfitting)', fontsize=13, fontweight='bold')
    ax.set_title('Generalization Gap: Raw vs Dynamics-Only',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(degrees)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / 'generalization_gap_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: generalization_gap_comparison.png")


def _plot_lda_comparison(raw_segments, ms_segments, labels, tmp_weights, out_dir):
    """Side-by-side LDA: Raw deg-1, Mean-sub deg-1, TMP weights."""
    n_panels = 2 + (1 if tmp_weights is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7))
    if n_panels == 1:
        axes = [axes]

    classes = np.unique(labels)
    n_classes = len(classes)
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

    datasets = []

    raw_coeffs = fit_legendre_polynomials(raw_segments, 1)
    X_raw = np.array([c.flatten() for c in raw_coeffs])
    datasets.append(('Legendre d=1\n(posture + dynamics)', X_raw))

    ms_coeffs = fit_legendre_polynomials(ms_segments, 1)
    X_ms = np.array([c.flatten() for c in ms_coeffs])
    datasets.append(('Legendre d=1\n(dynamics only)', X_ms))

    if tmp_weights is not None:
        datasets.append(('TMP Weights', tmp_weights))

    for idx, (title, X) in enumerate(datasets):
        ax = axes[idx]
        Xs = StandardScaler().fit_transform(X)
        nc = min(n_classes - 1, X.shape[1])
        lda = LinearDiscriminantAnalysis(n_components=nc)
        Xl = lda.fit_transform(Xs, labels)
        evr = lda.explained_variance_ratio_

        for ci, cls in enumerate(classes):
            m = labels == cls
            ax.scatter(Xl[m, 0], Xl[m, 1], c=[colors[ci]], alpha=0.35, s=20)
            cx, cy = Xl[m, 0].mean(), Xl[m, 1].mean()
            ax.scatter(cx, cy, c=[colors[ci]], s=200, marker='X',
                       edgecolors='black', linewidths=2, zorder=10)
            ax.annotate(str(cls), (cx, cy), fontsize=8, fontweight='bold',
                        ha='center', va='bottom', xytext=(0, 6), textcoords='offset points')

        ax.set_xlabel(f'LD1 ({evr[0]:.1%})', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'LD2 ({evr[1]:.1%})', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.suptitle('LDA Projection: Posture-Based vs Dynamics-Based Representations',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / 'lda_comparison_raw_vs_meansub.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: lda_comparison_raw_vs_meansub.png")


# =========================================================================
# Summary
# =========================================================================

def _print_summary_table(raw_results, ms_results, comp_results, posture_res):
    print(f"\n{'=' * 85}")
    print(f"  SUMMARY TABLE")
    print(f"{'=' * 85}")
    print(f"  {'Method':<40} {'Dim':>5} {'Test':>7} {'CV Mean':>8} {'CV Std':>7} {'Gap':>6}")
    print(f"  {'-'*75}")

    r = posture_res
    print(f"  {'Posture Only (seg. mean)':<40} {r['n_features']:>5} "
          f"{r['test_acc']:>7.4f} {r['cv_mean']:>8.4f} {r['cv_std']:>7.4f} "
          f"{r['train_acc']-r['test_acc']:>6.3f}")
    print(f"  {'-'*75}")

    for r in raw_results:
        print(f"  {'Raw Legendre deg=' + str(r['degree']):<40} {r['n_features']:>5} "
              f"{r['test_acc']:>7.4f} {r['cv_mean']:>8.4f} {r['cv_std']:>7.4f} "
              f"{r['train_acc']-r['test_acc']:>6.3f}")
    print(f"  {'-'*75}")

    for r in ms_results:
        print(f"  {'Dynamics-only Legendre deg=' + str(r['degree']):<40} {r['n_features']:>5} "
              f"{r['test_acc']:>7.4f} {r['cv_mean']:>8.4f} {r['cv_std']:>7.4f} "
              f"{r['train_acc']-r['test_acc']:>6.3f}")
    print(f"  {'-'*75}")

    for name, r in comp_results.items():
        print(f"  {name:<40} {r['n_features']:>5} "
              f"{r['test_acc']:>7.4f} {r['cv_mean']:>8.4f} {r['cv_std']:>7.4f} "
              f"{r['train_acc']-r['test_acc']:>6.3f}")
    print(f"{'=' * 85}\n")


def _save_results(raw_results, ms_results, comp_results, posture_res, out_dir):
    path = out_dir / 'experiment_results.txt'
    with open(path, 'w') as f:
        f.write("POSTURE-REMOVAL EXPERIMENT RESULTS\n")
        f.write("=" * 85 + "\n\n")

        best_raw = max(raw_results, key=lambda r: r['test_acc'])
        best_ms = max(ms_results, key=lambda r: r['test_acc'])
        f.write("KEY FINDING:\n")
        f.write(f"  Best raw Legendre:          deg={best_raw['degree']}, "
                f"acc={best_raw['test_acc']:.4f}\n")
        f.write(f"  Best dynamics-only Legendre: deg={best_ms['degree']}, "
                f"acc={best_ms['test_acc']:.4f}\n")
        f.write(f"  Posture-only accuracy:       {posture_res['test_acc']:.4f}\n")
        f.write(f"  Accuracy drop from removing posture: "
                f"{best_raw['test_acc'] - best_ms['test_acc']:.4f}\n\n")

        for name, r in comp_results.items():
            f.write(f"  {name}: acc={r['test_acc']:.4f}, dim={r['n_features']}\n")

        f.write(f"\n\nDETAILED RESULTS:\n")
        f.write(f"{'Method':<40} {'Dim':>5} {'Test':>7} {'CV':>7} {'Gap':>6}\n")
        f.write("-" * 70 + "\n")

        all_res = (
            [('Posture only', posture_res)] +
            [(f'Raw deg={r["degree"]}', r) for r in raw_results] +
            [(f'DynOnly deg={r["degree"]}', r) for r in ms_results] +
            list(comp_results.items())
        )
        for name, r in all_res:
            f.write(f"{name:<40} {r['n_features']:>5} "
                    f"{r['test_acc']:>7.4f} {r['cv_mean']:>7.4f} "
                    f"{r['train_acc']-r['test_acc']:>6.3f}\n")

    print(f"  Saved: experiment_results.txt")


# =========================================================================
# Demo
# =========================================================================

if __name__ == "__main__":
    print("Running synthetic demo...\n")
    np.random.seed(42)

    n_classes, n_per_class, n_signals = 12, 80, 48

    segments = []
    labels = []
    for cls in range(n_classes):
        class_posture = np.random.randn(n_signals) * 3 + cls * 0.5
        for _ in range(n_per_class):
            t_len = np.random.randint(40, 80)
            t = np.linspace(0, 2 * np.pi, t_len)
            dynamics = 0.3 * np.outer(np.random.randn(n_signals), np.sin(t + np.random.rand() * np.pi))
            noise = np.random.randn(n_signals, t_len) * 0.2
            seg = class_posture[:, None] + dynamics + noise
            segments.append(seg)
            labels.append(cls)
    labels = np.array(labels)

    run_posture_removal_experiment(
        processed_segments=segments,
        segment_motion_ids=labels,
        out_dir='/tmp/posture_demo',
        max_degrees=list(range(1, 7)),
    )