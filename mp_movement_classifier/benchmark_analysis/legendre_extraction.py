
from scipy import special
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from pathlib import Path
from mp_movement_classifier.utils import config
from mp_movement_classifier.classification.utils import calculate_rdm
from mp_movement_classifier.classification.classification_pipeline import run_classification_pipeline
from mp_movement_classifier.tmp_extraction.weight_visulaization import (extract_and_save_avg_weights_for_motions,
                                                                        load_motion_mapping,
                                                                        weights_barplot_across_channels)
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,

)
from mp_movement_classifier.benchmark_analysis.lda_analysis import run_lda_analysis
# from posture_removal_experiment import run_posture_removal_experiment



def shifted_legendre_polynomial(degree, r):
    """
    Compute shifted Legendre polynomial of specified degree at point r.

    Parameters:
    - degree: int, degree of the polynomial
    - r: float or array, points where to evaluate the polynomial in range [0, 1]

    Returns:
    - values of the shifted Legendre polynomial at r
    """
    x = 2 * r - 1  # Map from [0, 1] to [-1, 1]
    return special.eval_legendre(degree, x)


def generate_legendre_basis(max_degree, time_points):
    """
    Generate basis of Legendre polynomials up to max_degree.

    Parameters:
    - max_degree: int, maximum degree of polynomials
    - time_points: array, normalized time points in [0, 1]

    Returns:
    - basis: array of shape (len(time_points), max_degree + 1)
    """
    basis = np.zeros((len(time_points), max_degree + 1))
    for i in range(max_degree + 1):
        basis[:, i] = shifted_legendre_polynomial(i, time_points)
    return basis


def fit_legendre_polynomials(data, max_degree):
    """
    Fit Legendre polynomials to movement data.

    Parameters:
    - data: list of arrays, where each array has shape [joints, time]
    - max_degree: int, maximum degree of Legendre polynomials to use

    Returns:
    - coefficients: list of arrays, each with shape [joints, max_degree + 1]
    """
    coefficients = []

    for segment in data:
        joints, time_steps = segment.shape

        # Normalize time to [0, 1]
        time_normalized = np.linspace(0, 1, time_steps)

        # Generate Legendre basis
        basis = generate_legendre_basis(max_degree, time_normalized)

        # Solve for coefficients using least squares
        segment_coeffs = np.zeros((joints, max_degree + 1))
        for j in range(joints):
            segment_coeffs[j] = np.linalg.lstsq(basis, segment[j], rcond=None)[0]

        coefficients.append(segment_coeffs)

    return coefficients


def reconstruct_from_coefficients(coefficients, time_steps, max_degree):
    """
    Reconstruct movement data from Legendre coefficients.

    Parameters:
    - coefficients: array of shape [joints, max_degree + 1]
    - time_steps: int, number of time steps for reconstruction
    - max_degree: int, maximum degree of Legendre polynomials used

    Returns:
    - reconstructed: array of shape [joints, time_steps]
    """
    joints = coefficients.shape[0]

    # Normalize time to [0, 1]
    time_normalized = np.linspace(0, 1, time_steps)

    # Generate Legendre basis
    basis = generate_legendre_basis(max_degree, time_normalized)

    # Reconstruct
    reconstructed = np.zeros((joints, time_steps))
    for j in range(joints):
        reconstructed[j] = np.dot(basis, coefficients[j])

    return reconstructed


def process_with_legendre_basis(processed_data, max_degree):
    """
    Process all movement segments using Legendre polynomial basis.

    Parameters:
    - processed_data: list of arrays, each with shape [joints, time]
    - max_degree: int, maximum degree of Legendre polynomials to use

    Returns:
    - coefficients: list of arrays, each with shape [joints, max_degree + 1]
    - reconstruction_error: list of errors for each segment
    """
    coefficients = fit_legendre_polynomials(processed_data, max_degree)

    # Calculate reconstruction error for each segment
    reconstruction_error = []
    for i, segment in enumerate(processed_data):
        joints, time_steps = segment.shape
        reconstructed = reconstruct_from_coefficients(coefficients[i], time_steps, max_degree)
        error = np.mean((segment - reconstructed) ** 2)
        reconstruction_error.append(error)

    return coefficients, reconstruction_error


def plot_coefficient_distributions(coefficients, motion_ids,
                                   save_dir):
    """
    Create separate figures for each motion ID.
    """
    save_dir = Path(save_dir) / 'coefficient_plots'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Group coefficients
    coef_by_motion = defaultdict(list)
    for coef, motion_id in zip(coefficients, motion_ids):
        coef_by_motion[motion_id].append(coef)

    unique_motion_ids = sorted(coef_by_motion.keys())
    n_dims = coefficients[0].shape[1]
    n_signals = coefficients[0].shape[0]

    figures = {}

    for motion_id in unique_motion_ids:
        # Create figure for this motion_id
        # fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig, axes = plt.subplots(3, 4, figsize=(15, 8))
        axes = axes.flatten()

        motion_coefs = np.array(coef_by_motion[motion_id])
        n_samples = len(motion_coefs)

        for dim in range(n_dims):
            ax = axes[dim]

            # Extract and compute statistics
            dim_coefs = motion_coefs[:, :, dim]
            means = np.mean(dim_coefs, axis=0)
            stds = np.std(dim_coefs, axis=0)

            # Plot
            signal_indices = np.arange(n_signals)
            ax.bar(signal_indices, means, yerr=stds,
                   capsize=3, alpha=0.7,
                   error_kw={'linewidth': 1, 'capthick': 1})

            ax.set_xlabel('Joint coordinate', fontsize=10)
            ax.set_ylabel('Coefficient Value', fontsize=10)
            ax.set_title(f'Polynomial degree {dim}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)

        fig.suptitle(f'Motion ID: {motion_id} (n={n_samples} samples)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = Path(save_dir) / f'motion_{motion_id}_coefficients.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        figures[motion_id] = fig


    return figures


def prepare_coefficient_data(coefficients, motion_ids):
    X = np.array([coef.flatten() for coef in coefficients])
    y = np.array(motion_ids)

    return X, y


def visualize_with_pca(X, y, out_dir):
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='hsv', alpha=0.8)

    plt.colorbar(scatter, label='Motion Type')

    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.title('PCA of Legendre Polynomial Coefficients')

    unique_motions = np.unique(y)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.hsv(i / len(unique_motions)),
                          markersize=10) for i in range(len(unique_motions))]
    plt.legend(handles, unique_motions, title='Motion Types')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(out_dir) / 'pca_visualization.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    #plot variance explained

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    n_components = min(40, len(pca.explained_variance_ratio_))
    ax1.bar(range(1, n_components + 1),
            pca.explained_variance_ratio_[:n_components],
            alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_[:n_components])
    ax2.plot(range(1, n_components + 1), cumsum,
             marker='o', linewidth=2, markersize=5)
    ax2.axhline(y=0.9, color='r', linestyle='--',
                label='90% variance', linewidth=2)
    ax2.axhline(y=0.95, color='g', linestyle='--',
                label='95% variance', linewidth=2)
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out_path = Path(out_dir) / 'pca_variance_explained.png'
    plt.savefig(out_path, dpi=150)
    plt.close()

    return pca


def visualize_with_tsne(X, y, out_dir):
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
    X_tsne = tsne.fit_transform(X)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='hsv', alpha=0.8)

    # Add a colorbar
    plt.colorbar(scatter, label='Motion Type')

    # Add labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE of Legendre Polynomial Coefficients')

    # Add legend for unique motion types
    unique_motions = np.unique(y)
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.hsv(i / len(unique_motions)),
                          markersize=10) for i in range(len(unique_motions))]
    plt.legend(handles, unique_motions, title='Motion Types')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(out_dir) / 'tsne_visualization.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    return tsne


def classify_motion_types(X, y, out_dir):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)


    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a classifier
    # classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # classifier = SVC(random_state=42)
    classifier = LinearSVC(C=1.0, penalty='l2', dual=True)
    classifier.fit(X_train_scaled, y_train)
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
    # Make predictions
    y_pred = classifier.predict(X_test_scaled)

    # Evaluate the classifier
    accuracy = np.mean(y_pred == y_test)
    print(f"Classification Accuracy: {accuracy:.4f}")
    report = classification_report(y_test, y_pred)
    path = os.path.join(out_dir, "classification_report.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    # Create confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out_path = Path(out_dir) / 'confusion_matrix.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    return classifier, accuracy


def visualize_feature_importance(classifier, X, y, out_dir):
    if hasattr(classifier, 'feature_importances_'):
        # Get feature importances
        importances = classifier.feature_importances_

        # Number of joints and coefficients per joint
        n_joints = X.shape[1] // 10  # Assuming 10 coefficients per joint
        n_coeffs = 10

        # Reshape importances to visualize by joint and coefficient
        imp_reshaped = importances.reshape(n_joints, n_coeffs)

        # Create a heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(imp_reshaped, cmap='hsv', annot=False)
        plt.xlabel('Legendre Coefficient Index')
        plt.ylabel('Joint Index')
        plt.title('Feature Importance by Joint and Coefficient')
        plt.tight_layout()
        out_path = Path(out_dir) / 'feature_importance.png'
        plt.savefig(out_path, dpi=150)
        plt.close()
        # Plot top 20 most important features
        indices = np.argsort(importances)[-20:]
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importances[indices])
        plt.yticks(range(20), [f"Joint {i // n_coeffs}, Coeff {i % n_coeffs}" for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        out_path = Path(out_dir) / 'top_features.png'
        plt.savefig(out_path, dpi=150)
        plt.close()


def analyze_first_degree_coefficients(coefficients, motion_ids, save_dir):
    """
    Analyze first degree polynomial coefficients across motion IDs using PCA.
    """
    save_dir = Path(save_dir) / 'pca_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FIRST DEGREE COEFFICIENT PCA ANALYSIS")
    print("=" * 80)

    # Step 1: Extract first dimension coefficients for all segments
    first_deg_coeffs = []
    motion_id_list = []

    for coef, motion_id in zip(coefficients, motion_ids):
        # Extract first dimension (polynomial degree 0)
        first_deg = coef[:, 0]  # shape: (51,)
        first_deg_coeffs.append(first_deg)
        motion_id_list.append(motion_id)

    first_deg_coeffs = np.array(first_deg_coeffs)  # shape: (n_segments, 51)
    motion_id_list = np.array(motion_id_list)

    print(f"Total segments: {len(first_deg_coeffs)}")
    print(f"Coefficient shape per segment: {first_deg_coeffs[0].shape}")

    # Step 2: Group by motion_id and compute average
    coef_by_motion = defaultdict(list)
    for coef, motion_id in zip(first_deg_coeffs, motion_id_list):
        coef_by_motion[motion_id].append(coef)

    # Compute averaged coefficients for each motion
    unique_motion_ids = sorted(coef_by_motion.keys())
    n_motions = len(unique_motion_ids)
    n_signals = first_deg_coeffs.shape[1]  # 51

    averaged_coeffs = np.zeros((n_motions, n_signals))
    motion_labels = []
    motion_counts = []

    for idx, motion_id in enumerate(unique_motion_ids):
        motion_coefs = np.array(coef_by_motion[motion_id])  # shape: (n_samples, 51)
        averaged_coeffs[idx] = np.mean(motion_coefs, axis=0)  # shape: (51,)
        motion_labels.append(motion_id)
        motion_counts.append(len(motion_coefs))

    print(f"\nUnique motion IDs: {n_motions}")
    print(f"Averaged coefficient matrix shape: {averaged_coeffs.shape}")
    print(f"Motion ID distribution:")
    for motion_id, count in zip(motion_labels, motion_counts):
        print(f"  Motion {motion_id}: {count} segments")

    # Step 3: Standardize features (important for PCA)
    scaler = StandardScaler()
    averaged_coeffs_scaled = scaler.fit_transform(averaged_coeffs)

    # Step 4: Apply PCA
    pca = PCA(n_components=min(n_motions, n_signals))
    pca_result = pca.fit_transform(averaged_coeffs_scaled)

    print(f"\nPCA Results:")
    print(f"  Explained variance ratio (first 5 PCs): {pca.explained_variance_ratio_[:5]}")
    print(f"  Cumulative variance (first 5 PCs): {np.cumsum(pca.explained_variance_ratio_[:5])}")
    print(f"  Total variance explained by PC1 and PC2: {sum(pca.explained_variance_ratio_[:2]):.2%}")

    # Step 5: Visualization
    results = {
        'averaged_coeffs': averaged_coeffs,
        'averaged_coeffs_scaled': averaged_coeffs_scaled,
        'motion_labels': motion_labels,
        'motion_counts': motion_counts,
        'pca': pca,
        'pca_result': pca_result,
        'scaler': scaler
    }

    # Create visualizations
    _plot_pca_2d(results, save_dir)
    _plot_pca_3d(results, save_dir)
    _plot_variance_explained(results, save_dir)
    _plot_feature_importance(results, save_dir)
    _plot_distance_matrix(results, save_dir)

    return results


def _plot_pca_2d(results, save_dir):
    """Create 2D PCA scatter plot."""
    pca_result = results['pca_result']
    motion_labels = results['motion_labels']
    motion_counts = results['motion_counts']
    pca = results['pca']

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create color map
    n_motions = len(motion_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_motions))

    # Plot points
    for idx, (motion_id, count) in enumerate(zip(motion_labels, motion_counts)):
        ax.scatter(pca_result[idx, 0], pca_result[idx, 1],
                   c=[colors[idx]], s=200, alpha=0.7,
                   edgecolors='black', linewidth=2,
                   label=f'Motion {motion_id} (n={count})')

        # Add motion ID as text label
        ax.annotate(str(motion_id),
                    (pca_result[idx, 0], pca_result[idx, 1]),
                    fontsize=10, fontweight='bold',
                    ha='center', va='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                  fontsize=12, fontweight='bold')
    ax.set_title('PCA: Motion Separation Based on First Degree Coefficients',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Legend outside plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_dir / 'pca_2d_motion_separation.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'pca_2d_motion_separation.png'}")


def _plot_pca_3d(results, save_dir):
    """Create 3D PCA scatter plot."""
    from mpl_toolkits.mplot3d import Axes3D

    pca_result = results['pca_result']
    motion_labels = results['motion_labels']
    pca = results['pca']

    if pca_result.shape[1] < 3:
        print("Not enough PCs for 3D plot")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    n_motions = len(motion_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, n_motions))

    for idx, motion_id in enumerate(motion_labels):
        ax.scatter(pca_result[idx, 0],
                   pca_result[idx, 1],
                   pca_result[idx, 2],
                   c=[colors[idx]], s=200, alpha=0.7,
                   edgecolors='black', linewidth=2,
                   label=f'Motion {motion_id}')

        ax.text(pca_result[idx, 0],
                pca_result[idx, 1],
                pca_result[idx, 2],
                str(motion_id), fontsize=10, fontweight='bold')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                  fontsize=10, fontweight='bold')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
                  fontsize=10, fontweight='bold')
    ax.set_title('3D PCA: Motion Separation', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / 'pca_3d_motion_separation.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'pca_3d_motion_separation.png'}")


def _plot_variance_explained(results, save_dir):
    """Plot explained variance by principal components."""
    pca = results['pca']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    n_components = min(20, len(pca.explained_variance_ratio_))
    ax1.bar(range(1, n_components + 1),
            pca.explained_variance_ratio_[:n_components],
            alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_[:n_components])
    ax2.plot(range(1, n_components + 1), cumsum,
             marker='o', linewidth=2, markersize=8)
    ax2.axhline(y=0.8, color='r', linestyle='--',
                label='80% variance', linewidth=2)
    ax2.axhline(y=0.9, color='g', linestyle='--',
                label='90% variance', linewidth=2)
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_dir / 'pca_variance_explained.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'pca_variance_explained.png'}")


def _plot_feature_importance(results, save_dir):
    """Plot feature importance (loading) for PC1 and PC2."""
    pca = results['pca']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    n_signals = pca.components_.shape[1]
    signal_indices = np.arange(n_signals)

    # PC1 loadings
    ax1.bar(signal_indices, pca.components_[0, :],
            alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Joint Coordinate Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loading Value', fontsize=12, fontweight='bold')
    ax1.set_title('PC1 Feature Loadings (Most Important Features)',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)

    # PC2 loadings
    ax2.bar(signal_indices, pca.components_[1, :],
            alpha=0.7, edgecolor='black', color='orange')
    ax2.set_xlabel('Joint Coordinate Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loading Value', fontsize=12, fontweight='bold')
    ax2.set_title('PC2 Feature Loadings', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(save_dir / 'pca_feature_loadings.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'pca_feature_loadings.png'}")


def _plot_distance_matrix(results, save_dir):
    """Plot pairwise distance matrix between motions in PCA space."""
    from scipy.spatial.distance import pdist, squareform

    pca_result = results['pca_result']
    motion_labels = results['motion_labels']

    # Compute pairwise distances using first 2 PCs
    distances = squareform(pdist(pca_result[:, :2], metric='euclidean'))

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(distances, cmap='hsv', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Euclidean Distance in PC Space',
                   fontsize=12, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(len(motion_labels)))
    ax.set_yticks(range(len(motion_labels)))
    ax.set_xticklabels(motion_labels, rotation=45, ha='right')
    ax.set_yticklabels(motion_labels)

    # Add values in cells
    for i in range(len(motion_labels)):
        for j in range(len(motion_labels)):
            text = ax.text(j, i, f'{distances[i, j]:.2f}',
                           ha="center", va="center",
                           color="white" if distances[i, j] > distances.max() / 2 else "black",
                           fontsize=8)

    ax.set_title('Pairwise Distance Matrix Between Motions (PC1-PC2 Space)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Motion ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Motion ID', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / 'pca_distance_matrix.png',
                dpi=300, bbox_inches='tight')

    plt.close()
    print(f"Saved: {save_dir / 'pca_distance_matrix.png'}")


def main():

    num_MPs = 5

    model_dir = os.path.join("./../../results/tmp_configs", f"new_seg_mp_model_{num_MPs}_phase_three")
    out_dir = os.path.join(model_dir, "legendre_analysis")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # folder_path = "./../../data/pymotion_exponential_csv_files"
    folder_path = "./../../data/pymotion_position_csv_files"
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(folder_path=folder_path,
                                                                             data_type = "position",
                                                                             filtering= False)
    num_segments = len(processed_segments)
    print(f"Number of segments: {num_segments}")
    num_signals = processed_segments[0].shape[0]
    print(f"Number of signals: {num_signals}")

    max_degree = 0  # for polynomial degrees as basis function , For 10 degrees (0 to 9)
    coefficients, errors = process_with_legendre_basis(processed_segments, max_degree)
    coefficients_array = np.stack(coefficients, axis=0)
    # coefficients_array shape (num_segments,n_joints, max_degree+1)

    # results = analyze_first_degree_coefficients(
    #     coefficients,
    #     segment_motion_ids,
    #     save_dir=out_dir
    # )

    #save avarage coefficients among each motion

    DEFAULT_MOTION_MAPPING = "../../data/motion_mapping.json"
    motion_id_to_name = load_motion_mapping(DEFAULT_MOTION_MAPPING)
    # avg_weights_dict = extract_and_save_avg_weights_for_motions(
    #     weights=coefficients_array,
    #     motion_ids=segment_motion_ids,
    #     save_dir=os.path.join(out_dir, "averaged_weights"),
    #     motion_names_dict=motion_id_to_name
    # )


    X, y = prepare_coefficient_data(coefficients, segment_motion_ids)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label array shape: {y.shape}")
    print(f" {len(np.unique(y))} unique motion types")

    # Build feature names: deg_k_signal_j
    max_degree_local = max_degree
    # Determine n_signals from coefficients shape: first sample has shape (n_signals, max_degree+1)
    n_signals = coefficients[0].shape[0]
    feature_names = []
    for j in range(n_signals):
        for k in range(max_degree_local + 1):
            feature_names.append(f"deg_{k}_signal_{j}")

    cls_out_dir = os.path.join(out_dir, 'classification')
    run_classification_pipeline(
        X=X,
        y=y,
        out_dir=cls_out_dir,
        feature_names=feature_names,
        feature_structure={'n_signals': n_signals, 'n_features_per_signal': max_degree_local + 1},
        primary_classifier='linear_svc',
        fixed_cm_vmin=0.0,
        fixed_cm_vmax=1.0,
        seed=42,cv_folds=5,
        perform_cv=True
    )

    figures = plot_coefficient_distributions(
        coefficients,
        segment_motion_ids,
        out_dir
    )

    # results_legendre = run_lda_analysis(
    #     X=X, y=y,
    #     out_dir=out_dir,
    #     method_name='Legendre Coefficients',
    #     feature_structure={'n_signals': 48, 'n_features_per_signal': 2},
    #     # feature_structure is optional — used for heatmap layout
    # )


    # results = run_posture_removal_experiment(
    #     processed_segments=processed_segments,
    #     segment_motion_ids=segment_motion_ids,
    #     out_dir=os.path.join(out_dir, "posture_experiment"),
    #     tmp_weights=None,  # or None
    #     ae_latents=None,  # or your AE features
    #     max_degrees=list(range(1, 10)),
    # )

if __name__ == "__main__":
    main()