"""
Script to optimize max_degree parameter for Legendre polynomial fitting.
Tests different polynomial degrees and evaluates classification performance.
"""

from scipy import special
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from pathlib import Path


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


def prepare_coefficient_data(coefficients):
    """Flatten coefficients into feature matrix."""
    X = np.array([coef.flatten() for coef in coefficients])
    return X


def evaluate_classification(X, y, max_degree, random_state=42):
    """
    Train SVM classifier and evaluate performance.

    Parameters:
    - X: feature matrix
    - y: labels
    - max_degree: current max_degree being tested (for logging)
    - random_state: random seed for reproducibility

    Returns:
    - dict with test accuracy, train accuracy, and cross-validation score
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train classifier
    classifier = LinearSVC(C=1.0, penalty='l2', dual=True, random_state=random_state)
    classifier.fit(X_train_scaled, y_train)

    # Evaluate
    train_accuracy = classifier.score(X_train_scaled, y_train)
    test_accuracy = classifier.score(X_test_scaled, y_test)

    # Cross-validation on training set
    cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_scores
    }


def optimize_max_degree(processed_segments, motion_ids, degree_range, out_dir):
    """
    Test different max_degree values and evaluate classification performance.

    Parameters:
    - processed_segments: list of motion segments
    - motion_ids: labels for each segment
    - degree_range: range of max_degree values to test (e.g., range(3, 10))
    - out_dir: directory to save results

    Returns:
    - results_dict: dictionary containing all results
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'degrees': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'cv_means': [],
        'cv_stds': [],
        'feature_dims': []
    }

    print("=" * 80)
    print("LEGENDRE POLYNOMIAL DEGREE OPTIMIZATION")
    print("=" * 80)
    print(f"\nTesting max_degree values: {list(degree_range)}")
    print(f"Number of segments: {len(processed_segments)}")
    print(f"Number of signals per segment: {processed_segments[0].shape[0]}")
    print(f"Number of motion classes: {len(np.unique(motion_ids))}\n")

    for max_degree in degree_range:
        print(f"\n{'=' * 60}")
        print(f"Testing max_degree = {max_degree}")
        print(f"{'=' * 60}")

        # Fit Legendre polynomials
        coefficients = fit_legendre_polynomials(processed_segments, max_degree)

        # Prepare data
        X = prepare_coefficient_data(coefficients)
        y = np.array(motion_ids)

        # Feature dimensionality
        feature_dim = X.shape[1]
        print(f"Feature dimension: {feature_dim} ({processed_segments[0].shape[0]} joints × {max_degree + 1} coeffs)")

        # Evaluate classification
        eval_results = evaluate_classification(X, y, max_degree)

        # Store results
        results['degrees'].append(max_degree)
        results['train_accuracies'].append(eval_results['train_accuracy'])
        results['test_accuracies'].append(eval_results['test_accuracy'])
        results['cv_means'].append(eval_results['cv_mean'])
        results['cv_stds'].append(eval_results['cv_std'])
        results['feature_dims'].append(feature_dim)

        # Print results
        print(f"Train Accuracy: {eval_results['train_accuracy']:.4f}")
        print(f"Test Accuracy:  {eval_results['test_accuracy']:.4f}")
        print(f"CV Mean ± Std:  {eval_results['cv_mean']:.4f} ± {eval_results['cv_std']:.4f}")
        print(f"CV Scores: {[f'{score:.4f}' for score in eval_results['cv_scores']]}")

    # Convert to numpy arrays for easier plotting
    results['degrees'] = np.array(results['degrees'])
    results['train_accuracies'] = np.array(results['train_accuracies'])
    results['test_accuracies'] = np.array(results['test_accuracies'])
    results['cv_means'] = np.array(results['cv_means'])
    results['cv_stds'] = np.array(results['cv_stds'])
    results['feature_dims'] = np.array(results['feature_dims'])

    # Find best degree
    best_idx = np.argmax(results['test_accuracies'])
    best_degree = results['degrees'][best_idx]
    best_accuracy = results['test_accuracies'][best_idx]

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best max_degree: {best_degree}")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Corresponding CV accuracy: {results['cv_means'][best_idx]:.4f} ± {results['cv_stds'][best_idx]:.4f}")
    print(f"Feature dimension at best: {results['feature_dims'][best_idx]}")

    # Save numerical results
    save_results_to_file(results, out_dir, best_degree, best_accuracy)

    # Plot results
    plot_optimization_results(results, out_dir, best_degree)

    return results


def save_results_to_file(results, out_dir, best_degree, best_accuracy):
    """Save numerical results to text file."""
    results_file = out_dir / "optimization_results.txt"

    with open(results_file, 'w') as f:
        f.write("LEGENDRE POLYNOMIAL DEGREE OPTIMIZATION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Best max_degree: {best_degree}\n")
        f.write(f"Best test accuracy: {best_accuracy:.4f}\n\n")

        f.write("Detailed Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Degree':<10} {'Features':<12} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<12}\n")
        f.write("-" * 80 + "\n")

        for i in range(len(results['degrees'])):
            f.write(f"{results['degrees'][i]:<10} "
                    f"{results['feature_dims'][i]:<12} "
                    f"{results['train_accuracies'][i]:<12.4f} "
                    f"{results['test_accuracies'][i]:<12.4f} "
                    f"{results['cv_means'][i]:<12.4f} "
                    f"{results['cv_stds'][i]:<12.4f}\n")

    print(f"\nResults saved to: {results_file}")


def plot_optimization_results(results, out_dir, best_degree):
    """Create visualization of optimization results."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    degrees = results['degrees']

    # Plot 1: Train vs Test Accuracy
    ax1 = axes[0, 0]
    ax1.plot(degrees, results['train_accuracies'],
             marker='o', linewidth=2, markersize=8,
             label='Train Accuracy', color='blue')
    ax1.plot(degrees, results['test_accuracies'],
             marker='s', linewidth=2, markersize=8,
             label='Test Accuracy', color='red')
    ax1.axvline(x=best_degree, color='green', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Best: {best_degree}')
    ax1.set_xlabel('Max Degree', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Train vs Test Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(degrees)

    # Plot 2: Cross-Validation Results with Error Bars
    ax2 = axes[0, 1]
    ax2.errorbar(degrees, results['cv_means'], yerr=results['cv_stds'],
                 marker='o', linewidth=2, markersize=8, capsize=5,
                 label='CV Mean ± Std', color='purple')
    ax2.plot(degrees, results['test_accuracies'],
             marker='s', linewidth=2, markersize=8,
             label='Test Accuracy', color='red', alpha=0.7)
    ax2.axvline(x=best_degree, color='green', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Best: {best_degree}')
    ax2.set_xlabel('Max Degree', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Cross-Validation Performance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(degrees)

    # Plot 3: Generalization Gap (Train - Test)
    ax3 = axes[1, 0]
    gap = results['train_accuracies'] - results['test_accuracies']
    ax3.plot(degrees, gap, marker='o', linewidth=2, markersize=8, color='orange')
    ax3.axvline(x=best_degree, color='green', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Best: {best_degree}')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax3.set_xlabel('Max Degree', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Train - Test Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Generalization Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(degrees)

    # Plot 4: Feature Dimension vs Test Accuracy
    ax4 = axes[1, 1]
    ax4.plot(results['feature_dims'], results['test_accuracies'],
             marker='o', linewidth=2, markersize=8, color='teal')
    best_idx = np.argmax(results['test_accuracies'])
    ax4.scatter(results['feature_dims'][best_idx],
                results['test_accuracies'][best_idx],
                s=200, color='green', edgecolors='black',
                linewidth=2, zorder=5, label=f'Best (degree={best_degree})')
    ax4.set_xlabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Test Accuracy vs Feature Dimension', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add degree labels to points
    for i, degree in enumerate(degrees):
        ax4.annotate(f'd={degree}',
                     (results['feature_dims'][i], results['test_accuracies'][i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)

    plt.tight_layout()

    # Save figure
    save_path = out_dir / 'legendre_degree_optimization.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Optimization plot saved to: {save_path}")


def main():
    """Main function to run the optimization."""

    # Import data processing function from your original script
    # Assuming you have access to this function
    from mp_movement_classifier.utils.utils import process_motion_data

    # Configuration
    num_MPs = 5
    model_dir = os.path.join("./../../results/tmp_configs",
                             f"new_seg_pymotion_position_mp_model_{num_MPs}_phase_two")
    out_dir = os.path.join(model_dir, "legendre_optimization")
    folder_path = "./../../data/pymotion_position_csv_files"

    # Load data
    print("Loading motion data...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=folder_path,
        data_type="position",
        filtering=False
    )

    print(f"Loaded {len(processed_segments)} segments")
    print(f"Number of signals per segment: {processed_segments[0].shape[0]}")
    print(f"Number of unique motion types: {len(np.unique(segment_motion_ids))}")

    # Define range of max_degree values to test
    degree_range = range(1, 10)  # Test degrees 3 through 9

    # Run optimization
    results = optimize_max_degree(
        processed_segments=processed_segments,
        motion_ids=segment_motion_ids,
        degree_range=degree_range,
        out_dir=out_dir
    )

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()