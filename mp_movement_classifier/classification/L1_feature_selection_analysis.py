"""
L1 Regularization Feature Selection Analysis for Movement Classification
WITH PROPER SIGNAL NAMES
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Import your existing utilities
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
)


# ========== SIGNAL NAMES DEFINITION ==========
SIGNAL_NAMES = [
    'Hip_x', 'Hip_y', 'Hip_z',
    'RHip_x', 'RHip_y', 'RHip_z',
    'RKnee_x', 'RKnee_y', 'RKnee_z',
    'RAnkle_x', 'RAnkle_y', 'RAnkle_z',
    'LHip_x', 'LHip_y', 'LHip_z',
    'LKnee_x', 'LKnee_y', 'LKnee_z',
    'LAnkle_x', 'LAnkle_y', 'LAnkle_z',
    'Spine_x', 'Spine_y', 'Spine_z',
    'Thorax_x', 'Thorax_y', 'Thorax_z',
    'Neck_x', 'Neck_y', 'Neck_z',
    'LShoulder_x', 'LShoulder_y', 'LShoulder_z',
    'LElbow_x', 'LElbow_y', 'LElbow_z',
    'LWrist_x', 'LWrist_y', 'LWrist_z',
    'RShoulder_x', 'RShoulder_y', 'RShoulder_z',
    'RElbow_x', 'RElbow_y', 'RElbow_z',
    'RWrist_x', 'RWrist_y', 'RWrist_z',
]


def create_feature_names(num_signals: int, num_MPs: int) -> List[str]:
    """
    Create descriptive feature names using signal names and MP indices

    Args:
        num_signals: Number of signals (should be 48)
        num_MPs: Number of movement primitives per signal

    Returns:
        List of feature names in the format "SignalName_MP#"
    """
    if num_signals != len(SIGNAL_NAMES):
        print(f"WARNING: Expected {len(SIGNAL_NAMES)} signals, got {num_signals}")
        print(f"Using generic names for signals beyond {len(SIGNAL_NAMES)}")

    feature_names = []
    for signal_idx in range(num_signals):
        if signal_idx < len(SIGNAL_NAMES):
            signal_name = SIGNAL_NAMES[signal_idx]
        else:
            signal_name = f"Signal_{signal_idx}"

        for mp_idx in range(num_MPs):
            feature_names.append(f"{signal_name}_MP{mp_idx}")

    return feature_names


class L1FeatureSelector:
    """
    Systematic L1 regularization analysis for feature selection
    """

    def __init__(self, C_values: List[float] = None, random_state: int = 42):
        """
        Args:
            C_values: List of C values to test. If None, uses default range.
            random_state: Random seed for reproducibility
        """
        if C_values is None:
            # Logarithmically spaced C values from very strong to weak regularization
            self.C_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        else:
            self.C_values = sorted(C_values)

        self.random_state = random_state
        self.results = {}
        self.best_model = None
        self.best_C = None

    def prepare_weights_for_classification(
        self, model, num_segments: int, num_signals: int, num_MPs: int = 20
    ) -> np.ndarray:
        """
        Extract weight features from movement primitive model
        """
        X = np.zeros((num_segments, num_signals * num_MPs))

        for seg_idx in range(num_segments):
            for signal_idx in range(num_signals):
                for mp_idx in range(num_MPs):
                    feature_idx = signal_idx * num_MPs + mp_idx
                    X[seg_idx, feature_idx] = model.weights[seg_idx][signal_idx, mp_idx].item()

        return X

    def analyze_single_C(
        self,
        C: float,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        Train model and collect statistics for a single C value

        Returns:
            Dictionary with results for this C value
        """
        print(f"\n{'='*60}")
        print(f"Testing C = {C}")
        print(f"{'='*60}")

        # Train model with L1 penalty
        # CRITICAL: dual=False is required for L1 penalty
        clf = LinearSVC(
            C=C,
            penalty='l1',
            dual=False,  # Must be False for L1
            max_iter=10000,
            random_state=self.random_state
        )

        clf.fit(X_train, y_train)

        # Get predictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        # Calculate accuracies
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Get coefficients (shape: [n_classes, n_features] for multiclass)
        coefs = clf.coef_  # Shape: [n_classes, n_features]

        # Calculate feature importance as mean absolute coefficient across classes
        feature_importance = np.mean(np.abs(coefs), axis=0)

        # Identify non-zero features
        non_zero_mask = feature_importance > 1e-10  # Numerical tolerance
        non_zero_indices = np.where(non_zero_mask)[0]
        num_selected = np.sum(non_zero_mask)

        # Calculate sparsity
        sparsity = 1.0 - (num_selected / len(feature_importance))

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Features Selected: {num_selected} / {len(feature_importance)}")
        print(f"  Sparsity: {sparsity:.2%}")

        # Get top features
        top_k = min(20, num_selected)
        top_indices = np.argsort(feature_importance)[::-1][:top_k]

        print(f"\n  Top {top_k} Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            feat_name = feature_names[idx] if feature_names else f"Feature_{idx}"
            print(f"    {i:2d}. {feat_name:30s}: {feature_importance[idx]:.6f}")

        # Store results
        results = {
            'C': C,
            'model': clf,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'num_features_selected': num_selected,
            'sparsity': sparsity,
            'non_zero_indices': non_zero_indices.tolist(),
            'feature_importance': feature_importance,
            'coefficients': coefs,
            'y_test_pred': y_test_pred,
        }

        return results

    def run_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None,
        test_size: float = 0.25
    ):
        """
        Run complete L1 regularization analysis across all C values
        """
        print(f"\n{'#'*70}")
        print(f"# L1 REGULARIZATION FEATURE SELECTION ANALYSIS")
        print(f"{'#'*70}")
        print(f"\nDataset Info:")
        print(f"  - Total samples: {X.shape[0]}")
        print(f"  - Total features: {X.shape[1]}")
        print(f"  - Unique classes: {len(np.unique(y))}")
        print(f"  - C values to test: {len(self.C_values)}")
        print(f"  - Test size: {test_size:.0%}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        self.X_test = X_test_scaled
        self.y_test = y_test

        # Analyze each C value
        for C in self.C_values:
            result = self.analyze_single_C(
                C, X_train_scaled, X_test_scaled,
                y_train, y_test, feature_names
            )
            self.results[C] = result

        # Find best model based on test accuracy
        best_C = max(self.results.keys(), key=lambda c: self.results[c]['test_accuracy'])
        self.best_C = best_C
        self.best_model = self.results[best_C]['model']

        print(f"\n{'='*60}")
        print(f"BEST MODEL: C = {best_C}")
        print(f"  Test Accuracy: {self.results[best_C]['test_accuracy']:.4f}")
        print(f"  Features Selected: {self.results[best_C]['num_features_selected']}")
        print(f"{'='*60}")

    def plot_results(self, out_dir: Path, feature_names: List[str] = None):
        """
        Create comprehensive visualization of results
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract metrics
        C_vals = []
        train_accs = []
        test_accs = []
        num_features = []
        sparsities = []

        for C in self.C_values:
            r = self.results[C]
            C_vals.append(C)
            train_accs.append(r['train_accuracy'])
            test_accs.append(r['test_accuracy'])
            num_features.append(r['num_features_selected'])
            sparsities.append(r['sparsity'])

        # ============ Plot 1: Accuracy vs C ============
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogx(C_vals, train_accs, 'o-', label='Train Accuracy',
                    linewidth=2, markersize=8, color='blue')
        ax.semilogx(C_vals, test_accs, 's-', label='Test Accuracy',
                    linewidth=2, markersize=8, color='red')

        # Mark best C
        best_idx = C_vals.index(self.best_C)
        ax.axvline(self.best_C, color='green', linestyle='--',
                   label=f'Best C = {self.best_C}', linewidth=2)
        ax.scatter(self.best_C, test_accs[best_idx],
                   s=200, color='green', marker='*', zorder=5,
                   edgecolors='black', linewidths=2)

        ax.set_xlabel('C (Regularization Parameter)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy vs Regularization Strength',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.savefig(out_dir / 'l1_accuracy_vs_C.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_dir / 'l1_accuracy_vs_C.png'}")

        # ============ Plot 2: Number of Features vs C ============
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.semilogx(C_vals, num_features, 'o-', linewidth=2,
                    markersize=8, color='purple')
        ax.axvline(self.best_C, color='green', linestyle='--',
                   label=f'Best C = {self.best_C}', linewidth=2)

        ax.set_xlabel('C (Regularization Parameter)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Selected Features', fontsize=12, fontweight='bold')
        ax.set_title('Feature Selection vs Regularization Strength',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'l1_num_features_vs_C.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_dir / 'l1_num_features_vs_C.png'}")

        # ============ Plot 3: Combined View ============
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Top: Accuracy
        ax1.semilogx(C_vals, train_accs, 'o-', label='Train',
                     linewidth=2, markersize=6, color='blue')
        ax1.semilogx(C_vals, test_accs, 's-', label='Test',
                     linewidth=2, markersize=6, color='red')
        ax1.axvline(self.best_C, color='green', linestyle='--', alpha=0.7)
        ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('Accuracy and Feature Selection vs Regularization',
                      fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Bottom: Number of features
        ax2.semilogx(C_vals, num_features, 'o-', linewidth=2,
                     markersize=6, color='purple')
        ax2.axvline(self.best_C, color='green', linestyle='--',
                    label=f'Best C = {self.best_C}', alpha=0.7)
        ax2.set_xlabel('C (Regularization Parameter)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('# Features Selected', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'l1_combined_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_dir / 'l1_combined_analysis.png'}")

        # ============ Plot 4: Feature Selection Heatmap ============
        n_features = len(self.results[self.C_values[0]]['feature_importance'])
        n_C = len(self.C_values)

        # Create binary matrix: 1 if feature selected, 0 otherwise
        selection_matrix = np.zeros((n_C, n_features))
        for i, C in enumerate(self.C_values):
            non_zero_idx = self.results[C]['non_zero_indices']
            selection_matrix[i, non_zero_idx] = 1

        # Calculate feature selection frequency
        selection_freq = np.sum(selection_matrix, axis=0) / n_C

        # Sort features by selection frequency
        sorted_indices = np.argsort(selection_freq)[::-1]

        # Plot only features that are selected at least once
        selected_at_least_once = sorted_indices[selection_freq[sorted_indices] > 0]
        n_plot = min(100, len(selected_at_least_once))  # Plot top 100 or fewer

        if n_plot > 0:
            plot_indices = selected_at_least_once[:n_plot]

            fig, ax = plt.subplots(figsize=(max(12, n_plot * 0.15), 8))

            # Transpose for better visualization (features on y-axis)
            heatmap_data = selection_matrix[:, plot_indices].T

            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                          interpolation='nearest')

            # Set ticks
            ax.set_yticks(np.arange(n_plot))
            ax.set_xticks(np.arange(n_C))

            # Labels
            if feature_names:
                y_labels = [feature_names[i] for i in plot_indices]
            else:
                y_labels = [f'F{i}' for i in plot_indices]

            ax.set_yticklabels(y_labels, fontsize=7)
            ax.set_xticklabels([f'{c:.0e}' for c in self.C_values],
                              rotation=45, ha='right', fontsize=9)

            # Mark best C
            best_C_idx = self.C_values.index(self.best_C)
            ax.axvline(best_C_idx, color='green', linewidth=3,
                      linestyle='--', label='Best C')

            ax.set_xlabel('C Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Feature Index', fontsize=12, fontweight='bold')
            ax.set_title(f'Feature Selection Pattern Across C Values\n'
                        f'(Top {n_plot} most frequently selected features)',
                        fontsize=13, fontweight='bold')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Selected (1) / Not Selected (0)',
                          rotation=270, labelpad=20, fontsize=10)

            plt.tight_layout()
            plt.savefig(out_dir / 'l1_feature_selection_heatmap.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {out_dir / 'l1_feature_selection_heatmap.png'}")

        # ============ Plot 5: Feature Stability Analysis ============
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by frequency
        sorted_freq = selection_freq[sorted_indices]
        n_plot_stability = min(50, len(selected_at_least_once))

        if n_plot_stability > 0:
            plot_freq = sorted_freq[:n_plot_stability]
            plot_idx = sorted_indices[:n_plot_stability]

            colors = plt.cm.RdYlGn(plot_freq)
            bars = ax.barh(np.arange(n_plot_stability), plot_freq, color=colors)

            if feature_names:
                labels = [feature_names[i] for i in plot_idx]
            else:
                labels = [f'F{i}' for i in plot_idx]

            ax.set_yticks(np.arange(n_plot_stability))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Selection Frequency', fontsize=12, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {n_plot_stability} Most Stable Features\n'
                        '(Selected across different C values)',
                        fontsize=13, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='x')

            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                       norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label('Frequency', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.savefig(out_dir / 'l1_feature_stability.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {out_dir / 'l1_feature_stability.png'}")

        # ============ Plot 6: Best Model Feature Importance ============
        best_result = self.results[self.best_C]
        importance = best_result['feature_importance']
        non_zero_idx = np.array(best_result['non_zero_indices'])

        if len(non_zero_idx) > 0:
            sorted_idx = non_zero_idx[np.argsort(importance[non_zero_idx])[::-1]]
            n_plot_best = min(40, len(sorted_idx))  # Increased to 40 for better visibility

            fig, ax = plt.subplots(figsize=(12, max(8, n_plot_best * 0.25)))

            top_idx = sorted_idx[:n_plot_best]
            top_importance = importance[top_idx]

            if feature_names:
                labels = [feature_names[i] for i in top_idx]
            else:
                labels = [f'F{i}' for i in top_idx]

            colors = plt.cm.viridis(np.linspace(0.3, 0.9, n_plot_best))
            bars = ax.barh(np.arange(n_plot_best), top_importance, color=colors)

            ax.set_yticks(np.arange(n_plot_best))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Feature Importance (Mean |Coefficient|)',
                         fontsize=12, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
            ax.set_title(f'Top {n_plot_best} Features for Best Model (C={self.best_C})',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            # Add values on bars
            for i, (bar, val) in enumerate(zip(bars, top_importance)):
                ax.text(val, i, f' {val:.4f}',
                       va='center', fontsize=7, fontweight='bold')

            plt.tight_layout()
            plt.savefig(out_dir / 'l1_best_model_features.png',
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {out_dir / 'l1_best_model_features.png'}")

    def plot_confusion_matrix(self, out_dir: Path, class_names: List[str] = None):
        """
        Plot confusion matrix for best model
        """
        out_dir = Path(out_dir)

        y_pred = self.results[self.best_C]['y_test_pred']
        cm = confusion_matrix(self.y_test, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   cbar=True, ax=ax, square=True)

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix (Best Model: C={self.best_C})',
                    fontsize=14, fontweight='bold')

        if class_names:
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names, rotation=0)

        plt.tight_layout()
        plt.savefig(out_dir / 'l1_confusion_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_dir / 'l1_confusion_matrix.png'}")

    def save_results(self, out_dir: Path, feature_names: List[str] = None):
        """
        Save detailed results to files
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ============ Classification Report for Best Model ============
        y_pred = self.results[self.best_C]['y_test_pred']
        report = classification_report(self.y_test, y_pred)

        with open(out_dir / 'l1_classification_report.txt', 'w') as f:
            f.write(f"Classification Report (Best Model: C={self.best_C})\n")
            f.write("="*70 + "\n\n")
            f.write(report)

        print(f"  ✓ Saved: {out_dir / 'l1_classification_report.txt'}")

        # ============ Save Best Model ============
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'C': self.best_C,
            'selected_features': self.results[self.best_C]['non_zero_indices'],
            'feature_importance': self.results[self.best_C]['feature_importance'],
            'test_accuracy': self.results[self.best_C]['test_accuracy'],
            'feature_names': feature_names,  # Include feature names
        }

        with open(out_dir / 'l1_best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"  ✓ Saved: {out_dir / 'l1_best_model.pkl'}")

def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("L1 REGULARIZATION FEATURE SELECTION FOR MOVEMENT CLASSIFICATION")
    print("="*70 + "\n")

    # ========== Configuration ==========
    num_MPs = 5
    cutoff_freq = 3.0
    tpoints = 30

    model_dir = os.path.join(
        "./../../results/tmp_configs",
        f"new_seg_pymotion_position_mp_model_{num_MPs}_phase_two"
    )
    model_file = os.path.join(
        model_dir,
        f"mp_model_{num_MPs}_PC_tpoints_{tpoints}"
    )

    out_dir = os.path.join(model_dir, "l1_feature_selection")

    folder_path = "../../data/pymotion_position_csv_files"

    # ========== Load Data ==========
    print("Loading and processing data...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=folder_path,
        data_type="position",
        filtering=False
    )

    num_segments = len(processed_segments)
    num_signals = processed_segments[0].shape[0]

    print(f"  - Number of segments: {num_segments}")
    print(f"  - Number of signals: {num_signals}")
    print(f"  - Unique motions: {len(np.unique(segment_motion_ids))}")

    # ========== Load Model ==========
    print("\nLoading movement primitive model...")
    model = load_model_with_full_state(
        model_file,
        num_segments=num_segments,
        num_signals=num_signals
    )

    # ========== Prepare Features ==========
    print("\nPreparing feature matrix from model weights...")
    selector = L1FeatureSelector()

    X = selector.prepare_weights_for_classification(
        model, num_segments, num_signals, num_MPs
    )
    y = np.array(segment_motion_ids)

    # Create feature names using proper signal names
    feature_names = create_feature_names(num_signals, num_MPs)

    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Label array shape: {y.shape}")
    print(f"  - Number of feature names: {len(feature_names)}")

    # Print sample feature names
    print("\n  Sample feature names:")
    for i in [0, 1, 2, -3, -2, -1]:
        print(f"    {i:3d}: {feature_names[i]}")

    # ========== Run L1 Analysis ==========
    print("\nStarting L1 regularization analysis...")
    print(f"Testing C values: {selector.C_values}")

    selector.run_analysis(
        X=X,
        y=y,
        feature_names=feature_names,
        test_size=0.25
    )

    # ========== Generate Visualizations ==========
    print("\nGenerating visualizations...")
    selector.plot_results(Path(out_dir), feature_names)
    selector.plot_confusion_matrix(Path(out_dir))

    # ========== Save Results ==========
    print("\nSaving results...")
    selector.save_results(Path(out_dir), feature_names)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {out_dir}")
    print(f"\nBest Model Summary:")
    print(f"  - C = {selector.best_C}")
    print(f"  - Test Accuracy: {selector.results[selector.best_C]['test_accuracy']:.4f}")
    print(f"  - Features Selected: {selector.results[selector.best_C]['num_features_selected']} / {X.shape[1]}")
    print(f"  - Feature Reduction: {selector.results[selector.best_C]['sparsity']:.1%}")

    # Print top 10 selected features for best model
    print(f"\nTop 10 Selected Features (Best Model):")
    best_result = selector.results[selector.best_C]
    importance = best_result['feature_importance']
    non_zero_idx = np.array(best_result['non_zero_indices'])
    sorted_idx = non_zero_idx[np.argsort(importance[non_zero_idx])[::-1]]

    for i, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {i:2d}. {feature_names[idx]:35s}  Importance: {importance[idx]:.6f}")

    print("\n")


if __name__ == "__main__":
    main()