import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from scipy import special
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from pathlib import Path
from mp_movement_classifier.utils import config
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
    read_bvh_files,
    save_model_with_full_state,

)


class MotionDataset(Dataset):
    """Dataset for variable-length motion segments with padding"""

    def __init__(self, segments: List[np.ndarray], motion_ids: List[int],
                 max_length: int = None, pad_mode: str = 'constant'):
        """
        Args:
            segments: List of numpy arrays with shape (timesteps, features)
            motion_ids: Corresponding motion class labels
            max_length: Maximum sequence length (if None, use longest sequence)
            pad_mode: 'edge', 'constant', or 'reflect'
        """
        self.segments = segments
        self.motion_ids = motion_ids
        self.pad_mode = pad_mode

        # Find max length if not provided
        if max_length is None:
            self.max_length = max(seg.shape[0] for seg in segments)
        else:
            self.max_length = max_length

        # Get feature dimension
        self.n_features = segments[0].shape[1]

        # Precompute padded segments and masks
        self.padded_segments, self.masks = self._pad_segments()

    def _pad_segments(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pad all segments to max_length and create attention masks"""
        n_samples = len(self.segments)
        padded = np.zeros((n_samples, self.max_length, self.n_features))
        masks = np.zeros((n_samples, self.max_length), dtype=bool)

        for i, segment in enumerate(self.segments):
            length = segment.shape[0]

            if length >= self.max_length:
                # Truncate if longer than max_length
                padded[i] = segment[:self.max_length]
                masks[i] = True
            else:
                # Pad if shorter
                padded[i, :length] = segment
                masks[i, :length] = True

                # Pad the remaining part
                if self.pad_mode == 'edge':
                    padded[i, length:] = segment[-1]
                elif self.pad_mode == 'constant':
                    pass  # Already zeros
                elif self.pad_mode == 'reflect':
                    remaining = self.max_length - length
                    if length > 1:
                        reflected = np.flip(segment[1:], axis=0)
                        for j in range(remaining):
                            padded[i, length + j] = reflected[j % len(reflected)]

        return padded, masks

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return {
            'data': torch.FloatTensor(self.padded_segments[idx]),
            'mask': torch.BoolTensor(self.masks[idx]),
            'label': self.motion_ids[idx]
        }


def prepare_datasets(processed_segments: List[np.ndarray],
                     segment_motion_ids: List[int],
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     max_length: int = None,
                     normalize: bool = True) -> Tuple:
    """
    Prepare train/val/test datasets with normalization

    Returns:
        train_dataset, val_dataset, test_dataset, scaler
    """
    # Split indices
    n_samples = len(processed_segments)
    indices = np.arange(n_samples)

    # Train/test split
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=segment_motion_ids, random_state=42
    )

    # Train/val split
    train_labels = [segment_motion_ids[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size / (1 - test_size),
        stratify=train_labels, random_state=42
    )

    # Normalize if requested
    scaler = None
    if normalize:
        # Fit scaler on training data only
        train_data = np.vstack([processed_segments[i] for i in train_idx])
        scaler = StandardScaler()
        scaler.fit(train_data)

        # Transform all segments
        normalized_segments = []
        for seg in processed_segments:
            normalized_segments.append(scaler.transform(seg))
        processed_segments = normalized_segments

    # Create datasets
    train_dataset = MotionDataset(
        [processed_segments[i] for i in train_idx],
        [segment_motion_ids[i] for i in train_idx],
        max_length=max_length
    )

    val_dataset = MotionDataset(
        [processed_segments[i] for i in val_idx],
        [segment_motion_ids[i] for i in val_idx],
        max_length=train_dataset.max_length  # Use same max_length
    )

    test_dataset = MotionDataset(
        [processed_segments[i] for i in test_idx],
        [segment_motion_ids[i] for i in test_idx],
        max_length=train_dataset.max_length
    )

    return train_dataset, val_dataset, test_dataset, scaler


class TemporalEncoder(nn.Module):
    """Encoder that processes temporal sequences"""

    def __init__(self, input_dim: int = 28, hidden_dim: int = 64,
                 latent_dim: int = 32, use_lstm: bool = True):
        super().__init__()
        self.use_lstm = use_lstm

        if use_lstm:
            # LSTM for temporal processing
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                                num_layers=2, dropout=0.2)
        else:
            # MLP for frame-by-frame processing
            self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, time, features)
            mask: (batch, time) boolean mask
        Returns:
            latent: (batch, latent_dim)
        """
        if self.use_lstm:
            # LSTM processes the full sequence
            lstm_out, (h_n, c_n) = self.lstm(x)

            if mask is not None:
                # Use masked average pooling
                mask_expanded = mask.unsqueeze(-1).float()
                masked_out = lstm_out * mask_expanded
                pooled = masked_out.sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                # Simple average pooling
                pooled = lstm_out.mean(dim=1)

            x = pooled
        else:
            # Process frame by frame
            x = self.leaky_relu(self.fc1(x))
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                x = x * mask_expanded
                x = x.sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                x = x.mean(dim=1)

        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        latent = self.fc3(x)

        return latent


class TemporalDecoder(nn.Module):
    """Decoder that reconstructs temporal sequences"""

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 output_dim: int = 28, max_length: int = 100, use_lstm: bool = True):
        super().__init__()
        self.use_lstm = use_lstm
        self.max_length = max_length

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                                num_layers=2, dropout=0.2)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()

    def forward(self, latent, target_length=None):
        """
        Args:
            latent: (batch, latent_dim)
            target_length: int, sequence length to generate
        Returns:
            reconstructed: (batch, time, features)
        """
        batch_size = latent.size(0)
        if target_length is None:
            target_length = self.max_length

        x = self.leaky_relu(self.fc1(latent))
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)

        # Repeat for each timestep
        x = x.unsqueeze(1).repeat(1, target_length, 1)

        if self.use_lstm:
            x, _ = self.lstm(x)

        reconstructed = self.tanh(self.fc3(x))

        return reconstructed


class TemporalAutoencoder(nn.Module):
    """Complete autoencoder for temporal motion data"""

    def __init__(self, input_dim: int = 28, hidden_dim: int = 64,
                 latent_dim: int = 32, max_length: int = 100, use_lstm: bool = True):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, latent_dim, use_lstm)
        self.decoder = TemporalDecoder(latent_dim, hidden_dim, input_dim,
                                       max_length, use_lstm)

    def forward(self, x, mask=None):
        latent = self.encoder(x, mask)
        reconstructed = self.decoder(latent, target_length=x.size(1))
        return reconstructed, latent

    def encode(self, x, mask=None):
        """Extract latent representations"""
        return self.encoder(x, mask)


def masked_mse_loss(pred, target, mask):
    """MSE loss that ignores padded regions"""
    mask_expanded = mask.unsqueeze(-1).float()
    squared_diff = (pred - target) ** 2
    masked_squared_diff = squared_diff * mask_expanded

    # Average only over valid timesteps
    loss = masked_squared_diff.sum() / mask_expanded.sum()
    return loss


def train_autoencoder(model, train_loader, val_loader,
                      n_epochs: int = 100,
                      lr: float = 0.001,
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                      patience: int = 15,
                      save_path: Path = None):
    """
    Train the autoencoder with early stopping

    Returns:
        model, train_losses, val_losses
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            data = batch['data'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            reconstructed, latent = model(data, mask)
            loss = masked_mse_loss(reconstructed, data, mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data = batch['data'].to(device)
                mask = batch['mask'].to(device)

                reconstructed, latent = model(data, mask)
                loss = masked_mse_loss(reconstructed, data, mask)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    if save_path and save_path.exists():
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, train_losses, val_losses


def extract_representations(model, dataloader, device='cuda'):
    """Extract latent representations from trained autoencoder"""
    model.eval()
    representations = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            data = batch['data'].to(device)
            mask = batch['mask'].to(device)

            latent = model.encode(data, mask)
            representations.append(latent.cpu().numpy())
            labels.append(batch['label'].numpy())

    representations = np.vstack(representations)
    labels = np.concatenate(labels)

    return representations, labels


def train_classifiers(X_train, y_train, X_test, y_test):
    """Train and compare multiple classifiers"""
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                                random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                             random_state=42)
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)

        # Cross-validation on training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5)

        # Test set evaluation
        y_pred = clf.predict(X_test)

        results[name] = {
            'model': clf,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(classification_report(y_test, y_pred))

    return results


def visualize_reconstruction(model, dataset, n_samples=5, device='cuda'):
    """Compare original vs reconstructed sequences"""
    model.eval()
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 3 * n_samples))

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            data = sample['data'].unsqueeze(0).to(device)
            mask = sample['mask'].unsqueeze(0).to(device)

            reconstructed, _ = model(data, mask)

            # Convert to numpy
            original = data[0].cpu().numpy()
            recon = reconstructed[0].cpu().numpy()
            mask_np = mask[0].cpu().numpy()

            # Plot original
            axes[i, 0].imshow(original.T, aspect='auto', cmap='viridis')
            axes[i, 0].set_title(f'Original (Label: {sample["label"]})')
            axes[i, 0].set_ylabel('Feature Dimension')

            # Plot reconstruction
            axes[i, 1].imshow(recon.T, aspect='auto', cmap='viridis')
            axes[i, 1].set_title('Reconstructed')

            # Mark padding
            for ax in axes[i]:
                valid_length = mask_np.sum()
                ax.axvline(x=valid_length - 0.5, color='red', linestyle='--',
                           label='Padding Start')

    axes[-1, 0].set_xlabel('Time Step')
    axes[-1, 1].set_xlabel('Time Step')
    plt.tight_layout()
    return fig


def visualize_latent_space(representations, labels, method='tsne'):
    """Visualize latent representations"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # PCA
    pca = PCA(n_components=2)
    pca_repr = pca.fit_transform(representations)

    scatter = axes[0].scatter(pca_repr[:, 0], pca_repr[:, 1],
                              c=labels, cmap='tab10', alpha=0.6)
    axes[0].set_title(f'PCA (Variance Explained: {pca.explained_variance_ratio_.sum():.2%})')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter, ax=axes[0], label='Class')

    # t-SNE or UMAP
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE'
    else:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
        title = 'UMAP'

    reduced = reducer.fit_transform(representations)
    scatter = axes[1].scatter(reduced[:, 0], reduced[:, 1],
                              c=labels, cmap='tab10', alpha=0.6)
    axes[1].set_title(title)
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    plt.colorbar(scatter, ax=axes[1], label='Class')

    plt.tight_layout()
    return fig, pca


def plot_training_curves(train_losses, val_losses):
    """Plot training and validation losses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def analyze_variance_explained(representations):
    """Analyze variance explained by latent dimensions"""
    pca = PCA()
    pca.fit(representations)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Explained variance ratio
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Variance Explained by Each Component')
    axes[0].grid(True, alpha=0.3)

    # Cumulative explained variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum) + 1), cumsum, marker='o')
    axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Find number of components for 95% variance
    n_components_95 = np.argmax(cumsum >= 0.95) + 1
    print(f"Number of components for 95% variance: {n_components_95}")

    return fig, pca


def main():
    num_MPs = 20
    model_dir = os.path.join("./../../results/tmp_configs", f"pymotion_quaternion_mp_model_{num_MPs}")
    out_dir = os.path.join(model_dir, "autoencoder_analysis")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR = "./../../data/pymotion_quat_csv_files"
    MODEL_SAVE_DIR = Path(out_dir) / 'models'
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR = Path(out_dir) / 'results'
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("MOTION AUTOENCODER TRAINING AND CLASSIFICATION")
    print("=" * 80)

    # Step 1: Load and process data
    print("\n[1/7] Loading motion data...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=DATA_DIR,
        data_type='quaternion',
    )
    #flip segments
    all_segmants = []
    for segment in processed_segments:
        all_segmants.append(segment.T)  # Transpose from  [signals, time] to [time, signals] again
    processed_segments = all_segmants
    print(f"Loaded {len(processed_segments)} segments across {len(set(motion_ids))} motions")
    print(f"Number of classes: {len(set(segment_motion_ids))}")

    print("\n[2/7] Preparing datasets...")
    train_dataset, val_dataset, test_dataset, scaler = prepare_datasets(
        processed_segments, segment_motion_ids,
        test_size=0.2,
        val_size=0.1,
        normalize=False
    )

    actual_input_dim = train_dataset.n_features

    CONFIG = {
        'input_dim': actual_input_dim,  # ← Use actual feature dimension (71 quaternion) 17*3 for position
        'hidden_dim': 128,
        'latent_dim': 32,  # Increased for more features , 32 for position 64 for quaternion
        'use_lstm': False,
        'batch_size': 32,
        'n_epochs': 100,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=False, num_workers=4)

    # Step 3: Initialize model
    print("\n[3/7] Initializing autoencoder...")
    model = TemporalAutoencoder(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        latent_dim=CONFIG['latent_dim'],
        max_length=train_dataset.max_length,
        use_lstm=CONFIG['use_lstm']
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Step 4: Train autoencoder
    print("\n[4/7] Training autoencoder...")
    model, train_losses, val_losses = train_autoencoder(
        model, train_loader, val_loader,
        n_epochs=CONFIG['n_epochs'],
        lr=CONFIG['lr'],
        device=CONFIG['device'],
        save_path=MODEL_SAVE_DIR / 'best_autoencoder.pt'
    )

    # Plot training curves
    fig = plot_training_curves(train_losses, val_losses)
    fig.savefig(RESULTS_DIR / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Step 5: Extract representations
    print("\n[5/7] Extracting latent representations...")
    train_repr, train_labels = extract_representations(model, train_loader, CONFIG['device'])
    test_repr, test_labels = extract_representations(model, test_loader, CONFIG['device'])

    print(f"Representation shape: {train_repr.shape}")

    # Step 6: Visualize
    print("\n[6/7] Generating visualizations...")

    # Reconstruction quality
    fig = visualize_reconstruction(model, test_dataset, n_samples=5,
                                   device=CONFIG['device'])
    fig.savefig(RESULTS_DIR / 'reconstruction_samples.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Latent space visualization
    fig, pca = visualize_latent_space(test_repr, test_labels, method='tsne')
    fig.savefig(RESULTS_DIR / 'latent_space.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Variance analysis
    fig, pca_full = analyze_variance_explained(train_repr)
    fig.savefig(RESULTS_DIR / 'variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Step 7: Classification
    print("\n[7/7] Training classifiers on learned representations...")
    results = train_classifiers(train_repr, train_labels, test_repr, test_labels)

    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    for idx, (name, result) in enumerate(results.items()):
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d',
                    cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'{name}\nAccuracy: {result["report"]["accuracy"]:.4f}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save results summary
    with open(RESULTS_DIR / 'classification_results.txt', 'w') as f:
        f.write("CLASSIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  CV Accuracy: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})\n")
            f.write(f"  Test Accuracy: {result['report']['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {result['report']['macro avg']['f1-score']:.4f}\n")
            f.write(f"  Weighted F1: {result['report']['weighted avg']['f1-score']:.4f}\n")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
#
# def autoencoder()
#
# def train_autoencoder(data,model):
#     def main() -> None:
#
#         # Fixed configuration (no CLI for these)
#         data_dir = DEFAULT_DATA_DIR
#         tail_window = DEFAULT_TAIL_WINDOW
#         model_name_suffix = MODEL_NAME_SUFFIX
#
#         motion_ids, processed_segments, segment_motion_ids = process_motion_data(folder_path=data_dir)
#
#
#         process_data ()
#         becasue segments does not have same length to be used for auto encode. we need to do padding to make sure all segments have s=xonstant length but keep segment motion ids as they are for using for classification
#
#
#         # Initialize or load model
#         model =
#
#         train_and_save_model(
#             )
#
#
#         evaluate_and_plot(
#             visulaize(rep): mapping
#         from input to
#         output
#         of
#         train_autoencoder(compare
#         this
#         two in terms
#         of
#         variance
#         explained)
#
#         )
#
#
#     if __name__ == "__main__":
#         main()
