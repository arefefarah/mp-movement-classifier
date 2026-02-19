from scipy import special
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
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
from mp_movement_classifier.classification.classification import calculate_rdm
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,

)
from mp_movement_classifier.benchmark_analysis.lda_analysis import run_lda_analysis
from posture_removal_experiment import run_posture_removal_experiment
from mp_movement_classifier.benchmark_analysis.legendre_extraction import process_with_legendre_basis, prepare_coefficient_data

import sys
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import matplotlib

from mp_movement_classifier.classification.classification import prepare_weights_for_classification
from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,

)
from mp_movement_classifier.utils.plotting import (
    plot_learn_curve, plot_mp,
    plot_reconstructions,
    set_figures_directory
)



num_MPs = 5
tpoints = 30
# model_dir = os.path.join("./../../results/tmp_configs", f"new_seg_pymotion_position_mp_model_{num_MPs}_phase_two")
model_dir = os.path.join("./../../results/tmp_configs", f"mean_subtracted_mp_model_{num_MPs}_tpoints_{tpoints}_phase_two")
model_path = os.path.join(model_dir, f"mp_model_{num_MPs}_PC_tpoints_{tpoints}")
out_dir = os.path.join(model_dir, "legandre_analysis")
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)


folder_path = "./../../data/pymotion_position_csv_files"
motion_ids, processed_segments, segment_motion_ids = process_motion_data(folder_path=folder_path,
                                                                         data_type = "position",
                                                                         filtering= False)
num_segments = len(processed_segments)
print(f"Number of segments: {num_segments}")
num_signals = processed_segments[0].shape[0]
print(f"Number of signals: {num_signals}")

tmp_model = load_model_with_full_state(
        model_path,
        num_segments=num_segments,
        num_signals=num_signals
    )

max_degree = 1  # for polynomial degrees as basis function , For 10 degrees (0 to 9)
coefficients, errors = process_with_legendre_basis(processed_segments, max_degree)

X_legendre, y_legendre = prepare_coefficient_data(coefficients, segment_motion_ids)
print(f"legendre Feature matrix shape: {X_legendre.shape}")
print(f"Label array shape: {y_legendre.shape}")
print(f" {len(np.unique(y_legendre))} unique motion types")

X_tmp = prepare_weights_for_classification(tmp_model, num_segments, num_signals, num_MPs)
y_tmp = np.array(segment_motion_ids)
print(f"TMP Feature matrix shape: {X_tmp.shape}")
print(f"Label array shape: {y_tmp.shape}")

results = run_posture_removal_experiment(
        processed_segments=processed_segments,
        segment_motion_ids=segment_motion_ids,
        out_dir=os.path.join(out_dir, "posture_experiment"),
        tmp_weights=X_tmp,  # or None
        ae_latents=None,  # or your AE features
        max_degrees=list(range(1, 10)),
    )