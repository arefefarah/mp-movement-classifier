# mp-movement-classifier

Temporal Movement Primitives (TMP) extraction and multi‑model classification from motion capture data, with unified analysis across TMP, Autoencoder, and Legendre features.

## What’s inside
- TMP (Temporal Movement Primitives): feature extraction and reconstruction.
- Autoencoder: temporal AE for sequence embeddings (latent vectors).
- Legendre analysis: polynomial‑basis coefficients from segments.
- Unified classification pipeline: single code path (LinearSVC + optional RandomForest), PCA/t‑SNE/RDM visuals, fixed‑scale percentage confusion matrices, feature importance.
- One‑shot “classification only” runner to compare the three models without retraining.

## Install
- Python 3.10+
- Install dependencies:
  - pip: `pip install -e .`
  - or Poetry: `poetry install`

## Data preparation
- Place motion CSVs under: `data/pymotion_position_csv_files/`
  - Each CSV is a time series (rows = time, columns = joint coordinates). Position mode typically uses 48 channels (16 joints × 3 axes), but other channel counts are supported if consistent.
  - Use your own CSV exporter or convert BVH to CSV via the utilities in `bvh_converter/` or `PyMO/` (optional).
- Optional label mapping: `data/motion_mapping.json` can map integer motion IDs to human‑readable names (used in some plots).
- The scripts handle segmentation internally; no manual segmenting is required for the default pipeline.

## Key entry points
- TMP training/eval: `mp_movement_classifier/tmp_extraction/main.py`
- Autoencoder analysis: `mp_movement_classifier/benchmark_analysis/autoencoder_extraction.py`
- Legendre analysis: `mp_movement_classifier/benchmark_analysis/legendre_extraction.py`
- Unified classification pipeline: `mp_movement_classifier/classification/pipeline.py`
- Run classification only (compare models): `mp_movement_classifier/classification/run_models.py`

## Quick start
1) Train or load TMP (saves under results/...):
   - `python mp_movement_classifier/tmp_extraction/main.py --num-mps 20 --num-t-points 30 --load 0`
   - After training, TMP features are classified automatically via the unified pipeline.
2) Autoencoder:
   - `python mp_movement_classifier/benchmark_analysis/autoencoder_extraction.py`
   - Script loads a saved AE checkpoint if present and runs unified classification on latents.
3) Legendre:
   - `python mp_movement_classifier/benchmark_analysis/legendre_extraction.py`
   - Extracts coefficients and runs unified classification.

## Run classification only (no retraining)
Use the multi‑model runner to classify existing features and compare outputs:
```
python run_classification_all_models.py \
  --models tmp ae legendre \
  --data-dir ../../data/pymotion_position_csv_files \
  --tmp-model-dir ../../results/tmp_configs/new_seg_mp_model_5_phase_three \
  --ae-model-path ../../results/tmp_configs/new_seg_mp_model_5_phase_three/autoencoder_analysis/models/best_autoencoder.pt \
  --legendre-out-dir  ../../results/tmp_configs/new_seg_mp_model_5_phase_three/legendre_analysis \
  --primary-classifier linear_svc --rf 0 --seed 42
```
Notes:
- To disable the secondary RandomForest run, pass `--rf 0`.
- AE latents are cached under the data directory to avoid recomputation (created automatically).
- Confusion matrices are row‑normalized percentages with a fixed [0, 1] color scale for fair comparison.

## Outputs
For each model and run, artifacts are saved under a `classification/` subfolder:
- Classification report(s) (txt)
- Confusion matrix (fixed‑scale, percentage)
- Feature importance (when available)
- PCA/t‑SNE/RDM visualizations

## Repository layout (high level)
- `mp_movement_classifier/tmp_extraction/`: TMP model training, evaluation, visualizations.
- `mp_movement_classifier/benchmark_analysis/`: Autoencoder + Legendre feature extraction and analyses.
- `mp_movement_classifier/classification/`: unified pipeline, helpers, and classification runner.
- `utils/`, `bvh_converter/`, `PyMO/`: utilities and data converters.
- `results/`: experiment outputs.
- `data/`: input CSVs and caches (not tracked).

## License
Research/educational use. See project files for details.
