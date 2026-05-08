"""
Joint-subset classification.

Runs three complementary subset experiments on TMP weights and compares
them with confusion matrices + accuracy tables. Each experiment uses
both Linear SVC and Random Forest. Only the supervised parts of the
shared pipeline are run (train/test, CV, confusion matrix, classification
report, feature importance) — PCA, t-SNE, RDM, and motion legend are
skipped on purpose.

Experiments
-----------
1) keep_5_joints_all_MPs   — keep ONLY {Wrists, Knees, Ankles, Neck,
                              Elbows} (L+R, x/y/z), all MPs per channel.
2) exclude_5_joints_all_MPs — INVERSE of (1): drop those 5 joints, use all
                              OTHER signals (all MPs per channel).
3) keep_5_joints_MP0_only   — same channels as (1), but only MP index 0
                              per channel (so we can see what dropping
                              the higher-order primitives costs).

Reuses (no duplication):
  - SIGNAL_NAMES + create_feature_names from L1_feature_selection_analysis
  - prepare_weights_for_classification from classification.utils
  - _plot_confusion_matrix_percent + _perform_cross_validation +
    _save_cross_validation_results from classification.classification_pipeline
  - save_classification_report + plot_and_save_feature_importance from
    classification.utils

Outputs (under <model_dir>/joint_subset_classification/):
  - experiments_comparison.txt           : cross-experiment summary table
  - <experiment_name>/
      - selected_channels.txt            : exact list of signals + MPs used
      - subset_summary.txt               : train/test acc per classifier
      - confusion_matrix_<clf>.{png,svg}
      - classification_report_<clf>.txt
      - cross_validation_<clf>.txt
      - feature_importance_<clf>.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from mp_movement_classifier.utils.utils import (
    load_model_with_full_state,
    process_motion_data,
)
from mp_movement_classifier.classification.utils import (
    prepare_weights_for_classification,
    save_classification_report,
    plot_and_save_feature_importance,
)
from mp_movement_classifier.classification.classification_pipeline import (
    _plot_confusion_matrix_percent,
    _perform_cross_validation,
    _save_cross_validation_results,
)
from mp_movement_classifier.classification.L1_feature_selection_analysis import (
    SIGNAL_NAMES,
    create_feature_names,
)


# ===== Joint subset definition =====
# Substring keywords matched against SIGNAL_NAMES (case-insensitive).
# This automatically picks up both L/R variants AND all axes (x/y/z).
JOINT_KEYWORDS: Tuple[str, ...] = (
    "Wrist",
    "Knee",
    "Ankle",
    "Neck",
    "Elbow",
)


def select_signal_indices(
        signal_names: Sequence[str],
        keywords: Sequence[str],
) -> List[int]:
    """Return indices of signals whose name contains any keyword (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    return [i for i, name in enumerate(signal_names)
            if any(k in name.lower() for k in kws)]


def subset_feature_matrix(
        X: np.ndarray,
        signal_indices: Sequence[int],
        num_MPs: int,
        mp_indices: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Subset columns of X corresponding to selected signals (and optionally a
    subset of MP indices within each signal).

    X has columns laid out as [signal_0_mp_0, signal_0_mp_1, ..., signal_0_mp_{K-1},
                               signal_1_mp_0, ...] (matches prepare_weights_for_classification).

    Parameters
    ----------
    mp_indices : sequence of ints in [0, num_MPs), optional
        Which MP positions to keep within each selected signal. None = all MPs.
    """
    if mp_indices is None:
        mp_indices = range(num_MPs)
    feat_idx: List[int] = []
    for s in signal_indices:
        base = s * num_MPs
        for m in mp_indices:
            feat_idx.append(base + m)
    return X[:, feat_idx], feat_idx


def write_selected_channels_report(
        out_path: Path,
        experiment_name: str,
        description: str,
        signal_indices: Sequence[int],
        signal_names: Sequence[str],
        feature_names_subset: Sequence[str],
        num_MPs: int,
        num_signals_total: int,
        mp_indices: Optional[Sequence[int]] = None,
) -> None:
    n_sel = len(signal_indices)
    n_mps_used = num_MPs if mp_indices is None else len(list(mp_indices))
    mps_used_str = "all" if mp_indices is None else str(list(mp_indices))
    with open(out_path, 'w') as f:
        f.write(f"Joint-subset classification: {experiment_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Description:             {description}\n")
        f.write(f"Keywords (matched on):   {list(JOINT_KEYWORDS)}\n")
        f.write(f"Signals selected:        {n_sel} / {num_signals_total}\n")
        f.write(f"MP indices used:         {mps_used_str}  "
                f"(MPs/signal in model = {num_MPs})\n")
        f.write(f"Total features used:     {n_sel * n_mps_used}\n")
        f.write(f"Total features dropped:  "
                f"{num_signals_total * num_MPs - n_sel * n_mps_used}\n\n")

        f.write("Signal channels included (index : name):\n")
        f.write("-" * 70 + "\n")
        for i in signal_indices:
            f.write(f"  {i:3d} : {signal_names[i]}\n")

        f.write("\nAll feature columns used (signal_MP):\n")
        f.write("-" * 70 + "\n")
        for name in feature_names_subset:
            f.write(f"  {name}\n")
    print(f"  ✓ Wrote: {out_path}")


def run_supervised_only(
        X: np.ndarray,
        y: np.ndarray,
        clf,
        clf_key: str,
        clf_label: str,
        scale: bool,
        feature_names: Sequence[str],
        out_dir: Path,
        seed: int = 42,
        test_size: float = 0.25,
        cv_folds: int = 5,
) -> dict:
    """
    Train + evaluate a single classifier with the same conventions as the
    shared pipeline (split, scaling, CV, confusion matrix, report, feature
    importance) — but no PCA/t-SNE/RDM/legend.
    """
    # 1) Split + scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    if scale:
        scaler = StandardScaler()
        X_train_use = scaler.fit_transform(X_train)
        X_test_use = scaler.transform(X_test)
    else:
        X_train_use, X_test_use = X_train, X_test

    # 2) Fit + predict
    clf = clone(clf)
    clf.fit(X_train_use, y_train)
    y_train_pred = clf.predict(X_train_use)
    y_test_pred = clf.predict(X_test_use)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"  [{clf_label}]  train={train_acc:.4f}  test={test_acc:.4f}")

    # 3) Classification report
    rep_str = classification_report(y_test, y_test_pred, labels=np.unique(y_test))
    save_classification_report(
        rep_str, str(out_dir),
        filename=f"classification_report_{clf_key}.txt",
    )

    # 4) Confusion matrix (canonical paper-style helper)
    labels_sorted = [str(lbl) for lbl in sorted(np.unique(y))]
    _plot_confusion_matrix_percent(
        y_true=y_test, y_pred=y_test_pred,
        out_path=out_dir / f"confusion_matrix_{clf_key}.png",
        labels=labels_sorted, vmin=0.0, vmax=1.0,
    )

    # 5) Feature importance (when supported)
    try:
        plot_and_save_feature_importance(
            model=clf,
            feature_names=list(feature_names),
            topn=min(20, X.shape[1]),
            out_dir=out_dir,
            filename=f"feature_importance_{clf_key}.png",
        )
    except (AttributeError, TypeError):
        pass

    # 6) Cross-validation (using the canonical helper)
    cv_results = _perform_cross_validation(
        X, y, clone(clf.__class__(**clf.get_params())),
        clf_label, seed=seed, cv_folds=cv_folds, scale_data=scale,
    )
    _save_cross_validation_results(
        cv_results, out_dir / f"cross_validation_{clf_key}.txt", clf_label,
    )
    print(f"  [{clf_label}]  CV acc = "
          f"{cv_results['accuracy_test_mean']:.4f} ± "
          f"{cv_results['accuracy_test_std']:.4f}")

    return {
        "key": clf_key, "label": clf_label,
        "train_acc": train_acc, "test_acc": test_acc,
        "n_train": len(y_train), "n_test": len(y_test),
        "cv_acc_mean": cv_results["accuracy_test_mean"],
        "cv_acc_std": cv_results["accuracy_test_std"],
    }


def run_experiment(
        experiment_name: str,
        description: str,
        X_full: np.ndarray,
        y: np.ndarray,
        signal_indices: Sequence[int],
        full_feature_names: Sequence[str],
        num_signals: int,
        num_MPs: int,
        out_dir_root: Path,
        classifier_specs: Sequence[dict],
        mp_indices: Optional[Sequence[int]] = None,
        seed: int = 42,
) -> List[dict]:
    """
    Run a single subset experiment end-to-end and write all artifacts to
    ``out_dir_root / experiment_name``. Returns the per-classifier summary
    rows so callers can build a cross-experiment comparison table.
    """
    out_dir = out_dir_root / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 78}")
    print(f"# Experiment: {experiment_name}")
    print(f"# {description}")
    print(f"{'#' * 78}")

    if len(signal_indices) == 0:
        raise RuntimeError(f"[{experiment_name}] No signals selected.")

    X_sub, feat_idx = subset_feature_matrix(
        X_full, signal_indices, num_MPs, mp_indices=mp_indices,
    )
    feature_names_subset = [full_feature_names[i] for i in feat_idx]

    n_mps_used = num_MPs if mp_indices is None else len(list(mp_indices))
    print(f"  feature matrix: {X_sub.shape}  "
          f"({len(signal_indices)} signals × {n_mps_used} MPs)")

    write_selected_channels_report(
        out_path=out_dir / "selected_channels.txt",
        experiment_name=experiment_name,
        description=description,
        signal_indices=signal_indices,
        signal_names=SIGNAL_NAMES,
        feature_names_subset=feature_names_subset,
        num_MPs=num_MPs,
        num_signals_total=num_signals,
        mp_indices=mp_indices,
    )

    summary_rows = []
    for spec in classifier_specs:
        print(f"\n=== {spec['label']} ===")
        row = run_supervised_only(
            X=X_sub, y=y, clf=spec["model"],
            clf_key=spec["key"], clf_label=spec["label"],
            scale=spec["scale"],
            feature_names=feature_names_subset,
            out_dir=out_dir, seed=seed, cv_folds=5,
        )
        row["experiment"] = experiment_name
        row["n_features"] = X_sub.shape[1]
        summary_rows.append(row)

    # Per-experiment summary table
    with open(out_dir / "subset_summary.txt", "w") as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"  {description}\n")
        f.write("=" * 78 + "\n")
        f.write(f"Signals selected:   {len(signal_indices)} / {num_signals}\n")
        mps_used_str = "all" if mp_indices is None else str(list(mp_indices))
        f.write(f"MP indices used:    {mps_used_str}\n")
        f.write(f"Feature dim used:   {X_sub.shape[1]}\n\n")
        f.write(f"{'Classifier':<18}{'Train acc':>12}{'Test acc':>12}"
                f"{'CV mean':>12}{'CV std':>10}{'n_train':>10}{'n_test':>10}\n")
        f.write("-" * 78 + "\n")
        for s in summary_rows:
            f.write(
                f"{s['label']:<18}"
                f"{s['train_acc']:>12.4f}{s['test_acc']:>12.4f}"
                f"{s['cv_acc_mean']:>12.4f}{s['cv_acc_std']:>10.4f}"
                f"{s['n_train']:>10d}{s['n_test']:>10d}\n"
            )
    print(f"  ✓ Wrote: {out_dir / 'subset_summary.txt'}")
    return summary_rows


def main() -> None:
    # ---------- Configuration (mirrors L1_feature_selection_analysis.main) ----------
    num_MPs = 5
    tpoints = 35
    seed = 42

    model_dir = os.path.join(
        "./../../results/tmp_configs",
        f"new_seg_mp_model_{num_MPs}_phase_three",
    )
    model_file = os.path.join(model_dir, f"mp_model_{num_MPs}_PC_tpoints_{tpoints}")
    out_dir = Path(model_dir) / "joint_subset_classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    folder_path = "../../data/pymotion_position_csv_files"

    # ---------- Load data ----------
    print("Loading data ...")
    motion_ids, processed_segments, segment_motion_ids = process_motion_data(
        folder_path=folder_path, data_type="position", filtering=False,
    )
    num_segments = len(processed_segments)
    num_signals = processed_segments[0].shape[0]
    print(f"  segments={num_segments}  signals={num_signals}")

    # ---------- Load TMP model ----------
    print("Loading TMP model ...")
    model = load_model_with_full_state(
        model_file, num_segments=num_segments, num_signals=num_signals,
    )

    # ---------- Build full feature matrix (reuse) ----------
    X_full = prepare_weights_for_classification(
        model, num_segments=num_segments, num_signals=num_signals, num_MPs=num_MPs,
    )
    y = np.array(segment_motion_ids)
    full_feature_names = create_feature_names(num_signals, num_MPs)
    print(f"  full feature matrix: {X_full.shape}")

    # ---------- Define joint partitions ----------
    keep_indices = select_signal_indices(SIGNAL_NAMES, JOINT_KEYWORDS)
    if len(keep_indices) == 0:
        raise RuntimeError("No signals matched JOINT_KEYWORDS; check SIGNAL_NAMES.")
    drop_indices = [i for i in range(num_signals) if i not in set(keep_indices)]
    print(f"  Joints kept (kw match): {len(keep_indices)} / {num_signals}")
    print(f"  Joints excluded:        {len(drop_indices)} / {num_signals}")

    # ---------- Classifiers (shared across experiments) ----------
    classifier_specs = [
        {
            "key": "linear_svc",
            "label": "Linear SVC",
            "model": LinearSVC(C=1.0, max_iter=10000, random_state=seed),
            "scale": True,
        },
        {
            "key": "random_forest",
            "label": "Random Forest",
            "model": RandomForestClassifier(n_estimators=200, random_state=seed),
            "scale": False,
        },
    ]

    # ---------- Three experiments ----------
    # 1) Original: keep the 5 listed joints, all xyz, all MPs
    # 2) Inverse:  EXCLUDE those 5 joints (= use everything else), all MPs
    # 3) MP0-only: same 5 joints / all xyz, but only MP index 0 per channel
    experiments = [
        dict(
            experiment_name="keep_5_joints_all_MPs",
            description=(f"Keep only {list(JOINT_KEYWORDS)} (L+R, x/y/z) "
                         f"with all {num_MPs} MPs per channel."),
            signal_indices=keep_indices,
            mp_indices=None,
        ),
        dict(
            experiment_name="exclude_5_joints_all_MPs",
            description=(f"Exclude {list(JOINT_KEYWORDS)} — use all OTHER "
                         f"signals with all {num_MPs} MPs per channel."),
            signal_indices=drop_indices,
            mp_indices=None,
        ),
        dict(
            experiment_name="keep_5_joints_MP0_only",
            description=(f"Keep only {list(JOINT_KEYWORDS)} (L+R, x/y/z), "
                         f"but only MP index 0 per channel."),
            signal_indices=keep_indices,
            mp_indices=[0],
        ),
    ]

    all_rows: List[dict] = []
    for exp in experiments:
        rows = run_experiment(
            experiment_name=exp["experiment_name"],
            description=exp["description"],
            X_full=X_full, y=y,
            signal_indices=exp["signal_indices"],
            full_feature_names=full_feature_names,
            num_signals=num_signals, num_MPs=num_MPs,
            out_dir_root=out_dir,
            classifier_specs=classifier_specs,
            mp_indices=exp["mp_indices"],
            seed=seed,
        )
        all_rows.extend(rows)

    # ---------- Cross-experiment comparison table ----------
    cmp_path = out_dir / "experiments_comparison.txt"
    with open(cmp_path, "w") as f:
        f.write("Joint-subset classification — cross-experiment comparison\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"{'Experiment':<28}{'Classifier':<18}{'#feat':>7}"
                f"{'Train':>10}{'Test':>10}{'CV mean':>11}{'CV std':>10}\n")
        f.write("-" * 110 + "\n")
        for s in all_rows:
            f.write(
                f"{s['experiment']:<28}{s['label']:<18}"
                f"{s['n_features']:>7d}"
                f"{s['train_acc']:>10.4f}{s['test_acc']:>10.4f}"
                f"{s['cv_acc_mean']:>11.4f}{s['cv_acc_std']:>10.4f}\n"
            )
    print(f"\n  ✓ Wrote: {cmp_path}")
    print(f"\n✓ Done. Artifacts under: {out_dir}")


if __name__ == "__main__":
    main()
