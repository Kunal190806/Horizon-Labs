"""
backend/candidate_validation/model_training.py
Full CNN training pipeline for transit classification.
Uses the Astronet-Triage-style dataset format (precomputed folded light curves).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

INPUT_LENGTH = 201


def prepare_synthetic_dataset(
    n_planet: int = 500,
    n_fp: int = 500,
    noise_level: float = 0.002,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for demonstration purposes.
    Real training should use labeled TESS data (e.g. Astronet-Triage).

    Planet class: box-like transit at centre, low noise
    FP class: v-shaped / noisy / asymmetric signals
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    phase = np.linspace(-0.5, 0.5, INPUT_LENGTH)

    # ── Planet samples (label = 1) ────────────────────────────────────────────
    for _ in range(n_planet):
        depth = rng.uniform(0.002, 0.05)
        duration = rng.uniform(0.02, 0.15)
        ingress = rng.uniform(0.1, 0.4) * duration
        flux = np.ones(INPUT_LENGTH)
        for i, p in enumerate(phase):
            if abs(p) < duration / 2 - ingress:
                flux[i] = 1.0 - depth
            elif abs(p) < duration / 2:
                t = (abs(p) - (duration / 2 - ingress)) / ingress
                flux[i] = 1.0 - depth * (1 - t)
        flux += rng.normal(0, noise_level, INPUT_LENGTH)
        X_list.append(flux)
        y_list.append(1)

    # ── False-Positive samples (label = 0) ────────────────────────────────────
    for _ in range(n_fp):
        fp_type = rng.integers(0, 3)
        flux = np.ones(INPUT_LENGTH)
        if fp_type == 0:  # V-shape (eclipsing binary)
            depth = rng.uniform(0.05, 0.3)
            for i, p in enumerate(phase):
                flux[i] = 1.0 - depth * max(0, 1 - abs(p) / 0.15)
        elif fp_type == 1:  # Secondary eclipse (background EB)
            depth = rng.uniform(0.002, 0.05)
            duration = rng.uniform(0.02, 0.1)
            offset = rng.uniform(-0.2, 0.2)
            for i, p in enumerate(phase):
                if abs(p - offset) < duration / 2:
                    flux[i] = 1.0 - depth
        else:  # Noise burst / instrumental
            flux = np.ones(INPUT_LENGTH) + rng.normal(0, noise_level * 10, INPUT_LENGTH)

        flux += rng.normal(0, noise_level, INPUT_LENGTH)
        X_list.append(flux)
        y_list.append(0)

    X = np.array(X_list, dtype=np.float32).reshape(-1, INPUT_LENGTH, 1)
    y = np.array(y_list, dtype=np.int32)
    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]


def train_model(
    model_save_path: str,
    n_planet: int = 1000,
    n_fp: int = 1000,
    epochs: int = 50,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Train the CNN transit classifier and save weights.
    Returns a dict of training history and evaluation metrics.
    """
    try:
        import tensorflow as tf
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
            precision_recall_curve, average_precision_score,
        )
    except ImportError as e:
        raise RuntimeError(f"Required library not found: {e}")

    logger.info("Generating synthetic training dataset (%d planet, %d FP)...", n_planet, n_fp)
    X, y = prepare_synthetic_dataset(n_planet, n_fp, seed=seed)

    # Train/val/test split
    n = len(X)
    n_test = max(100, int(n * 0.1))
    X_test, y_test = X[:n_test], y[:n_test]
    X_train, y_train = X[n_test:], y[n_test:]

    # Build model
    from backend.candidate_validation.ml_classifier import TransitClassifier
    clf = TransitClassifier()
    model = clf._build_model()
    if model is None:
        raise RuntimeError("TensorFlow not available — cannot train model.")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_accuracy"),
    ]

    # TensorBoard
    tb_log_dir = str(Path(model_save_path).parent / "logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1))

    logger.info("Training CNN: %d samples, %d epochs, batch=%d", len(X_train), epochs, batch_size)
    history = model.fit(
        X_train, y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation
    y_pred_prob = model.predict(X_test, verbose=0)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Precision-recall curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, y_pred_prob)
    avg_precision = float(average_precision_score(y_test, y_pred_prob))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_prob)),
        "average_precision": avg_precision,
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
        "model_path": model_save_path,
        "epochs_trained": len(history.history["loss"]),
        "model_type": "cnn",
    }
    logger.info(
        "Training complete: F1=%.4f, AUC=%.4f, AP=%.4f",
        metrics["f1"], metrics["roc_auc"], metrics["average_precision"]
    )
    return metrics
