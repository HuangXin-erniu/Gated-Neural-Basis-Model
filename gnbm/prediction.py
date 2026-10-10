"""Batch inference with ensemble averaging on the log1p scale."""
from __future__ import annotations

import numpy as np
import torch

from ._utils import choose_device, matrix, require


def predict_ensemble(models, x, *, device="cpu", chunk_size=8192):
    """Return (nonnegative_mean_log1p, response) for one or more GNBM models.

    Features must be preprocessed in the training feature order. Models are
    moved to the requested device and switched to evaluation mode.
    """
    models = list(models)
    require(len(models) > 0, "At least one fitted model is required")
    require(isinstance(chunk_size, int) and chunk_size > 0, "chunk_size must be a positive integer")
    names = models[0].feature_names
    require(all(model.feature_names == names for model in models), "Ensemble feature order differs")
    rows = matrix(x, len(names), name="x", allow_empty=True)
    device = choose_device(device)
    seed_predictions = []
    for model in models:
        model.to(device).eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(rows), chunk_size):
                batch = torch.as_tensor(rows[start:start + chunk_size], device=device)
                predictions.append(model(batch).detach().cpu().numpy().astype(float))
        seed_predictions.append(np.concatenate(predictions) if predictions else np.empty(0, dtype=float))
    mean_log = np.mean(np.vstack(seed_predictions), axis=0)
    if not np.isfinite(mean_log).all():
        raise FloatingPointError("GNBM ensemble emitted nonfinite predictions")
    mean_log = np.maximum(mean_log, 0.)
    with np.errstate(over="ignore", invalid="ignore"):
        response = np.expm1(mean_log)
    if not np.isfinite(response).all():
        raise FloatingPointError("Inverse-transformed prediction is nonfinite")
    return mean_log, response
