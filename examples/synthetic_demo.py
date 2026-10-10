"""Short CPU demonstration with synthetic model inputs only."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from gnbm import DEFAULT_SEEDS, fit_gnbm, predict_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261010)
    names = ["feature_1", "feature_2", "feature_3"]
    # Synthetic arrays are supplied directly; no dataset preprocessing is included.
    x = rng.normal(size=(384, len(names))).astype(np.float32)
    y_log = np.maximum(1.0 + 0.4 * np.sin(x[:, 0]) + 0.2 * x[:, 1]
                       + 0.1 * x[:, 2] ** 2 + rng.normal(0, 0.05, len(x)), 0.)
    models, epochs = [], []
    for seed in DEFAULT_SEEDS:
        model, epoch, _, _ = fit_gnbm(
            names, x[:256], y_log[:256], x_valid=x[256:320], y_valid_log1p=y_log[256:320],
            seed=seed, device="cpu", max_epochs=args.epochs)
        models.append(model)
        epochs.append(epoch)
    predicted_log, predicted_response = predict_ensemble(models, x[320:])
    assert np.isfinite(predicted_response).all()
    with torch.no_grad():
        raw, aux = models[0](torch.from_numpy(x[320:]), return_aux=True)
        reconstructed = aux["feature_contributions"].sum(dim=1) + models[0].output_bias
        torch.testing.assert_close(raw, reconstructed)
    print(f"Selected epochs: {epochs}")
    print(f"Synthetic test MAE (log1p): {np.abs(predicted_log - y_log[320:]).mean():.6f}")
    print(f"Per-model feature contributions: {tuple(aux['feature_contributions'].shape)}")
    print("These contextual contributions reconstruct raw log1p outputs; they are not causal effects.")


if __name__ == "__main__":
    main()
