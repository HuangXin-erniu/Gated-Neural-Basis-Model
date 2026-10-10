"""Manuscript AdamW training with raw-MAE stopping and clipped-MAE scoring."""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch

from ._utils import choose_device, matrix, require, target
from .defaults import DEFAULT_TRAINING, DEFAULT_SEEDS
from .model import GNBM


def fit_gnbm(feature_names, x_train, y_train_log1p, *,
             x_valid=None, y_valid_log1p=None, seed=DEFAULT_SEEDS[0],
             device="cpu", training_config=None, model_parameters=None,
             history_path=None, max_epochs=None, verbose=False):
    """Fit one seed model; return (model, best_epoch, validation_mae, history).

    Inputs are preprocessed feature matrices and already-transformed log1p
    targets. Partial configuration overrides are allowed. Early stopping uses
    raw log1p MAE. The returned validation_mae scores the retained model after
    clipping outputs to zero, for candidate ranking. With no validation all
    requested epochs run and validation_mae is None.
    """
    config = dict(DEFAULT_TRAINING)
    if training_config is not None:
        require(not (set(training_config) - set(config)), "Unknown training configuration keys")
        config.update(training_config)
    require(config["optimizer"] == "AdamW", "Only AdamW is supported")
    require(config["loss"] == "MSE", "Only MSE on log1p targets is supported")
    require(config["validation_metric"] == "MAE", "Only log1p MAE is supported")
    require(isinstance(seed, int) and 0 <= seed < 2**63, "seed must be a nonnegative integer below 2**63")
    require((x_valid is None) == (y_valid_log1p is None), "Incomplete validation inputs")
    maximum = config["maximum_epochs"] if max_epochs is None else max_epochs
    for name, value in (("maximum_epochs", maximum),
                        ("early_stopping_patience", config["early_stopping_patience"]),
                        ("batch_size", config["batch_size"]),
                        ("validation_batch_size", config["validation_batch_size"])):
        require(isinstance(value, int) and value > 0, f"{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay", "gradient_norm_clip"):
        value = float(config[name])
        require(math.isfinite(value) and value >= 0, f"{name} must be finite and nonnegative")
    require(float(config["gradient_norm_clip"]) > 0, "gradient_norm_clip must be positive")
    feature_names = tuple(feature_names)
    n_features = len(feature_names)
    rows = matrix(x_train, n_features, name="x_train")
    targets = target(y_train_log1p, len(rows), name="y_train_log1p")
    valid_rows = None if x_valid is None else matrix(x_valid, n_features, name="x_valid")
    valid_targets = None if valid_rows is None else target(y_valid_log1p, len(valid_rows), name="y_valid_log1p")
    device = choose_device(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = GNBM(feature_names, model_parameters).to(device)
    x = torch.as_tensor(rows, device=device)
    y = torch.as_tensor(targets, device=device)
    xv = None if valid_rows is None else torch.as_tensor(valid_rows, device=device)
    yv = None if valid_targets is None else torch.as_tensor(valid_targets, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch_size = config["batch_size"]
    validation_batch = config["validation_batch_size"]
    patience = config["early_stopping_patience"]
    best_score, best_epoch, stale = math.inf, maximum, 0
    best_state, history = None, []
    for epoch in range(1, maximum + 1):
        epoch_started = time.perf_counter()
        model.train()
        permutation = torch.randperm(len(x), generator=generator, device=device)
        loss_sum = 0.
        for start in range(0, len(x), batch_size):
            indexes = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indexes])
            loss = (prediction - y[indexes]).square().mean()
            require(bool(torch.isfinite(loss)), f"Nonfinite MSE at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_norm_clip"]),
                                           error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indexes)
        score = None
        model.eval()
        if xv is not None:
            absolute_sum = 0.
            with torch.no_grad():
                for start in range(0, len(xv), validation_batch):
                    error = (model(xv[start:start + validation_batch]) - yv[start:start + validation_batch]).abs()
                    absolute_sum += float(error.double().sum())
            score = absolute_sum / len(xv)
            require(math.isfinite(score), "Nonfinite validation MAE")
            if score < best_score - 1e-6:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append({"seed": seed, "epoch": epoch, "train_mse_log1p": loss_sum / len(x),
                        "validation_mae_log1p_raw": score, "best_epoch_so_far": best_epoch,
                        "elapsed_seconds": time.perf_counter() - epoch_started})
        if history_path is not None:
            destination = Path(history_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"seed={seed} epoch={epoch}/{maximum} train_MSE={loss_sum / len(x):.6f}"
                  + (f" valid_MAE={score:.6f} best_epoch={best_epoch}" if score is not None else ""),
                  flush=True)
        if xv is not None and stale >= patience:
            break
    if xv is not None:
        require(best_state is not None, "No finite retained validation state")
        model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError("Nonfinite fitted model parameters")
    validation_mae = None
    with torch.no_grad():
        # Check the retained final state, including the last optimizer update.
        for start in range(0, len(x), validation_batch):
            model(x[start:start + validation_batch])
        if xv is not None:
            # Keep full target precision for candidate scoring; optimization
            # and raw early-stopping calculations retain their float32 rule.
            scoring_targets = torch.as_tensor(
                target(y_valid_log1p, len(xv), name="y_valid_log1p", dtype="float64"),
                device=device)
            absolute_sum = 0.
            for start in range(0, len(xv), validation_batch):
                clipped = model(xv[start:start + validation_batch]).double().clamp_min(0.)
                absolute_sum += float((clipped - scoring_targets[start:start + validation_batch]).abs().sum())
            validation_mae = absolute_sum / len(xv)
            require(math.isfinite(validation_mae), "Nonfinite clipped validation MAE")
    return model, best_epoch, validation_mae, history
