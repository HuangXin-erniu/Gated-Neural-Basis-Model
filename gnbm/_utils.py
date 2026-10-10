"""Internal input and device checks."""
import numpy as np
import torch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def choose_device(requested):
    if str(requested) == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    require(device.type in ("cpu", "cuda"), "Only CPU and CUDA are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError("CUDA device index is unavailable")
    return device


def matrix(values, n_features, *, name, allow_empty=False):
    rows = np.array(values, dtype=np.float32, copy=True)
    require(rows.ndim == 2 and rows.shape[1] == n_features,
            f"{name} must have shape [rows, {n_features}]")
    require(allow_empty or len(rows) > 0, f"{name} must be nonempty")
    require(bool(np.isfinite(rows).all()), f"{name} contains nonfinite values")
    return rows


def target(values, n_rows, *, name, dtype=np.float32):
    values = np.array(values, dtype=dtype, copy=True)
    require(values.shape == (n_rows,), f"{name} must have shape [{n_rows}]")
    require(bool(np.isfinite(values).all()), f"{name} contains nonfinite values")
    return values
