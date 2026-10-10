"""Settings from manuscript Sections 2.3 and 2.4.

The default architecture is illustrative, not a reported optimum.
The manuscript does not report a final selected architecture.
"""
DEFAULT_SEEDS = (1729, 2718, 31415)

DEFAULT_PARAMETERS = {
    "depth": 2,
    "width": 64,
    "n_bases": 32,
    "basis_dropout": 0.3,
    "basis_norm_mode": "hidden_only",
    "gate_width": 64,
    "gate_input": "full_basis",
    "gate_input_layernorm": True,
    "gate_temperature": 0.75,
    "gate_dropout": 0.3,
    "gate_strength": 0.5,
}

DEFAULT_TRAINING = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "loss": "MSE",
    "batch_size": 256,
    "maximum_epochs": 240,
    "early_stopping_patience": 48,
    "validation_metric": "MAE",
    "gradient_norm_clip": 0.5,
    "validation_batch_size": 2048,
}
