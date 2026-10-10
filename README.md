# Gated Neural Basis Model

A standalone PyTorch implementation of the GNBM core: shared neural basis functions, full-basis context gating, training, ensemble prediction, and contribution decomposition.


## Running the example

The implementation has been verified with Python 3.12.13, PyTorch 2.5.1, and NumPy 2.4.6. Use an environment with these dependencies and run the following command from the repository root. Installing this repository as a package is unnecessary.

```powershell
python -B -m examples.synthetic_demo
```

The example runs on CPU with randomly generated input arrays. It neither reads nor writes data files and does not implement dataset preprocessing such as standardization. The demonstration uses a maximum of 20 training epochs; add `--epochs 5` for a shorter run. Callers supply their own preprocessed feature matrices and log1p-transformed targets.

## Core interfaces

`GNBM(feature_names, parameters=None)` constructs the model. Each scalar feature passes through the same basis network. Hidden blocks contain Linear, LayerNorm, GELU, and Dropout; the basis output layer has no normalization. The gate receives the complete input and flattened basis values, applies LayerNorm to their concatenation, and produces logits for every feature and basis channel.

Softmax is applied along the basis dimension. The gate multiplier is `(1-alpha) + alpha*B*q`, with a channel mean of 1 for each feature. The gate output layer is initialized to zero. Temperature and strength are fixed hyperparameters. The raw network prediction is the sum of feature contributions and a global bias; the network does not clip its output.

`fit_gnbm(...)` accepts feature matrices and already-transformed log1p targets, returning the model, best epoch, validation score, and training history. Defaults are AdamW, learning rate `1e-3`, weight decay `1e-5`, batch size `256`, at most `240` epochs, early-stopping patience `48`, and a gradient norm limit of `0.5`. Default training seeds are `1729, 2718, 31415`; optimization uses float32 MSE.

```python
from gnbm import fit_gnbm, predict_ensemble

model, best_epoch, validation_mae, history = fit_gnbm(
    feature_names, X_train, y_train_log1p,
    x_valid=X_valid, y_valid_log1p=y_valid_log1p,
    seed=1729, device="cpu",
)
predicted_log1p, predicted_response = predict_ensemble([model], X_test)
```

Inputs must be finite, with consistent feature ordering. Early stopping uses MAE on raw log1p predictions and restores the best state. The history field `validation_mae_log1p_raw` records this metric. The returned `validation_mae` instead uses nonnegative-clipped predictions from the retained model. Without validation inputs, training runs for the specified number of epochs and returns `None` for the validation score. The default architecture is a runnable example, not the manuscript's final optimal configuration.

`predict_ensemble(...)` first averages each model's raw log1p prediction, then clips the average to nonnegative values and applies `expm1`. It returns predictions on both scales.

`model(x, return_aux=True)` exposes gates, feature contributions, `ungated_feature_contributions` from the same fitted model with gate multipliers set to 1, and their difference, `net_gating_adjustments`. Summing feature contributions and adding the global bias reconstructs the raw log1p output. The ungated diagnostic is not a separately retrained NBM and does not represent causal effects.
