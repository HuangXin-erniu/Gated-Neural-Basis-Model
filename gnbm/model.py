"""Shared basis, full-basis gating and additive output (manuscript Eqs. 5-9)."""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


from .defaults import DEFAULT_PARAMETERS as PARAMETERS


class _InitialBasisNetwork(nn.Module):
    """Initialization-compatible plain basis module, replaced by the selected normalized one."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__()
        parameters = PARAMETERS if parameters is None else parameters
        layers: list[nn.Module] = []
        current = 1
        depth = int(parameters["depth"])
        width = int(parameters["width"])
        n_bases = int(parameters["n_bases"])
        for _ in range(max(1, depth)):
            layers.extend((nn.Linear(current, width), nn.GELU(),
                           nn.Dropout(float(parameters["basis_dropout"]))))
            current = width
        layers.extend((nn.Linear(current, n_bases), nn.GELU()))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _HiddenNormalizedBasis(nn.Module):
    """Hidden LayerNorm blocks; the basis output layer has no normalization."""

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__()
        parameters = PARAMETERS if parameters is None else parameters
        layers: list[nn.Module] = []
        current = 1
        for _ in range(int(parameters["depth"])):
            layers.extend((
                nn.Linear(current, int(parameters["width"])),
                nn.LayerNorm(int(parameters["width"])),
                nn.GELU(),
                nn.Dropout(float(parameters["basis_dropout"])),
            ))
            current = int(parameters["width"])
        layers.extend((
            nn.Linear(current, int(parameters["n_bases"])),
            nn.GELU(),
        ))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class GNBM(nn.Module):
    """Shared unary neural basis model with a bounded full-basis context gate."""

    def __init__(
        self,
        feature_names: list[str] | tuple[str, ...],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        overrides = {} if parameters is None else dict(parameters)
        unknown = set(overrides) - set(PARAMETERS)
        if unknown:
            raise ValueError(f"Unknown architecture parameters: {sorted(unknown)}")
        parameters = {**PARAMETERS, **overrides}
        for key in ("depth", "width", "n_bases", "gate_width"):
            if not isinstance(parameters[key], int) or parameters[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if parameters["width"] < 2:
            raise ValueError("width must be at least 2 to avoid constant LayerNorm bases")
        for key in ("basis_dropout", "gate_dropout"):
            value = float(parameters[key])
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{key} must be finite and in [0, 1)")
        if parameters["gate_input_layernorm"] is not True:
            raise ValueError("The gate requires input LayerNorm")
        if not math.isfinite(float(parameters["gate_temperature"])):
            raise ValueError("gate_temperature must be finite")
        self.architecture_parameters = dict(parameters)
        self.feature_names = tuple(feature_names)
        if (not self.feature_names or
                not all(isinstance(name, str) and name for name in self.feature_names) or
                len(set(self.feature_names)) != len(self.feature_names)):
            raise ValueError("feature_names must be nonempty and unique")
        self.n_unary = len(self.feature_names)
        self.n_bases = int(parameters["n_bases"])
        self.gate_strength = float(parameters["gate_strength"])
        self.gate_temperature = float(parameters.get("gate_temperature", 1.0))
        if parameters.get("basis_norm_mode") != "hidden_only":
            raise ValueError("GNBM implements hidden_only basis normalization")
        if parameters.get("gate_input") != "full_basis":
            raise ValueError("GNBM implements full_basis gate input")
        if not 0.0 <= self.gate_strength <= 1.0:
            raise ValueError("gate_strength must be between 0 and 1")
        temperature32 = torch.tensor(self.gate_temperature, dtype=torch.float32)
        if not bool(torch.isfinite(temperature32)) or float(temperature32) <= 0.:
            raise ValueError("gate_temperature must be positive and finite in float32")

        # Preserve temporary network creation and RNG consumption so fixed-seed
        # initialization stays compatible with the original implementation.
        self.unary_basis = _InitialBasisNetwork(parameters)
        self.unary_coefficients = nn.Parameter(torch.empty(self.n_unary, self.n_bases))
        self.unary_bias = nn.Parameter(torch.zeros(self.n_unary))
        self.output_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("unary_index", torch.arange(self.n_unary, dtype=torch.long))
        self.register_buffer("pair_index", torch.empty((0, 2), dtype=torch.long))
        nn.init.normal_(self.unary_coefficients, mean=0.0, std=0.02)

        gate_input = self.n_unary + self.n_unary * self.n_bases
        gate_layers: list[nn.Module] = []
        current = gate_input
        for _ in range(max(1, int(parameters["depth"]))):
            gate_layers.extend((
                nn.Linear(current, int(parameters["width"])),
                nn.GELU(),
                nn.Dropout(float(parameters["basis_dropout"])),
            ))
            current = int(parameters["width"])
        gate_layers.append(nn.Linear(current, self.n_unary * self.n_bases))
        initial_gate = nn.Sequential(*gate_layers)
        nn.init.zeros_(initial_gate[-1].weight)
        nn.init.zeros_(initial_gate[-1].bias)
        self.gate_network = initial_gate

        self.unary_basis = _HiddenNormalizedBasis(parameters)
        gate_layers = [nn.LayerNorm(gate_input)]
        current = gate_input
        for _ in range(int(parameters["depth"])):
            gate_layers.extend((
                nn.Linear(current, int(parameters["gate_width"])),
                nn.GELU(),
                nn.Dropout(float(parameters["gate_dropout"])),
            ))
            current = int(parameters["gate_width"])
        gate_layers.append(nn.Linear(current, self.n_unary * self.n_bases))
        selected_gate = nn.Sequential(*gate_layers)
        nn.init.zeros_(selected_gate[-1].weight)
        nn.init.zeros_(selected_gate[-1].bias)
        self.gate_network = selected_gate

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise ValueError("x must be a floating-point torch.Tensor")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x contains nonfinite values")
        if x.ndim != 2 or x.shape[1] != self.n_unary:
            raise ValueError(f"Expected input shape [rows, {self.n_unary}]")
        batch = x.shape[0]
        scalar_inputs = x[:, self.unary_index].unsqueeze(-1).reshape(batch * self.n_unary, 1)
        basis = self.unary_basis(scalar_inputs).reshape(batch, self.n_unary, self.n_bases)
        logits = self.gate_network(torch.cat((x, basis.flatten(start_dim=1)), dim=1))
        gate_probability = torch.softmax(
            logits.reshape(batch, self.n_unary, self.n_bases) / self.gate_temperature,
            dim=-1,
        )
        gate_multiplier = (1.0 - self.gate_strength) + (
            self.gate_strength * self.n_bases * gate_probability
        )
        contributions = (
            basis * self.unary_coefficients.unsqueeze(0) * gate_multiplier
        ).sum(dim=-1) + self.unary_bias.unsqueeze(0)
        prediction = contributions.sum(dim=1) + self.output_bias
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("GNBM emitted nonfinite predictions")
        if not return_aux:
            return prediction
        # Same fitted basis and coefficients, with gate multipliers fixed at 1.
        # This is an internal representation diagnostic, not a refitted NBM.
        ungated = (basis * self.unary_coefficients.unsqueeze(0)).sum(dim=-1) + self.unary_bias.unsqueeze(0)
        return prediction, {
            "ungated_feature_contributions": ungated,
            "net_gating_adjustments": contributions - ungated,
            "term_contributions": contributions,
            "feature_contributions": contributions,
            "gate_probability_unary": gate_probability,
            "gate_probability_pair": None,
            "gate_multiplier_unary": gate_multiplier,
        }

