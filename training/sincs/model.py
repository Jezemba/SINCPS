# Copyright 2024 Cerebras Systems.
#
# SINCS: Semantic-aware Implicit Neural Compression for Simulations
# Optimized for Cerebras CSX compilation

import math
from typing import List, Literal, Optional

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from typing_extensions import Annotated

import cerebras.pytorch as cstorch
from cerebras.modelzoo.config import ModelConfig


class SINCSConfig(ModelConfig):
    """Configuration for SINCS implicit neural representation model."""

    name: Literal["sincs"]
    """Name of the model."""

    input_dim: int = 3
    """Input coordinate dimensions (e.g., 3 for x,y,t)."""

    output_dim: int = 1
    """Output field dimensions."""

    hidden_dim: int = 256
    """Hidden layer width."""

    num_hidden_layers: int = 3
    """Number of hidden layers."""

    omega_0: float = 30.0
    """SIREN frequency parameter for first layer."""

    omega_hidden: float = 30.0
    """SIREN frequency parameter for hidden layers."""

    use_fourier_encoding: bool = True
    """Whether to use Fourier positional encoding."""

    encoding_levels: int = 10
    """Number of Fourier encoding frequency levels."""

    dropout: Annotated[float, Ge(0), Le(1)] = 0.0
    """Dropout probability."""

    @property
    def encoded_dim(self) -> int:
        """Calculate encoded input dimension."""
        if self.use_fourier_encoding:
            return self.input_dim * (2 * self.encoding_levels + 1)
        else:
            return self.input_dim

    @property
    def __model_cls__(self):
        return SINCS


class SirenLayer(nn.Module):
    """SIREN layer with sin activation and special initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False
    ):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.linear.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class FourierEncoder(nn.Module):
    """Fourier positional encoding for coordinates."""

    def __init__(self, input_dim: int, num_levels: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        # Pre-compute frequency multipliers: 2^0, 2^1, ..., 2^(L-1)
        freqs = torch.pow(2.0, torch.arange(num_levels, dtype=torch.float32))
        self.register_buffer('freqs', freqs * math.pi)

    def forward(self, x):
        # x: [batch, input_dim]
        # Expand x for broadcasting: [batch, input_dim, 1]
        x_expanded = x.unsqueeze(-1)
        # freqs: [num_levels] -> [1, 1, num_levels]
        freqs = self.freqs.view(1, 1, -1)
        # Compute sin and cos: [batch, input_dim, num_levels]
        x_freq = x_expanded * freqs
        sin_enc = torch.sin(x_freq)
        cos_enc = torch.cos(x_freq)
        # Concatenate: [batch, input_dim * 2 * num_levels]
        encoded = torch.cat([sin_enc, cos_enc], dim=-1).flatten(start_dim=1)
        # Append original coordinates
        return torch.cat([x, encoded], dim=-1)


class SINCS(nn.Module):
    """SINCS network: SIREN with optional Fourier encoding."""

    def __init__(self, config: SINCSConfig):
        super().__init__()

        self.use_fourier_encoding = config.use_fourier_encoding

        # Fourier encoder
        if self.use_fourier_encoding:
            self.encoder = FourierEncoder(config.input_dim, config.encoding_levels)
            layer_input_dim = config.encoded_dim
        else:
            self.encoder = None
            layer_input_dim = config.input_dim

        # Build SIREN network
        layers = []

        # First layer
        layers.append(SirenLayer(
            layer_input_dim,
            config.hidden_dim,
            omega=config.omega_0,
            is_first=True
        ))

        # Hidden layers
        for _ in range(config.num_hidden_layers - 1):
            layers.append(SirenLayer(
                config.hidden_dim,
                config.hidden_dim,
                omega=config.omega_hidden,
                is_first=False
            ))

        self.layers = nn.ModuleList(layers)

        # Output layer (linear, no sin activation)
        self.output_layer = nn.Linear(config.hidden_dim, config.output_dim)
        self._init_output_layer()

        # Optional dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

    def _init_output_layer(self):
        with torch.no_grad():
            bound = math.sqrt(6.0 / self.output_layer.in_features) / 30.0
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()

    def forward(self, coords):
        """
        Args:
            coords: [batch, input_dim] coordinate tensor
        Returns:
            predictions: [batch, output_dim] field values
        """
        # Encode coordinates
        if self.encoder is not None:
            x = self.encoder(coords)
        else:
            x = coords

        # Pass through SIREN layers
        for layer in self.layers:
            x = layer(x)
            if self.dropout is not None:
                x = self.dropout(x)

        # Output projection
        return self.output_layer(x)


class SINCSModelConfig(SINCSConfig):
    """Full model config including training settings."""

    name: Literal["sincs"]
    """Name of the model."""

    loss_type: Literal["mse", "l1", "smooth_l1"] = "mse"
    """Loss function type."""

    to_float16: bool = True
    """Whether to use bfloat16 precision."""


class SINCSModel(nn.Module):
    """SINCS model wrapper for Cerebras trainer."""

    def __init__(self, config: SINCSModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = SINCSModelConfig(**config)

        super().__init__()

        self.model = self.build_model(config)
        self.loss_fn = self._get_loss_fn(config.loss_type)

    def build_model(self, config: SINCSModelConfig):
        model = SINCS(config)
        if config.to_float16:
            return model.to(cstorch.amp.get_half_dtype())
        return model

    def _get_loss_fn(self, loss_type: str):
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "smooth_l1":
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, data):
        """
        Forward pass for training.

        Args:
            data: tuple of (coords, targets)
                coords: [batch, input_dim] coordinate tensor
                targets: [batch, output_dim] target field values

        Returns:
            loss: scalar loss tensor
        """
        coords, targets = data
        predictions = self.model(coords)
        loss = self.loss_fn(predictions, targets)
        return loss
