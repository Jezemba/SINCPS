"""
SINCPS: Semantic-aware Implicit Neural Compression for Physics Simulations

Minimal model definition for loading and decompressing trained checkpoints.
"""

import math
import torch
import torch.nn as nn


class FourierEncoder(nn.Module):
    """Multi-scale Fourier positional encoding with frequencies from π to 512π."""

    def __init__(self, input_dim: int, num_levels: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.num_levels = num_levels
        freqs = torch.pow(2.0, torch.arange(num_levels, dtype=torch.float32))
        self.register_buffer('freqs', freqs * math.pi)

    @property
    def output_dim(self) -> int:
        return self.input_dim * (2 * self.num_levels + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(-1)
        freqs = self.freqs.view(1, 1, -1)
        x_freq = x_expanded * freqs
        sin_enc = torch.sin(x_freq)
        cos_enc = torch.cos(x_freq)
        encoded = torch.cat([sin_enc, cos_enc], dim=-1).flatten(start_dim=1)
        return torch.cat([x, encoded], dim=-1)


class SirenLayer(nn.Module):
    """SIREN layer with sinusoidal activation."""

    def __init__(self, in_features: int, out_features: int,
                 omega: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class SINCPS(nn.Module):
    """
    SINCPS model: SIREN network with Fourier positional encoding.

    Maps spatiotemporal coordinates to physical field values, producing
    a continuous implicit representation of simulation data.

    Args:
        input_dim: Number of input coordinates (e.g., 3 for 2D+time, 4 for 3D+time)
        output_dim: Number of output fields
        hidden_dim: Hidden layer dimension (default: 1024)
        num_hidden_layers: Number of hidden layers (default: 4)
        omega_0: Frequency for first SIREN layer
        omega_hidden: Frequency for hidden SIREN layers
        use_fourier_encoding: Whether to use Fourier positional encoding
        encoding_levels: Number of Fourier frequency levels
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 1,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 4,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        use_fourier_encoding: bool = True,
        encoding_levels: int = 10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_fourier_encoding = use_fourier_encoding

        if use_fourier_encoding:
            self.encoder = FourierEncoder(input_dim, encoding_levels)
            layer_input_dim = self.encoder.output_dim
        else:
            self.encoder = None
            layer_input_dim = input_dim

        layers = []
        layers.append(SirenLayer(layer_input_dim, hidden_dim, omega=omega_0, is_first=True))
        for _ in range(num_hidden_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega_hidden))
        self.layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: map coordinates to field values.

        Args:
            coords: Input coordinates, shape (batch_size, input_dim)
                    Coordinates should be normalized to [0, 1]

        Returns:
            Field values, shape (batch_size, output_dim)
            Values are z-score normalized; denormalize using dataset statistics
        """
        if self.encoder is not None:
            x = self.encoder(coords)
        else:
            x = coords

        for layer in self.layers:
            x = layer(x)

        return self.output_layer(x)

    @classmethod
    def from_config(cls, config: dict) -> "SINCPS":
        """Create model from configuration dictionary."""
        return cls(
            input_dim=config.get('input_dim', 3),
            output_dim=config.get('output_dim', 1),
            hidden_dim=config.get('hidden_dim', 1024),
            num_hidden_layers=config.get('num_hidden_layers', 4),
            omega_0=config.get('omega_0', 30.0),
            omega_hidden=config.get('omega_hidden', 30.0),
            use_fourier_encoding=config.get('use_fourier_encoding', True),
            encoding_levels=config.get('encoding_levels', 10),
        )
