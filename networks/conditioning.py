""" Conditioning modules (e.g., on noise level). """

from typing import *

import math
import torch
from torch import nn


class ConditionalBlock(nn.Module):
    """ Any module where forward() takes an optional conditioning as a second argument. """
    def forward(self, x, conditioning=None):
        raise NotImplementedError


class ConditionalSequential(nn.Sequential, ConditionalBlock):
    """ A sequential module that passes conditioning information to the children that support it as an extra input. """
    def forward(self, x, conditioning=None):
        for layer in self:
            if isinstance(layer, ConditionalBlock):
                x = layer(x, conditioning)
            else:
                x = layer(x)
        return x


def sinusoidal_embedding(timesteps, dim, t_min=1, t_max=100, log_scale=False):
    """ Create sinusoidal embeddings. Typically timesteps are noise variances, compared to pixel value range in [0, 1].
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param t_min, t_max: estimated minimum and maximal values of the timesteps (widest values should be 1e-5 and 1e2, defaults are here for backwards compatibility).
    :param log_scale: if True, puts t in log space with linear frequencies, otherwise puts t in linear space with log frequencies.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert dim % 2 == 0, "dim must be even"
    if log_scale:
        # Put t in log space with linear frequencies.
        timesteps = torch.log(timesteps)
        freqs = (1 + torch.arange(dim // 2, dtype=torch.float32, device=timesteps.device)) / torch.log(t_max / t_min)  # (F,), ranges from 1/t_min to 1/t_max linearly.
        assert False, "not implemented because t = 0 behvaior is undefined"
    else:
        # Use log frequencies.
        freqs = torch.exp(-torch.linspace(math.log(t_min), math.log(t_max), dim // 2, dtype=torch.float32, device=timesteps.device))  # (F,), ranges from 1/t_min to 1/t_max logarithmically.
    args = timesteps[:, None] * freqs[None]  # (B, F)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, 2F)
    return embedding


class NoiseVarEmbedding(nn.Module):
    """ Embedding module for noise variance. """
    def __init__(self, fourier_dim=64, time_embed_dim=256, t_min=1, t_max=100):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.time_embed_dim = time_embed_dim
        self.t_min = t_min
        self.t_max = t_max
        self.time_embed = nn.Sequential(
            nn.Linear(self.fourier_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
        )

    def forward(self, noise_var: torch.Tensor):
        return self.time_embed(sinusoidal_embedding(timesteps=noise_var, dim=self.fourier_dim, t_min=self.t_min, t_max=self.t_max))

    def extra_repr(self):
        return f"fourier_dim={self.fourier_dim}, time_embed_dim={self.time_embed_dim}, t_min={self.t_min}, t_max={self.t_max}"

    def my_named_parameters(self, reduced=True, with_grad=True, prefix="") -> Dict[str, torch.Tensor]:
        """ More convenient version of nn.Module.named_parameters. Overridden by some modules to provide more helpful names.
        Possiblity to return a reduced list (for more concise logging) or filtering parameters that have gradient only.
        For NoiseVarEmbedding, all parameters are included (reduced is ignored).
        """
        parameters = {}

        # Just add both weight and bias parameters of both linear layers.
        for i in range(2):
            layer = self.time_embed[2 * i]
            parameters[f"{prefix}linear{i+1}.weight"] = layer.weight
            parameters[f"{prefix}linear{i+1}.bias"] = layer.bias

        return parameters
