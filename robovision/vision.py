"""Cyan segmentation and image moments for target prediction."""
from __future__ import annotations

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

VERSION = 'rgb-cyan-moments-v1'


def cup_color_mask(rgb):
    """Batched BCHW RGB float images in [0, 1], returning a BHW pixel mask."""
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return ((green > red + .12) & (blue > red + .12)
            & (green > .80 * blue) & (green > .20) & (red < .65))


def image_moments(image):
    """Six moments of the latest RGB frame; black images produce six zeros."""
    rgb = image[:, -3:]
    mask = cup_color_mask(rgb).to(rgb.dtype)
    height, width = mask.shape[-2:]
    x = torch.linspace(-1., 1., width, device=rgb.device, dtype=rgb.dtype).view(1, 1, width)
    y = torch.linspace(-1., 1., height, device=rgb.device, dtype=rgb.dtype).view(1, height, 1)
    mass = mask.sum(dim=(-2, -1))
    denominator = mass.clamp(min=1.)
    detected = (mass >= 3).to(rgb.dtype)
    center_x = (mask * x).sum(dim=(-2, -1)) / denominator
    center_y = (mask * y).sum(dim=(-2, -1)) / denominator
    spread_x = ((mask * (x - center_x[:, None, None]).square()).sum(dim=(-2, -1)) / denominator).sqrt()
    spread_y = ((mask * (y - center_y[:, None, None]).square()).sum(dim=(-2, -1)) / denominator).sqrt()
    area = (mass / (height * width)).sqrt()
    moments = torch.stack([center_x, center_y, spread_x, spread_y, area, detected], dim=1)
    return moments * detected[:, None]


class CupMomentExtractor(BaseFeaturesExtractor):
    """RGB moments + robot proprioception; the PPO actor learns the mapping."""

    def __init__(self, observation_space):
        if observation_space['image'].shape != (6, 96, 96) or observation_space['proprio'].shape != (16,):
            raise ValueError('Expected two RGB frames and 16 robot proprioceptive values')
        super().__init__(observation_space, features_dim=22)

    def forward(self, observations):
        # SB3 normalizes uint8 camera images to [0, 1] before this call.
        return torch.cat([image_moments(observations['image']), observations['proprio']], dim=1)
