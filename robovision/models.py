"""Inference networks and portable tensor checkpoints."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .vision import image_moments

TARGET_VERSION = "robovision-target-v1"
GRASP_VERSION = "robovision-grasp-v1"
FEATURES = [
    "hand_z", "hand_vz", "left_opening", "right_opening", "left_velocity",
    "right_velocity", "previous_vertical_action", "previous_jaw_action",
    "target_dx", "target_dy", "hand_vx", "hand_vy",
]


class TargetPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(22, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, image, proprio):
        features = torch.cat([image_moments(image), proprio], dim=1)
        return self.layers(features).clamp(-1, 1)

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("Exported actors support deterministic inference only")
        image = np.asarray(observation["image"])
        proprio = np.asarray(observation["proprio"])
        if image.shape != (6, 96, 96) or image.dtype != np.uint8:
            raise ValueError("Expected two 96x96 uint8 RGB frames")
        if proprio.shape != (16,) or not np.isfinite(proprio).all():
            raise ValueError("Expected 16 finite proprioceptive values")
        with torch.no_grad():
            action = self(
                torch.from_numpy(image.copy())[None].float() / 255,
                torch.as_tensor(proprio, dtype=torch.float32)[None],
            )[0].numpy()
        return action, None


class GraspNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(12, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def load_target(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != TARGET_VERSION:
        raise ValueError("Incompatible target checkpoint")
    model = TargetPolicy()
    model.load_state_dict(state["weights"])
    return model.eval()


def load_grasp(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != GRASP_VERSION or state.get("features") != FEATURES:
        raise ValueError("Incompatible grasp checkpoint")
    model = GraspNetwork()
    model.load_state_dict(state["weights"])
    return model.eval()


def export_target(model, path):
    """Save the PPO actor without its critic, optimizer, or pickled classes."""
    actor = TargetPolicy()
    state = model.policy.state_dict()
    weights = {}
    for destination, source in [("0", "mlp_extractor.policy_net.0"),
                                ("2", "mlp_extractor.policy_net.2"),
                                ("4", "action_net")]:
        for field in ("weight", "bias"):
            weights[f"layers.{destination}.{field}"] = state[f"{source}.{field}"].cpu()
    actor.load_state_dict(weights)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"version": TARGET_VERSION, "weights": actor.state_dict()}, temporary)
    temporary.replace(path)
