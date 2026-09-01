"""Learned visual features for closed-loop joint control; no segmentation."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraspCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        channels = observation_space["image"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            size = self.cnn(torch.zeros(1, *observation_space["image"].shape)).shape[1]
        self.visual = nn.Sequential(nn.Linear(size, 128), nn.ReLU())
        self.robot = nn.Sequential(
            nn.Linear(observation_space["proprio"].shape[0], 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(nn.Linear(192, features_dim), nn.ReLU())

    def forward(self, observations):
        visual = self.visual(self.cnn(observations["image"]))
        robot = self.robot(observations["proprio"])
        return self.fusion(torch.cat([visual, robot], dim=1))


class NormalizedGraspCNN(GraspCNN):
    """Balance sensory branches and bound feature scale before the policy MLP."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.visual = nn.Sequential(self.visual[0], nn.LayerNorm(128), nn.SiLU())
        self.robot = nn.Sequential(self.robot[0], nn.LayerNorm(64), nn.SiLU())
        self.fusion = nn.Sequential(
            self.fusion[0], nn.LayerNorm(features_dim), nn.SiLU()
        )


class AuxiliaryGraspCNN(NormalizedGraspCNN):
    """Learn visual localization from training labels without actor state leakage.

    The cup head reads only camera features. The actor still receives exactly
    the same visual/proprioceptive feature vector as NormalizedGraspCNN.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.cup_head = nn.Linear(128, 3)
        self.register_buffer("cup_origin", torch.tensor([0.32, 0.0, 0.32]))
        self.register_buffer("cup_scale", torch.tensor([0.1, 0.1, 0.1]))
        self._last_visual = None

    def forward(self, observations):
        visual = self.visual(self.cnn(observations["image"]))
        self._last_visual = visual
        robot = self.robot(observations["proprio"])
        return self.fusion(torch.cat([visual, robot], dim=1))

    def predict_cup(self):
        """Training diagnostic for the most recent camera forward, in metres."""
        if self._last_visual is None:
            raise RuntimeError(
                "Run visual feature extraction before predicting the cup"
            )
        return self.cup_origin + self.cup_scale * self.cup_head(self._last_visual)


class PoseGraspCNN(AuxiliaryGraspCNN):
    """Use learned camera localization as the only visual controller input.

    The bottleneck predicts three cup coordinates from camera pixels. It never
    reads simulator coordinates and contains no kinematics or force controller.
    Both the localization network and torque actor remain trainable by PPO.
    """

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3 + 64, features_dim), nn.LayerNorm(features_dim), nn.SiLU()
        )

    def forward(self, observations):
        self._last_visual = self.visual(self.cnn(observations["image"]))
        predicted_cup = self.predict_cup()
        normalized_cup = (predicted_cup - self.cup_origin) / self.cup_scale
        robot = self.robot(observations["proprio"])
        return self.fusion(torch.cat([normalized_cup, robot], dim=1))
