"""Load trusted local grasp checkpoints with the matching PPO optimizer layout."""

import json
from pathlib import Path
from zipfile import ZipFile
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import open_path
from .decoupled_ppo import DecoupledPPO


def checkpoint_algorithm(path):
    """Inspect only ZIP directory and small JSON metadata, never tensor payloads."""
    with open_path(Path(path), mode="r", suffix="zip") as stream:
        with ZipFile(stream) as archive:
            data = json.loads(archive.read("data"))
            marker = data.get("checkpoint_algorithm")
            metadata = data.get("algorithm_metadata", {})
            policy_kwargs = data.get("policy_kwargs", {})
            if (
                marker == "structured-decoupled-ppo-v1"
                or metadata.get("name") == "structured-decoupled-ppo-v1"
            ):
                return "structured-decoupled-ppo-v1"
            if (
                marker == "context-decoupled-ppo-v1"
                or metadata.get("name") == "context-decoupled-ppo-v1"
                or "std_scales" in policy_kwargs
            ):
                return "context-decoupled-ppo-v1"
            return (
                "decoupled-ppo-v1"
                if "critic_optimizer.pth" in archive.namelist()
                else "ppo"
            )


def load_grasp_policy(path, env=None, device="auto", **kwargs):
    """Preserve the correct training algorithm as well as actor/optimizer state.

    Supports ordinary, decoupled, and contextual PPO filesystem checkpoints,
    with or without the .zip suffix. No learning-rate or context overrides are
    introduced. Role-buffer metadata remains compatible with ordinary PPO.
    """
    from .torch_precision import configure_policy_precision

    configure_policy_precision()
    kind = checkpoint_algorithm(path)
    if kind == "structured-decoupled-ppo-v1":
        from .structured_exploration import StructuredPPO

        algorithm = StructuredPPO
    elif kind == "context-decoupled-ppo-v1":
        from .context_exploration import ContextExplorationPPO

        algorithm = ContextExplorationPPO
    else:
        algorithm = DecoupledPPO if kind == "decoupled-ppo-v1" else PPO
    return algorithm.load(path, env=env, device=device, **kwargs)
