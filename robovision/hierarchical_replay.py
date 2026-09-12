"""Successor reset curriculum from hash-verified actual predecessor arrivals.

Pools contain trusted simulator snapshots from approach-policy rollouts.
They are simulation reset states, never action targets or a replay PPO buffer.
Every training action is sampled afresh from the successor PPO policy.
"""

import gzip
import pickle
from pathlib import Path
import gymnasium as gym
import numpy as np
from .skill_hard_evaluation import sha256
from .skill_state import restore_skill_handoff

POOL_VERSION = "hierarchical-arrival-pool-v1"


def load_arrival_pool(path, expected_sha256, target_phase):
    path = Path(path)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or sha256(path) != expected_sha256
    ):
        raise ValueError("Arrival pool bytes differ from the recorded probe hash")
    with gzip.open(path, "rb") as stream:
        pool = pickle.load(stream)
    source_phase = {"grasp": "approach", "lift": "grasp"}.get(target_phase)
    if (
        not isinstance(pool, dict)
        or pool.get("version") != POOL_VERSION
        or source_phase is None
        or (pool.get("phase") != source_phase)
    ):
        raise ValueError("Arrival pool phase does not match the receiving specialist")
    snapshots = pool.get("snapshots")
    entries = pool.get("entries")
    if (
        not isinstance(snapshots, list)
        or not 1 <= len(snapshots) <= 512
        or (not isinstance(entries, list))
        or (len(entries) != len(snapshots))
    ):
        raise ValueError(
            "Require a nonempty bounded arrival pool with aligned entry metadata"
        )
    for snapshot in snapshots:
        if (
            snapshot.get("phase") != source_phase
            or snapshot.get("phase_success") is not True
            or (not 0 < snapshot.get("step_count", 500) < 500)
        ):
            raise ValueError(
                "Pool contains an unverified or exhausted predecessor arrival"
            )
    return pool


class ArrivalResetEnv(gym.Wrapper):
    """Mix actual arrival resets with the specialist's existing synthetic starts."""

    def __init__(self, env, *, pool_path, pool_sha256, probability=0.75, seed=0):
        super().__init__(env)
        if not np.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("Arrival probability must lie in (0,1]")
        self.pool = load_arrival_pool(pool_path, pool_sha256, env.unwrapped.phase)
        self.pool_path = str(pool_path)
        self.pool_sha256 = pool_sha256
        self.arrival_probability = float(probability)
        self.arrival_rng = np.random.default_rng(seed)
        self._arrival_entry = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.arrival_rng = np.random.default_rng(seed)
        observation, info = self.env.reset(seed=seed, options=options)
        self._arrival_entry = None
        if self.arrival_rng.random() < self.arrival_probability:
            index = int(self.arrival_rng.integers(len(self.pool["snapshots"])))
            observation, info = restore_skill_handoff(
                self.env, self.pool["snapshots"][index]
            )
            self._arrival_entry = dict(index=index, **self.pool["entries"][index])
        return (observation, self._info(info))

    def _info(self, info):
        return {
            **info,
            "actual_arrival_reset": self._arrival_entry is not None,
            "arrival_entry": self._arrival_entry,
            "arrival_pool_sha256": self.pool_sha256,
        }

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (observation, reward, terminated, truncated, self._info(info))

    def specification(self):
        return {
            **self.env.unwrapped.specification(),
            "arrival_reset_curriculum": True,
            "arrival_pool": self.pool_path,
            "arrival_pool_sha256": self.pool_sha256,
            "arrival_count": len(self.pool["snapshots"]),
            "arrival_probability": self.arrival_probability,
            "preserve_original_clock_velocity_force_and_image_history": True,
            "no_action_labels_or_behavior_cloning": True,
        }


def write_arrival_pool(path, phase, entries, *, identity, sources, protocol):
    payload = dict(
        version=POOL_VERSION,
        phase=phase,
        trusted_pickle_only=True,
        oracle_handoff_diagnostic=True,
        deployment_candidate=False,
        no_imitation_objective=True,
        no_action_targets=True,
        source_phase=phase,
        target_phase="grasp" if phase == "approach" else "lift",
        identity=identity,
        source_provenance=sources,
        protocol=protocol,
        snapshots=[entry["snapshot"] for entry in entries],
        entries=[
            {key: value for key, value in entry.items() if key != "snapshot"}
            for entry in entries
        ],
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return dict(
        filename=path.name,
        sha256=sha256(path),
        bytes=path.stat().st_size,
        entries=len(entries),
        ready_for_successor_resets=bool(entries),
        target_phase=payload["target_phase"],
        trusted_pickle_only=True,
        no_action_targets=True,
        no_imitation_objective=True,
    )
