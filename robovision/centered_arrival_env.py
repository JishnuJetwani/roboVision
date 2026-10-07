"""Exact successful skill arrivals as resets for the centered full pickup task.

Arrival pools are trusted simulator snapshots, never action demonstrations.
Restore preserves the physical clock and observation history. Only the new
task's reward accounting is initialized; a phase endpoint does not terminate
the centered full task. No runtime control is added.
"""

from __future__ import annotations
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import numpy as np
from .centered_grasp_env import CenteredGraspConfig, CenteredGraspEnv, POTENTIAL_KEYS
from .grasp_reward import GraspReward
from .skill_state import (
    PHYSICAL_EPISODE_FIELDS,
    _restore_physics,
    _unwrap,
    _validate_observation,
    _validate_snapshot,
)

VERSION = "centered-full-task-arrivals-v1"
POOL_VERSION = "hierarchical-arrival-pool-v1"
MOVING_GRASP_CLASS = "robovision.centered_grasp_bootstrap.MovingGraspBootstrapEnv"


def arrival_return_ordering(config, gamma, remaining_steps, initial_potential=None):
    """Completed-return bounds from an arrival, with discount origin at arrival.

    For each fixed state the same initial potential cancels on success and failure.
    Across different arrivals it can vary across the complete potential budget;
    that broader bound is reported separately and is not asserted positive.
    """
    if type(remaining_steps) is not int or not 1 <= remaining_steps <= 500:
        raise ValueError("Remaining deadline must be an integer from 1 to 500")
    if not np.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise ValueError("Require a finite gamma in (0,1]")
    budget = sum((getattr(config, f"{key}_budget") for key in POTENTIAL_KEYS))
    if initial_potential is not None and (
        not np.isfinite(initial_potential)
        or not 0.0 <= initial_potential <= budget + 1e-10
    ):
        raise ValueError("Arrival potential exceeds its full potential budget")
    reward = GraspReward(success_bonus=config.success_bonus)
    horizon_sum = (
        remaining_steps
        if gamma == 1.0
        else (1.0 - gamma**remaining_steps) / (1.0 - gamma)
    )
    terminal_weight = gamma ** (remaining_steps - 1)
    success_base = (
        config.success_bonus * terminal_weight
        - 0.02
        * (reward.time_cost + reward.effort_cost + 4.0 * reward.force_change_cost)
        * horizon_sum
    )
    failure_base = (
        -0.02 * reward.time_cost * horizon_sum - reward.failure_cost * terminal_weight
    )
    result = dict(
        scope="Each fixed arrival state and remaining original deadline; not a comparison across different resets",
        remaining_steps=remaining_steps,
        initial_potential=initial_potential,
        initial_potential_upper_bound=budget,
        minimum_gap_same_arrival=success_base - failure_base,
        cross_arrival_success_lower_bound=success_base - budget,
        cross_arrival_failure_upper_bound=failure_base,
        cross_arrival_minimum_gap=success_base - budget - failure_base,
        proof="Terminal-zero shaping telescopes to minus the actual arrival potential; it cancels between outcomes from the same state",
    )
    if initial_potential is not None:
        result.update(
            success_lower_bound=success_base - initial_potential,
            failure_upper_bound=failure_base - initial_potential,
        )
    return result


def _validate_arrival(base, snapshot, expected_phase=None):
    _validate_snapshot(base, snapshot)
    phase = snapshot.get("phase")
    info = snapshot.get("info")
    if phase not in ("approach", "grasp") or (
        expected_phase is not None and phase != expected_phase
    ):
        raise ValueError(
            "Arrival must have the expected approach or grasp source phase"
        )
    if (
        snapshot.get("phase_success") is not True
        or not isinstance(info, dict)
        or info.get("phase_success") is not True
        or (info.get("phase") != phase)
        or (info.get("reason") != f"{phase}_success")
        or (info.get("step") != snapshot["step_count"])
    ):
        raise ValueError("Arrival requires consistent successful phase evidence")
    if type(snapshot["step_count"]) is not int or not 0 < snapshot["step_count"] < 500:
        raise ValueError(
            "Arrival must retain time within the original 500-action deadline"
        )
    state = snapshot["python_state"]
    if (
        state.get("observation_mode") != "pixels"
        or state.get("physics_steps") != base.physics_steps
        or (
            not np.isclose(
                snapshot["model"].opt.timestep,
                base.model.opt.timestep,
                rtol=0.0,
                atol=0.0,
            )
        )
    ):
        raise ValueError(
            "Arrival must retain original pixel observations and physical action frequency"
        )
    if not np.isclose(
        snapshot["data"].time,
        snapshot["step_count"] * base.control_dt,
        rtol=0.0,
        atol=1e-09,
    ):
        raise ValueError(
            "Arrival simulator time disagrees with its physical action clock"
        )
    preview = copy.copy(base)
    preview.data = snapshot["data"]
    preview.step_count = snapshot["step_count"]
    preview.last_action = state["last_action"]
    preview._previous_frame = state["_previous_frame"]
    _validate_observation(preview, snapshot["observation"])


def load_centered_arrival_pool(path, *, expected_sha256, expected_phase):
    """Hash-check, then deserialize a trusted locally produced gzip pickle pool.

    Only call this for trusted simulation outputs: pickle is executable Python.
    The required hash identifies the exact file before it is deserialized.
    """
    if expected_phase not in ("approach", "grasp"):
        raise ValueError("Expected pool phase must be approach or grasp")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any((char not in "0123456789abcdef" for char in expected_sha256))
    ):
        raise ValueError(
            "Expected SHA256 must contain 64 lowercase hexadecimal characters"
        )
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Arrival pool SHA256 verification failed")
    payload = pickle.loads(gzip.decompress(raw))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != POOL_VERSION
        or payload.get("phase") != expected_phase
        or (payload.get("source_phase") != expected_phase)
        or (
            payload.get("target_phase")
            != ("grasp" if expected_phase == "approach" else "lift")
        )
        or any(
            (
                payload.get(flag) is not True
                for flag in (
                    "trusted_pickle_only",
                    "no_action_targets",
                    "no_imitation_objective",
                )
            )
        )
    ):
        raise ValueError("Invalid trusted arrival pool protocol or source phase")
    snapshots, entries = (payload.get("snapshots"), payload.get("entries"))
    if (
        not isinstance(snapshots, (list, tuple))
        or not snapshots
        or (not isinstance(entries, (list, tuple)))
        or (len(entries) != len(snapshots))
    ):
        raise ValueError(
            "Arrival pool must contain matching nonempty snapshots and entries"
        )
    base = CenteredGraspEnv(render_images=False)
    try:
        for snapshot, entry in zip(snapshots, entries):
            _validate_arrival(base, snapshot, expected_phase)
            if (
                entry.get("source_phase") != expected_phase
                or entry.get("arrival_step") != snapshot["step_count"]
                or entry.get("remaining_actions") != 500 - snapshot["step_count"]
            ):
                raise ValueError(
                    "Arrival pool entry counters or source phase disagree with snapshot"
                )
    finally:
        base.close()
    provenance = dict(
        version=POOL_VERSION,
        sha256=expected_sha256,
        filename=path.name,
        source_phase=expected_phase,
        count=len(snapshots),
        source_provenance=payload.get("source_provenance"),
        identity=payload.get("identity"),
        protocol=payload.get("protocol"),
    )
    provenance = json.loads(json.dumps(provenance, allow_nan=False))
    return dict(
        provenance=provenance,
        snapshots=tuple(snapshots),
        entries=tuple(copy.deepcopy(entries)),
    )


def _centered_history(base, snapshot, info, metrics):
    """Reuse only a measured, matching counter suffix; unknown history is zero."""
    state = snapshot["python_state"]
    if (
        snapshot["environment_class"] == MOVING_GRASP_CLASS
        and base.reward_config == CenteredGraspConfig()
    ):
        count = state.get("_centered_full_pickup_steps")
        if (
            type(count) is not int
            or not 0 <= count < base.hold_steps
            or count > snapshot["step_count"]
            or (snapshot["info"].get("centered_full_pickup_stable_steps") != count)
        ):
            raise ValueError(
                "Invalid or already completed centered hold history in arrival"
            )
        stable_now = (
            metrics["centered_geometry"]
            and all(info["contacts"])
            and (info["clearance"] >= 0.06)
            and (info["upright"] >= math.cos(math.radians(20.0)))
            and (info["cup_speed"] < 0.1)
        )
        if count and (not stable_now):
            raise ValueError(
                "Saved centered hold history disagrees with current physical state"
            )
        return (
            count,
            "Verified matching centered counter suffix; earlier unobserved history is not credited",
        )
    return (
        0,
        "Unknown or differently defined centered history conservatively initialized to zero",
    )


def restore_centered_arrival(env, snapshot):
    """Restore a successful approach/grasp arrival into a fresh centered task.

    Returns the exact saved actor observation and new full-task diagnostic info.
    Applicable original full-hold history survives; new centered potential starts
    at the restored physical state, so no artificial reward jump is introduced.
    """
    base, wrappers = _unwrap(env)
    if not isinstance(base, CenteredGraspEnv):
        raise ValueError("Receiving task must be CenteredGraspEnv")
    _validate_arrival(base, snapshot)
    if (
        base.step_count != 0
        or not hasattr(base, "_centered_previous_potentials")
        or any((wrapper.needs_reset or wrapper.rewards for wrapper in wrappers))
    ):
        raise ValueError(
            "Reset the receiving centered environment before restoring an arrival"
        )
    state = snapshot["python_state"]
    if (
        base.observation_mode != state["observation_mode"]
        or base.render_images != state["render_images"]
    ):
        raise ValueError(
            "Arrival must retain the source observation and rendering mode"
        )
    _restore_physics(base, snapshot)
    for name in PHYSICAL_EPISODE_FIELDS:
        if name in state:
            setattr(base, name, copy.deepcopy(state[name]))
    base._original_success_ever = bool(state.get("_full_pickup_success_ever", False))
    info = base._generalization_info(base._info())
    metrics = base._center_metrics(info)
    base._centered_hold_steps, history_policy = _centered_history(
        base, snapshot, info, metrics
    )
    base._centered_reward_totals = {}
    base._centered_shaping_totals = {key: 0.0 for key in POTENTIAL_KEYS}
    base._centered_previous_potentials = base._potentials(metrics)
    base._centered_initial_potential = sum(base._centered_previous_potentials.values())
    certificate = arrival_return_ordering(
        base.reward_config,
        base.gamma,
        base.max_steps - base.step_count,
        base._centered_initial_potential,
    )
    base.return_ordering_bounds = certificate
    original_success = bool(snapshot["info"].get("original_is_success", False))
    base._original_success_ever |= original_success
    info = base._center_info(
        {
            **info,
            "is_success": False,
            "reason": "",
            "original_is_success": original_success,
        },
        metrics,
    )
    metadata = dict(
        arrival_source_phase=snapshot["phase"],
        arrival_source_step=base.step_count,
        arrival_remaining_steps=base.max_steps - base.step_count,
        arrival_clock_reset=False,
        arrival_velocity_reset=False,
        arrival_source_phase_success=True,
        arrival_original_hold_steps=base._hold_steps,
        arrival_centered_hold_steps=base._centered_hold_steps,
        arrival_centered_history_policy=history_policy,
        arrival_reward_certificate=certificate,
    )
    base._centered_arrival_metadata = metadata
    observation = copy.deepcopy(snapshot["observation"])
    _validate_observation(base, observation)
    return (observation, {**info, **copy.deepcopy(metadata)})


class CenteredArrivalGraspEnv(CenteredGraspEnv):
    """Uniformly sample verified successful arrivals, then train full pickup."""

    version = VERSION

    def __init__(self, *, arrival_pool, **kwargs):
        if (
            not isinstance(arrival_pool, dict)
            or not arrival_pool.get("snapshots")
            or len(arrival_pool.get("entries", ())) != len(arrival_pool["snapshots"])
            or (
                arrival_pool.get("provenance", {}).get("count")
                != len(arrival_pool["snapshots"])
            )
        ):
            raise ValueError(
                "Require a nonempty pool returned by load_centered_arrival_pool"
            )
        if kwargs.get("observation", "pixels") != "pixels":
            raise ValueError(
                "Arrival training retains pixel and 18-value proprioception observations"
            )
        self.arrival_pool = arrival_pool
        self._centered_arrival_metadata = {}
        super().__init__(**kwargs)
        for snapshot in arrival_pool["snapshots"]:
            _validate_arrival(
                self, snapshot, arrival_pool["provenance"]["source_phase"]
            )
            if snapshot["python_state"]["render_images"] != self.render_images:
                raise ValueError("Arrival pool and environment rendering modes differ")
        self.return_ordering_bounds = arrival_return_ordering(
            self.reward_config, self.gamma, 500
        )

    def specification(self):
        return {
            **super().specification(),
            "version": VERSION,
            "arrival_pool": copy.deepcopy(self.arrival_pool["provenance"]),
            "reset_distribution": "Uniform sampling of exact successful source phase snapshots",
            "arrival_clock_reset": False,
            "arrival_velocity_reset": False,
            "arrival_centered_history": "Preserve verified matching suffix, otherwise zero",
            "action_demonstrations": False,
            "runtime_controller": False,
        }

    def reset(self, *, seed=None, options=None):
        rendering = self.render_images
        self.render_images = False
        try:
            super().reset(seed=seed, options=options)
        finally:
            self.render_images = rendering
        index = int(self.rng.integers(len(self.arrival_pool["snapshots"])))
        observation, info = restore_centered_arrival(
            self, self.arrival_pool["snapshots"][index]
        )
        self._centered_arrival_metadata.update(
            arrival_pool_index=index,
            arrival_pool_sha256=self.arrival_pool["provenance"]["sha256"],
            arrival_pool_entry=copy.deepcopy(self.arrival_pool["entries"][index]),
        )
        return (observation, {**info, **copy.deepcopy(self._centered_arrival_metadata)})

    def step(self, action):
        observation, reward, done, truncated, info = super().step(action)
        return (
            observation,
            reward,
            done,
            truncated,
            {**info, **copy.deepcopy(self._centered_arrival_metadata)},
        )
