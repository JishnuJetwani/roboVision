"""Shared duration, initialization, and evaluation rules for the pickup manager."""

import math
import numpy as np
from .policy_loading import load_grasp_policy
from .skill_hard_evaluation import resolve_skill_checkpoint, sha256
from .training_checkpoint import _parameter_digest

OPTION_STEPS = 5
PHYSICAL_GAMMA = 0.999
CHUNK = 4096
GATE_INTERVAL = 16384
ROLLOUT_TRANSITIONS = 512


def manager_duration_settings(option_steps=OPTION_STEPS):
    if type(option_steps) is not int or option_steps not in (5, 25):
        raise ValueError("Registered manager option_steps must be5 or25")
    chunk = CHUNK if option_steps == 5 else ROLLOUT_TRANSITIONS
    gates = GATE_INTERVAL if option_steps == 5 else 3072
    return dict(
        option_steps=option_steps,
        physical_control_hz=50,
        option_seconds=0.02 * option_steps,
        manager_gamma=PHYSICAL_GAMMA**option_steps,
        checkpoint_chunk_manager_steps=chunk,
        checkpoint_chunk_physical_upper_bound=chunk * option_steps,
        gate_interval_manager_steps=gates,
        gate_interval_physical_upper_bound=gates * option_steps,
    )


def validate_manager_duration(manager, option_steps=OPTION_STEPS):
    settings = manager_duration_settings(option_steps)
    gamma = getattr(manager, "gamma", None)
    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (float, int, np.floating))
        or (
            not math.isclose(
                float(gamma), settings["manager_gamma"], rel_tol=0.0, abs_tol=1e-12
            )
        )
    ):
        raise ValueError(
            "Saved manager gamma does not match explicit option_steps; duration transfer is unsupported"
        )
    return settings


def _source(volume_root, spec, *, loader=load_grasp_policy, device="cuda", env=None):
    if not isinstance(spec, dict) or set(spec) != {"run", "checkpoint"}:
        raise ValueError("Each source must explicitly contain run and checkpoint")
    path, record, metadata, metadata_path = resolve_skill_checkpoint(
        volume_root, spec["run"], spec["checkpoint"]
    )
    model = loader(path, env=env, device=device)
    return (
        model,
        dict(
            **spec,
            policy_sha256=sha256(path),
            metadata_sha256=sha256(metadata_path),
            inherited_model_steps=int(model.num_timesteps),
        ),
        path,
    )


def initialize_manager_features(manager, approach, *, option_steps=OPTION_STEPS):
    """Copy learned sensory extractors only; categorical/value heads stay fresh."""
    duration = manager_duration_settings(option_steps)
    pairs = [
        (getattr(manager.policy, name), getattr(approach.policy, name))
        for name in ("pi_features_extractor", "vf_features_extractor")
    ]
    if pairs[0][0] is pairs[1][0]:
        raise ValueError(
            "Manager requires independent actor and critic feature extractors"
        )
    if manager.policy.optimizer.state:
        raise ValueError(
            "Feature initialization requires a new optimizer with no moments"
        )
    for target, source in pairs:
        target_state, source_state = (target.state_dict(), source.state_dict())
        if set(target_state) != set(source_state) or any(
            (
                target_state[key].shape != source_state[key].shape
                or target_state[key].dtype != source_state[key].dtype
                for key in target_state
            )
        ):
            raise ValueError(
                "Approach/manager feature extractor names, shapes and dtypes must match"
            )
    heads_before = _parameter_digest(
        {
            name: getattr(manager.policy, name).state_dict()
            for name in ("mlp_extractor", "action_net", "value_net")
        }
    )
    for target, source in pairs:
        target.load_state_dict(source.state_dict(), strict=True)
        target.requires_grad_(True)
        if _parameter_digest(target.state_dict()) != _parameter_digest(
            source.state_dict()
        ):
            raise RuntimeError("Feature copy was not exact")
        if {value.data_ptr() for value in target.parameters()} & {
            value.data_ptr() for value in source.parameters()
        }:
            raise RuntimeError("Feature initialization shares source storage")
    if heads_before != _parameter_digest(
        {
            name: getattr(manager.policy, name).state_dict()
            for name in ("mlp_extractor", "action_net", "value_net")
        }
    ):
        raise RuntimeError("Feature initialization changed a manager head")
    return dict(
        source_role="approach",
        exact_feature_copy=True,
        independent_trainable_storage=True,
        option_steps=option_steps,
        manager_gamma=duration["manager_gamma"],
        physical_control_hz=50,
        copied_modules=["pi_features_extractor", "vf_features_extractor"],
        feature_sha256=[_parameter_digest(target.state_dict()) for target, _ in pairs],
        categorical_mlp_and_value_heads="Fresh initialization; not copied",
        optimizer_moments="Fresh; no source optimizer state copied",
        imitation_objective=False,
    )


def posthoc_center_metrics(env, info):
    """Evaluation-only geometry; never changes observations, reward or control."""
    offset = np.asarray(env.grasp_position) - np.asarray(env.cup_position)
    xy = float(np.linalg.norm(offset[:2]))
    z = float(offset[2])
    axis = env.data.site_xmat[env._grasp_sid].reshape(3, 3)[:, 0]
    downward = float(-axis[2])
    tolerance = 1e-12
    geometry = bool(
        -tolerance <= z <= 0.015 + tolerance
        and xy <= 0.015 + tolerance
        and (downward >= math.cos(math.radians(15.0)) - tolerance)
    )
    bilateral = bool(all(info["contacts"]))
    stable = bool(
        geometry
        and bilateral
        and (info["clearance"] >= 0.06)
        and (info["upright"] >= math.cos(math.radians(20.0)))
        and (info["cup_speed"] < 0.1)
    )
    return dict(
        center_z_relative_m=z,
        center_xy_error_m=xy,
        center_downward=downward,
        centered_geometry=geometry,
        centered_bilateral_contact=geometry and bilateral,
        centered_stable_lift=stable,
    )


class PosthocCenteredAudit:
    """Track centered events only until the original environment terminates."""

    def __init__(self):
        self.consecutive = 0
        self.maximum_consecutive = 0
        self.first_contact = None
        self.first_full_lift = None
        self.peak_clearance = float("-inf")

    def observe(self, env, info, time_seconds):
        metrics = posthoc_center_metrics(env, info)
        self.peak_clearance = max(self.peak_clearance, float(info["clearance"]))
        if metrics["centered_bilateral_contact"] and self.first_contact is None:
            self.first_contact = float(time_seconds)
        self.consecutive = (
            self.consecutive + 1 if metrics["centered_stable_lift"] else 0
        )
        self.maximum_consecutive = max(self.maximum_consecutive, self.consecutive)
        physical_failure = info.get("reason", "") not in (
            "",
            "success",
            "timeout",
            "dropped",
        )
        if (
            self.consecutive >= 25
            and self.first_full_lift is None
            and (not physical_failure)
        ):
            self.first_full_lift = float(time_seconds)
        return dict(**metrics, consecutive_centered_lift_actions=self.consecutive)

    def report(self):
        return dict(
            centered_success=self.first_full_lift is not None,
            centered_full_lift_success_observed=self.first_full_lift is not None,
            first_centered_contact=self.first_contact,
            first_centered_full_lift=self.first_full_lift,
            maximum_consecutive_centered_lift_actions=self.maximum_consecutive,
            peak_clearance_m=self.peak_clearance,
            observation_window="Observed through original termination; episode is never extended for this metric",
            centered_contact_definition="Bilateral contact plus Z0–15mm, XY<=15mm and wrist-down tilt<=15deg",
            centered_full_lift_definition="25 consecutive centered bilateral actions with clearance>=60mm, cup upright<=20deg and speed<0.1m/s",
        )


def manager_evaluation_cases(*, profile="fixed", final=False, seed=330100):
    if profile == "fixed":
        return [dict(height=h, offset=[0.0, 0.0]) for h in (0.025, 0.065, 0.1, 0.14)]
    if type(seed) is not int or not 0 <= seed < 699999900:
        raise ValueError("Use evaluation seeds below the reserved range")
    if profile != "fresh40":
        raise ValueError("Unknown evaluation profile")
    rng = np.random.default_rng(seed)
    return [
        dict(height=float(h), offset=[0.0, 0.0]) for h in rng.uniform(0.025, 0.14, 40)
    ]
