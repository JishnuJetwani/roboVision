"""Arrival resets retain dynamics and only change the full-task objective."""

import copy
import gzip
import hashlib
import json
import pickle
import mujoco
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from robovision.centered_arrival_env import (
    CenteredArrivalGraspEnv,
    arrival_return_ordering,
    load_centered_arrival_pool,
    restore_centered_arrival,
)
from robovision.centered_grasp_bootstrap import MovingGraspBootstrapEnv
from robovision.centered_grasp_env import (
    CenteredGraspConfig,
    CenteredGraspEnv,
    potential_increment,
)
from robovision.hierarchical_skills import ApproachSkillEnv
from robovision.hierarchical_replay import write_arrival_pool as _write_pool
from robovision.joint_env import JointGraspEnv
from robovision.skill_state import capture_skill_state


def successful_arrival(phase, *, rendering=False, height=0.0):
    """Small physical fixtures use force inputs, with no mocked success gate."""
    if phase == "approach":
        env = ApproachSkillEnv(fixed_height=height, render_images=rendering)
        action = np.array([0.0, -0.04, -0.02, 0.0, 0.05], np.float32)
    else:
        env = MovingGraspBootstrapEnv(
            fixed_relative_z=0.0075, fixed_opening=0.029, render_images=rendering
        )
        action = np.array([0.0, -0.04, -0.02, 0.0, -0.2], np.float32)
    observation, info = env.reset(seed=12)
    for _ in range(10):
        observation, _, done, _, info = env.step(action)
        if done:
            break
    assert info["phase_success"] and info["reason"] == f"{phase}_success"
    return (env, capture_skill_state(env, observation, info=info))


def save_pool(tmp_path, phase, snapshots):
    path = tmp_path / f"{phase}-arrivals.pkl.gz"
    entries = [
        dict(
            snapshot=snapshot,
            source_phase=phase,
            target_phase="grasp" if phase == "approach" else "lift",
            arrival_step=snapshot["step_count"],
            remaining_actions=500 - snapshot["step_count"],
        )
        for snapshot in snapshots
    ]
    record = _write_pool(
        path,
        phase,
        entries,
        identity="test-source-identity",
        sources={"test": "locally generated force fixture"},
        protocol={"test": True},
    )
    return (path, record["sha256"])


def assert_physics_equal(left, right):
    for name in (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "act",
        "qfrc_applied",
        "xfrc_applied",
        "mocap_pos",
        "mocap_quat",
        "userdata",
        "qfrc_constraint",
    ):
        np.testing.assert_array_equal(
            getattr(left.data, name), getattr(right.data, name)
        )
    assert left.data.time == right.data.time


@pytest.mark.parametrize("phase", ["approach", "grasp"])
def test_arrival_restores_exact_images_dynamic_state_and_next_force_action(
    phase, monkeypatch
):
    monkeypatch.setattr(
        JointGraspEnv,
        "render_camera",
        lambda self: np.full((96, 96, 3), self.step_count, np.uint8),
    )
    source, snapshot = successful_arrival(phase, rendering=True)
    target = Monitor(CenteredGraspEnv(gamma=0.999, render_images=True))
    try:
        assert np.linalg.norm(source.data.qvel) > 0.0
        assert np.linalg.norm(source.data.qacc_warmstart) > 0.0
        target.reset(seed=80)
        rng = copy.deepcopy(target.unwrapped.get_rng_state())
        observation, info = restore_centered_arrival(target, snapshot)
        assert_physics_equal(source, target.unwrapped)
        assert target.unwrapped.get_rng_state() == rng
        assert set(observation) == {"image", "proprio"}
        assert observation["proprio"].shape == (18,)
        assert observation["image"].shape == (6, 96, 96)
        for key in observation:
            np.testing.assert_array_equal(
                observation[key], snapshot["observation"][key]
            )
        assert observation["image"][0, 0, 0] != observation["image"][3, 0, 0]
        assert info["arrival_remaining_steps"] == 500 - snapshot["step_count"]
        assert not info["arrival_clock_reset"] and (not info["arrival_velocity_reset"])
        assert not info["centered_success"] and "phase_success" not in info
        assert target.rewards == [] and (not target.needs_reset)
        action = np.array([0.001, -0.039, -0.021, 0.001, -0.18], np.float32)
        expected = JointGraspEnv.step(source, action)
        actual = target.step(action)
        assert_physics_equal(source, target.unwrapped)
        for key in actual[0]:
            np.testing.assert_array_equal(actual[0][key], expected[0][key])
        assert target.unwrapped._hold_steps == source._hold_steps
    finally:
        source.close()
        target.close()


def test_grasp_arrival_uses_actual_nonopen_potential_and_next_increment():
    source, snapshot = successful_arrival("grasp")
    target = CenteredGraspEnv(render_images=False, gamma=0.995)
    try:
        target.reset(seed=3)
        _, info = restore_centered_arrival(target, snapshot)
        initial = info["centered_initial_potential"]
        assert 10.0 < initial <= 40.0
        assert target._centered_reward_totals == {}
        assert target._centered_shaping_totals == dict(
            approach=0.0, grasp=0.0, lift=0.0
        )
        metrics = target._center_metrics(target._info())
        assert target._centered_previous_potentials == target._potentials(metrics)
        previous = target._centered_previous_potentials.copy()
        _, _, done, _, stepped = target.step(np.array([0.0, -0.04, -0.02, 0.0, -0.2]))
        for key in previous:
            expected = potential_increment(
                previous[key],
                stepped["centered_current_potentials"][key],
                target.gamma,
                done,
            )
            assert stepped["reward_components"][
                f"centered_{key}_potential"
            ] == pytest.approx(expected)
        assert info["arrival_reward_certificate"]["initial_potential"] == initial
        assert info["arrival_reward_certificate"]["minimum_gap_same_arrival"] > 0.0
        assert "open-hand" not in json.dumps(
            target.specification()["return_ordering_bounds"]
        )
    finally:
        source.close()
        target.close()


@pytest.mark.parametrize("gamma", [0.995, 0.999, 1.0])
def test_reward_bounds_use_actual_arrival_and_do_not_assert_cross_reset_order(gamma):
    config = CenteredGraspConfig()
    for remaining in (1, 70, 499, 500):
        first = arrival_return_ordering(config, gamma, remaining, 3.0)
        second = arrival_return_ordering(config, gamma, remaining, 39.0)
        assert second["success_lower_bound"] == pytest.approx(
            first["success_lower_bound"] - 36.0
        )
        assert second["failure_upper_bound"] == pytest.approx(
            first["failure_upper_bound"] - 36.0
        )
        assert (
            second["minimum_gap_same_arrival"]
            == first["minimum_gap_same_arrival"]
            > 0.0
        )
        assert second["initial_potential_upper_bound"] == 40.0
    if gamma == 0.995:
        assert second["cross_arrival_minimum_gap"] < 0.0


def test_pool_hash_protocol_counters_and_success_evidence_are_validated(tmp_path):
    source, snapshot = successful_arrival("approach")
    try:
        path, digest = save_pool(tmp_path, "approach", [snapshot])
        pool = load_centered_arrival_pool(
            path, expected_sha256=digest, expected_phase="approach"
        )
        assert pool["provenance"]["count"] == 1
        assert pool["provenance"]["sha256"] == digest
        assert json.loads(json.dumps(pool["provenance"])) == pool["provenance"]
        with pytest.raises(ValueError, match="SHA256 verification"):
            load_centered_arrival_pool(
                path, expected_sha256="0" * 64, expected_phase="approach"
            )
        with pytest.raises(ValueError, match="protocol or source phase"):
            load_centered_arrival_pool(
                path, expected_sha256=digest, expected_phase="grasp"
            )
        original = gzip.decompress(path.read_bytes())
        for mutation, error in (
            (lambda payload: payload["entries"][0].update(arrival_step=77), "counters"),
            (
                lambda payload: payload["snapshots"][0].update(phase_success=False),
                "successful phase evidence",
            ),
            (
                lambda payload: payload["snapshots"][0]["info"].update(
                    phase_success=False
                ),
                "successful phase evidence",
            ),
            (
                lambda payload: payload["snapshots"][0]["observation"]["proprio"].fill(
                    0.0
                ),
                "stale",
            ),
            (lambda payload: payload.update(snapshots=[], entries=[]), "nonempty"),
        ):
            payload = pickle.loads(original)
            mutation(payload)
            changed = gzip.compress(pickle.dumps(payload))
            path.write_bytes(changed)
            with pytest.raises(ValueError, match=error):
                load_centered_arrival_pool(
                    path,
                    expected_sha256=hashlib.sha256(changed).hexdigest(),
                    expected_phase="approach",
                )
    finally:
        source.close()


def test_uniform_pool_reset_is_deterministic_reusable_and_reports_metadata(tmp_path):
    source, first = successful_arrival("approach")
    other, second = successful_arrival("approach", height=0.001)
    try:
        assert second["phase_success"]
        path, digest = save_pool(tmp_path, "approach", [first, second])
        pool = load_centered_arrival_pool(
            path, expected_sha256=digest, expected_phase="approach"
        )
        a = CenteredArrivalGraspEnv(arrival_pool=pool, render_images=False, gamma=0.999)
        b = CenteredArrivalGraspEnv(arrival_pool=pool, render_images=False, gamma=0.999)
        try:
            counts = set()
            for seed in range(20):
                left, li = a.reset(seed=seed)
                right, ri = b.reset(seed=seed)
                assert li["arrival_pool_index"] == ri["arrival_pool_index"]
                counts.add(li["arrival_pool_index"])
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
                assert a.step_count == li["arrival_source_step"]
                assert li["arrival_pool_sha256"] == digest
                assert li["arrival_centered_hold_steps"] == 0
                assert "Unknown" in li["arrival_centered_history_policy"]
            assert counts == {0, 1}
            _, _, _, _, stepped = a.step(np.array([0.0, -0.04, -0.02, 0.0, 0.05]))
            assert stepped["arrival_pool_index"] == li["arrival_pool_index"]
            assert a.specification()["arrival_pool"] == pool["provenance"]
            assert a.specification()["original_success_terminates"] is False
            assert a.action_space.shape == (5,)
        finally:
            a.close()
            b.close()
    finally:
        source.close()
        other.close()


def test_original_shallow_hold_does_not_end_arrival_task_and_clock_does(monkeypatch):
    source, snapshot = successful_arrival("grasp")
    target = CenteredGraspEnv(render_images=False)
    try:
        target.reset(seed=2)
        restore_centered_arrival(target, snapshot)
        target._hold_steps = 24
        target.data.qpos[target._cup_qadr : target._cup_qadr + 3] = [0.32, 0.0, 0.381]
        target.data.qpos[:4] = target.inverse_kinematics([0.32, 0.0, 0.441])
        mujoco.mj_forward(target.model, target.data)
        measured = target._info()
        monkeypatch.setattr(
            target,
            "_info",
            lambda: {
                **measured,
                "contacts": [True, True],
                "clearance": 0.08,
                "upright": 1.0,
                "cup_speed": 0.01,
                "step": target.step_count,
            },
        )
        monkeypatch.setattr(mujoco, "mj_step", lambda *_args, **_kwargs: None)
        _, _, done, _, info = target.step(np.zeros(5))
        assert (
            info["original_is_success"]
            and (not done)
            and (not info["centered_success"])
        )
        assert info["reward_components"]["success"] == 0.0
        target.step_count = 499
        _, _, done, truncated, info = target.step(np.zeros(5))
        assert done and (not truncated) and (info["reason"] == "timeout")
        assert info["centered_effective_next_potentials"] == dict(
            approach=0.0, grasp=0.0, lift=0.0
        )
    finally:
        source.close()
        target.close()


def test_history_reuses_only_matching_verified_suffix_and_preserves_original_counter(
    monkeypatch,
):
    source, _ = successful_arrival("grasp")
    target = CenteredGraspEnv(render_images=False)
    try:
        source.data.qpos[source._cup_qadr + 2] += 0.08
        source.data.qpos[:4] = source.inverse_kinematics(
            source.grasp_position + [0.0, 0.0, 0.08]
        )
        source.data.qvel[:] = 0.0
        mujoco.mj_forward(source.model, source.data)
        measured = source._info()
        assert all(measured["contacts"]) and measured["clearance"] >= 0.06
        source._hold_steps = 4
        source._centered_full_pickup_steps = 4
        observation = source._observation()
        metrics = source._phase_metrics(measured)
        info = source._skill_info(
            {**measured, "reason": "grasp_success", "original_is_success": False},
            metrics,
            success=True,
        )
        info = source._centered_full_pickup_info(info)
        snapshot = capture_skill_state(source, observation, info=info)
        target.reset(seed=1)
        _, restored = restore_centered_arrival(target, snapshot)
        assert target._hold_steps == target._centered_hold_steps == 4
        assert "Verified matching" in restored["arrival_centered_history_policy"]
        assert restored["centered_initial_potential"] > 30.0
        broken = copy.deepcopy(snapshot)
        broken["info"]["centered_full_pickup_stable_steps"] = 5
        target.reset(seed=1)
        with pytest.raises(ValueError, match="centered hold history"):
            restore_centered_arrival(target, broken)
    finally:
        source.close()
        target.close()


def test_restore_rejects_active_target_modified_physics_and_inconsistent_clock():
    source, snapshot = successful_arrival("approach")
    target = CenteredGraspEnv(render_images=False)
    try:
        target.reset(seed=1)
        target.step(np.zeros(5))
        with pytest.raises(ValueError, match="Reset the receiving"):
            restore_centered_arrival(target, snapshot)
        target.reset(seed=1)
        changed = copy.deepcopy(snapshot)
        changed["data"].qvel[0] += 0.01
        with pytest.raises(ValueError, match="modified after capture"):
            restore_centered_arrival(target, changed)
        changed = copy.deepcopy(snapshot)
        changed["step_count"] = changed["python_state"]["step_count"] = changed["info"][
            "step"
        ] = 499
        with pytest.raises(ValueError, match="simulator time"):
            restore_centered_arrival(target, changed)
    finally:
        source.close()
        target.close()
