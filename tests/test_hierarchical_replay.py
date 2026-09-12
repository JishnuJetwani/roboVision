import gzip
import pickle
import numpy as np
import pytest
from robovision.hierarchical_replay import (
    ArrivalResetEnv,
    load_arrival_pool,
    POOL_VERSION,
)
from robovision.hierarchical_skills import ApproachSkillEnv, GraspSkillEnv
from robovision.skill_hard_evaluation import sha256
from robovision.skill_state import capture_skill_state, restore_skill_handoff


@pytest.fixture
def arrival_pool(tmp_path):
    env = ApproachSkillEnv(fixed_height=0.005, render_images=False)
    try:
        obs, info = env.reset(seed=7)
        for _ in range(3):
            obs, _, _, _, info = env.step(np.zeros(5, np.float32))
        info["phase_success"] = True
        snapshot = capture_skill_state(env, obs, info=info)
        path = tmp_path / "arrivals.pkl.gz"
        with gzip.open(path, "wb") as stream:
            pickle.dump(
                dict(
                    version=POOL_VERSION,
                    phase="approach",
                    source_provenance={},
                    snapshots=[snapshot],
                    entries=[dict(case="test", arrival_step=3)],
                ),
                stream,
            )
        return (path, sha256(path), snapshot)
    finally:
        env.close()


def test_arrival_reset_preserves_complete_physical_continuation(arrival_pool):
    path, digest, snapshot = arrival_pool
    replay = ArrivalResetEnv(
        GraspSkillEnv(render_images=False),
        pool_path=path,
        pool_sha256=digest,
        probability=1.0,
    )
    reference = GraspSkillEnv(render_images=False)
    try:
        obs, info = replay.reset(seed=90)
        reference.reset(seed=90)
        expected, _ = restore_skill_handoff(reference, snapshot)
        assert info["actual_arrival_reset"] and info["handoff_clock_reset"] is False
        assert replay.unwrapped.step_count == snapshot["step_count"]
        for key in obs:
            np.testing.assert_array_equal(obs[key], expected[key])
        action = np.array([0.01, -0.04, 0.02, 0.01, -0.2], np.float32)
        actual = replay.step(action)
        wanted = reference.step(action)
        for key in actual[0]:
            np.testing.assert_array_equal(actual[0][key], wanted[0][key])
        assert actual[1:4] == wanted[1:4]
        np.testing.assert_array_equal(replay.unwrapped.data.ctrl, reference.data.ctrl)
        assert actual[4]["arrival_entry"]["case"] == "test"
    finally:
        replay.close()
        reference.close()


def test_hash_rejection_happens_before_deserializing_trusted_artifact(
    arrival_pool, monkeypatch
):
    path, _, _ = arrival_pool
    monkeypatch.setattr(
        pickle, "load", lambda _: pytest.fail("Corrupt file was deserialized")
    )
    with pytest.raises(ValueError, match="bytes differ"):
        load_arrival_pool(path, "0" * 64, "grasp")


def test_pool_phase_cannot_skip_to_lift(arrival_pool):
    path, digest, _ = arrival_pool
    with pytest.raises(ValueError, match="phase does not match"):
        load_arrival_pool(path, digest, "lift")
