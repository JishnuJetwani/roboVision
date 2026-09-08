import pytest
from robovision.train_skill_curriculum import (
    make_env,
    resolve_frontier_attempt_limit,
    validate_interaction_schedule,
)


def test_frontier_deadline_preserves_retention_and_unwrapped_evaluation():
    from robovision.near_pickup import FineNearPickupEnv

    for worker in range(4):
        env = make_env("fine_near_pickup", 69, worker, frontier_attempt_limit_steps=150)
        try:
            assert getattr(env.unwrapped, "frontier_attempt_limit", None) == (
                150 if worker < 2 else None
            )
            assert env.unwrapped.max_steps == 500
        finally:
            env.close()
    env = FineNearPickupEnv(subtask_stage=69, render_images=False, observation="state")
    try:
        assert env.frontier_attempt_limit is None
    finally:
        env.close()


def test_deadline_inherits_but_can_explicitly_disable_for_subtask():
    assert (
        resolve_frontier_attempt_limit(
            None, {"frontier_attempt_limit_steps": 150}, "fine_near_pickup"
        )
        == 150
    )
    assert (
        resolve_frontier_attempt_limit(
            0, {"frontier_attempt_limit_steps": 150}, "subtask"
        )
        == 0
    )
    with pytest.raises(ValueError):
        resolve_frontier_attempt_limit(
            None, {"frontier_attempt_limit_steps": 150}, "subtask"
        )
    for bad in [True, -1, 501, 150.5]:
        with pytest.raises(ValueError):
            resolve_frontier_attempt_limit(bad, {}, "fine_near_pickup")


def test_interaction_protocol_rejects_partial_rollout_chunks():
    validate_interaction_schedule(102400, 20480)
    validate_interaction_schedule(0, 0)
    for args in [(102401, 20480), (102400, 300), (0, 20480), (-2048, 0), (True, 0)]:
        with pytest.raises(ValueError):
            validate_interaction_schedule(*args)
