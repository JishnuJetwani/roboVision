"""New millimeter schema preserves old checkpoint indices and physical tasks."""

from contextlib import nullcontext
from dataclasses import asdict
import numpy as np
import pytest
from robovision.near_pickup import (
    NearPickupEnv,
    FineNearPickupEnv,
    MillimeterNearPickupEnv,
    HEIGHTS,
    FINE_HEIGHTS,
    MILLIMETER_HEIGHTS,
)
import robovision.train_skill_curriculum as trainer

VARIANT = "millimeter_near_pickup"


def test_schedule_specification_exact_mapping_and_baseline_next_stage():
    assert len(HEIGHTS) == 51 and len(FINE_HEIGHTS) == 79
    assert FINE_HEIGHTS[70:73] == (0.08, 0.085, 0.09)
    assert MILLIMETER_HEIGHTS == tuple((mm / 1000.0 for mm in range(10, 141)))
    spec = MillimeterNearPickupEnv.specification()
    assert spec["version"] == "millimeter-near-pickup-height-v1"
    assert spec["final_lesson"] == 130 and spec["heights"] == list(MILLIMETER_HEIGHTS)
    assert spec["success"] == "Original full pickup and stable hold"
    cls, lessons, last, first = trainer.curriculum_definition(VARIANT)
    assert (
        cls is MillimeterNearPickupEnv
        and len(lessons) == 131
        and ((last, first) == (-1, 0))
    )
    stage, mapping = trainer.resolve_source_lesson(
        VARIANT, {"variant": "fine_near_pickup"}, {"lesson": 70}
    )
    assert stage == 70 and mapping["method"] == "exact_dataclass_equality"
    assert mapping["matched_lesson"] == asdict(FineNearPickupEnv.lessons[70])
    stage, stop = trainer.gate_transition(stage, 130, True, baseline=True)
    assert stage == 71 and (not stop) and (MILLIMETER_HEIGHTS[stage] == 0.081)
    assert trainer.gate_configurations(VARIANT, stage) == [(0.081, 71, "pickup")]
    assert (
        trainer.resolve_source_lesson(
            VARIANT, {"variant": "fine_near_pickup"}, {"lesson": 71}
        )[0]
        == 75
    )
    with pytest.raises(ValueError, match="conflicts"):
        trainer.resolve_source_lesson(
            VARIANT, {"variant": "fine_near_pickup"}, {"lesson": 70}, 71
        )
    with pytest.raises(ValueError, match="No exact"):
        trainer.resolve_source_lesson(
            "fine_near_pickup", {"variant": VARIANT}, {"lesson": 71}
        )


@pytest.mark.parametrize("height", [0.01, 0.055, 0.08, 0.085, 0.1, 0.14])
def test_matched_heights_exact_physics_observation_rewards_and_actions(height):
    old = FineNearPickupEnv(
        subtask_stage=FINE_HEIGHTS.index(height),
        observation="state",
        render_images=False,
    )
    new = MillimeterNearPickupEnv(
        subtask_stage=MILLIMETER_HEIGHTS.index(height),
        observation="state",
        render_images=False,
    )
    try:
        a, _ = old.reset(seed=15)
        b, _ = new.reset(seed=15)
        assert old.max_steps == new.max_steps == 500
        assert old.hold_steps == new.hold_steps == 25
        assert new.frontier_attempt_limit is None
        assert (
            old.action_space == new.action_space
            and old.observation_space == new.observation_space
        )
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for action in np.random.default_rng(1).uniform(-0.01, 0.01, (5, 5)):
            a = old.step(action)
            b = new.step(action)
            assert a[1:4] == b[1:4]
            assert a[4]["is_success"] == b[4]["is_success"]
            assert a[4]["reward_components"] == b[4]["reward_components"]
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            for name in ("qpos", "qvel", "ctrl"):
                np.testing.assert_array_equal(
                    getattr(old.data, name), getattr(new.data, name)
                )
    finally:
        old.close()
        new.close()


def test_frozen_gate_uses_original_deadline_not_training_limit(monkeypatch):
    captured = []

    def rows(model, factory, seeds, deterministic):
        env = factory()
        try:
            captured.append(
                (type(env), getattr(env, "frontier_attempt_limit", None), env.max_steps)
            )
        finally:
            env.close()
        return [dict(success=True, deterministic=deterministic) for _ in seeds]

    monkeypatch.setattr(trainer, "rollouts", rows)
    monkeypatch.setattr(trainer.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(trainer.torch.random, "fork_rng", lambda **kw: nullcontext())
    gate = trainer.gate(None, VARIANT, 71)
    assert gate["passed"]
    assert captured[:2] == [(MillimeterNearPickupEnv, None, 500)] * 2
    assert all((horizon == 500 for _, _, horizon in captured))
