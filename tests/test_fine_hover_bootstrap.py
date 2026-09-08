import numpy as np
import pytest
from robovision.hover_bootstrap import HoverBootstrapEnv, FineHoverBootstrapEnv
from robovision.train_skill_curriculum import (
    resolve_source_lesson,
    resolve_frontier_fixed_height,
    curriculum_definition,
)


def test_exact_cross_variant_mapping_and_next_baseline_lesson():
    stage, metadata = resolve_source_lesson(
        "fine_hover_bootstrap", {"variant": "hover_bootstrap"}, {"lesson": 2}
    )
    assert stage == 4 and FineHoverBootstrapEnv.lessons[stage].stable_steps == 8
    assert FineHoverBootstrapEnv.lessons[stage + 1].stable_steps == 9
    assert metadata["source_lesson"] == 2 and metadata["destination_lesson"] == 4
    assert metadata["method"] == "exact_dataclass_equality"
    stage, _ = resolve_source_lesson(
        "fine_hover_bootstrap", {"variant": "hover_bootstrap"}, {"lesson": 3}
    )
    assert stage == 8 and FineHoverBootstrapEnv.lessons[stage].stable_steps == 12
    with pytest.raises(ValueError, match="No exact"):
        resolve_source_lesson(
            "hover_bootstrap", {"variant": "fine_hover_bootstrap"}, {"lesson": 5}
        )
    with pytest.raises(ValueError, match="conflicts"):
        resolve_source_lesson(
            "fine_hover_bootstrap", {"variant": "hover_bootstrap"}, {"lesson": 2}, 2
        )
    assert resolve_source_lesson(
        "hover_bootstrap", {"variant": "hover_bootstrap"}, {"lesson": 3}
    ) == (3, None)
    assert resolve_source_lesson(
        "fine_hover_bootstrap", {"variant": "subtask"}, {"lesson": 3}
    ) == (0, None)


def test_every_old_lesson_maps_exactly_and_descent_final_identical():
    for i, lesson in enumerate(HoverBootstrapEnv.lessons):
        j, _ = resolve_source_lesson(
            "fine_hover_bootstrap", {"variant": "hover_bootstrap"}, {"lesson": i}
        )
        assert FineHoverBootstrapEnv.lessons[j] == lesson
        reverse, _ = resolve_source_lesson(
            "hover_bootstrap", {"variant": "fine_hover_bootstrap"}, {"lesson": j}
        )
        assert reverse == i
    assert FineHoverBootstrapEnv.lessons[22:] == HoverBootstrapEnv.lessons[6:]
    cls, _, approach, pickup = curriculum_definition("fine_hover_bootstrap")
    assert cls is FineHoverBootstrapEnv and approach == 36 and (pickup == 40)
    assert resolve_frontier_fixed_height(0.08, {}, "fine_hover_bootstrap") == 0.08


def test_fine_strict_and_reset_indices():
    env = FineHoverBootstrapEnv(
        subtask_stage=21,
        strict_descent=True,
        replay_fraction=0,
        observation="state",
        render_images=False,
        fixed_height=0.08,
    )
    try:
        env.reset(seed=3)
        assert env._metrics()[1]
        env.set_subtask_stage(22)
        env.reset(seed=3)
        assert not env._metrics()[1]
        env.fixed_height = None
        for stage in [22, 23]:
            env.set_subtask_stage(stage)
            _, info = env.reset(seed=4)
            assert info["approach_height"] == 0.075
        spec = env.specification()
        assert spec["first_descent_lesson"] == 22 and spec["final_lesson"] == 41
    finally:
        env.close()


def test_matching_descent_and_final_have_identical_observation_physics_reward():
    for old_stage in [6, 25]:
        new_stage, _ = resolve_source_lesson(
            "fine_hover_bootstrap",
            {"variant": "hover_bootstrap"},
            {"lesson": old_stage},
        )
        kwargs = dict(
            strict_descent=True,
            replay_fraction=0,
            render_images=False,
            observation="state",
            fixed_height=0.075,
        )
        a = HoverBootstrapEnv(subtask_stage=old_stage, **kwargs)
        b = FineHoverBootstrapEnv(subtask_stage=new_stage, **kwargs)
        try:
            oa, _ = a.reset(seed=6)
            ob, _ = b.reset(seed=6)
            for k in oa:
                np.testing.assert_array_equal(oa[k], ob[k])
            for _ in range(3):
                ar = a.step(np.zeros(5))
                br = b.step(np.zeros(5))
                assert ar[1:4] == br[1:4]
                np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        finally:
            a.close()
            b.close()
