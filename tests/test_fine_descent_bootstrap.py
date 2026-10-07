import numpy as np
import pytest
from robovision.hover_bootstrap import FineHoverBootstrapEnv, FineDescentBootstrapEnv
from robovision.train_skill_curriculum import (
    resolve_source_lesson,
    curriculum_definition,
    gate_configurations,
    resolve_frontier_fixed_height,
)


def test_exact_ten_millimeter_mapping_then_nine_millimeter_lesson():
    source = {"variant": "fine_hover_bootstrap"}
    stage, mapping = resolve_source_lesson(
        "fine_descent_bootstrap", source, {"lesson": 34}
    )
    assert stage == 42
    assert FineDescentBootstrapEnv.lessons[stage].target_height == 0.01
    assert FineDescentBootstrapEnv.lessons[stage + 1].target_height == 0.009
    assert mapping["method"] == "exact_dataclass_equality"
    with pytest.raises(ValueError, match="No exact"):
        resolve_source_lesson(
            "fine_hover_bootstrap",
            {"variant": "fine_descent_bootstrap"},
            {"lesson": 43},
        )
    with pytest.raises(ValueError, match="conflicts"):
        resolve_source_lesson("fine_descent_bootstrap", source, {"lesson": 34}, 34)


def test_schedule_prefix_and_pickup_tail_identical():
    old = FineHoverBootstrapEnv.lessons
    new = FineDescentBootstrapEnv.lessons
    assert new[:33] == old[:33]
    assert [l.target_height for l in new[33:53]] == [
        mm / 1000.0 for mm in range(19, -1, -1)
    ]
    assert new[53:] == old[37:]
    _, lessons, approach, pickup = curriculum_definition("fine_descent_bootstrap")
    assert len(lessons) == 58 and approach == 52 and (pickup == 56)
    assert resolve_frontier_fixed_height(0.08, {}, "fine_descent_bootstrap") == 0.08
    spec = FineDescentBootstrapEnv.specification()
    assert spec["first_descent_lesson"] == 22 and spec["full_approach_lesson"] == 52


def test_every_existing_fine_lesson_maps_and_physics_gate_parity():
    for old_stage, lesson in enumerate(FineHoverBootstrapEnv.lessons):
        new_stage, _ = resolve_source_lesson(
            "fine_descent_bootstrap",
            {"variant": "fine_hover_bootstrap"},
            {"lesson": old_stage},
        )
        assert FineDescentBootstrapEnv.lessons[new_stage] == lesson
    for old_stage in [22, 34, 41]:
        new_stage, _ = resolve_source_lesson(
            "fine_descent_bootstrap",
            {"variant": "fine_hover_bootstrap"},
            {"lesson": old_stage},
        )
        old_gate = gate_configurations("fine_hover_bootstrap", old_stage)
        new_gate = gate_configurations("fine_descent_bootstrap", new_stage)
        assert [(h, label) for h, _, label in old_gate] == [
            (h, label) for h, _, label in new_gate
        ]
        kw = dict(
            strict_descent=True,
            replay_fraction=0,
            render_images=False,
            observation="state",
            fixed_height=0.075,
        )
        a = FineHoverBootstrapEnv(subtask_stage=old_stage, **kw)
        b = FineDescentBootstrapEnv(subtask_stage=new_stage, **kw)
        try:
            oa, _ = a.reset(seed=6)
            ob, _ = b.reset(seed=6)
            for k in oa:
                np.testing.assert_array_equal(oa[k], ob[k])
            assert a._metrics() == b._metrics()
            for _ in range(4):
                ar = a.step(np.zeros(5))
                br = b.step(np.zeros(5))
                assert ar[1:4] == br[1:4]
                np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
        finally:
            a.close()
            b.close()
