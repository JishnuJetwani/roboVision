import numpy as np
import pytest
from robovision.near_pickup import NearPickupEnv, FineNearPickupEnv, FINE_HEIGHTS
from robovision.train_skill_curriculum import (
    resolve_source_lesson,
    gate_configurations,
    make_env,
)


def test_matching_source_height_physics_rewards_gate_and_observation_parity():
    for height in [0.05, 0.055, 0.08, 0.14]:
        old_stage = NearPickupEnv.heights.index(height)
        new_stage, _ = resolve_source_lesson(
            "fine_near_pickup", {"variant": "near_pickup"}, {"lesson": old_stage}
        )
        assert [
            (h, label) for h, _, label in gate_configurations("near_pickup", old_stage)
        ] == [
            (h, label)
            for h, _, label in gate_configurations("fine_near_pickup", new_stage)
        ]
        kw = dict(render_images=False, observation="state")
        a = NearPickupEnv(subtask_stage=old_stage, **kw)
        b = FineNearPickupEnv(subtask_stage=new_stage, **kw)
        try:
            oa, _ = a.reset(seed=1)
            ob, _ = b.reset(seed=1)
            for k in oa:
                np.testing.assert_array_equal(oa[k], ob[k])
            for action in np.random.default_rng(1).uniform(-0.02, 0.02, (5, 5)):
                ar = a.step(action)
                br = b.step(action)
                assert (
                    ar[1:4] == br[1:4]
                    and ar[4]["pickup_success"] == br[4]["pickup_success"]
                )
                np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
                np.testing.assert_array_equal(a.data.ctrl, b.data.ctrl)
        finally:
            a.close()
            b.close()
    spec = FineNearPickupEnv.specification()
    assert (
        spec["version"] == "fine-near-pickup-height-v1" and spec["final_lesson"] == 78
    )
    assert spec["heights"] == list(FINE_HEIGHTS)


def test_ambiguous_exact_mapping_rejected(monkeypatch):
    lessons = FineNearPickupEnv.lessons
    monkeypatch.setattr(FineNearPickupEnv, "lessons", lessons + (lessons[40],))
    with pytest.raises(ValueError, match="No exact"):
        resolve_source_lesson(
            "fine_near_pickup", {"variant": "near_pickup"}, {"lesson": 40}
        )
