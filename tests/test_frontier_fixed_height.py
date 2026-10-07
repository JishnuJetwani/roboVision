import numpy as np
import pytest
from robovision.train_skill_curriculum import make_env, resolve_frontier_fixed_height


def test_default_inherit_disable_and_validation():
    for value in [None, "inherit"]:
        assert resolve_frontier_fixed_height(value, {}, "hover_bootstrap") is None
        assert (
            resolve_frontier_fixed_height(
                value, {"frontier_fixed_height": 0.08}, "hover_bootstrap"
            )
            == 0.08
        )
    assert (
        resolve_frontier_fixed_height(
            "off", {"frontier_fixed_height": 0.08}, "hover_bootstrap"
        )
        is None
    )
    assert (
        resolve_frontier_fixed_height(
            ".075", {"frontier_fixed_height": 0.08}, "open_bootstrap"
        )
        == 0.075
    )
    for value in [-0.01, 0.141, float("nan"), float("inf"), True, "nonsense"]:
        with pytest.raises(ValueError):
            resolve_frontier_fixed_height(value, {}, "hover_bootstrap")
    for variant in ["dense", "subtask"]:
        with pytest.raises(ValueError):
            resolve_frontier_fixed_height(0.08, {}, variant)


def test_only_worker0_changes_actual_reset_others_exactly_unchanged():
    for worker in range(4):
        a = make_env(
            "hover_bootstrap",
            2,
            worker,
            bootstrap_height_jitter=0.005,
            frontier_fixed_height=0.08,
        )
        b = make_env("hover_bootstrap", 2, worker, bootstrap_height_jitter=0.005)
        a.unwrapped.render_images = b.unwrapped.render_images = False
        try:
            random_heights = []
            for seed in range(5):
                oa, ia = a.reset(seed=seed)
                ob, ib = b.reset(seed=seed)
                if worker == 0:
                    assert ia["approach_height"] == 0.08
                    assert a.unwrapped.grasp_position[2] - a.unwrapped.cup_position[
                        2
                    ] - 0.014 == pytest.approx(0.08)
                    random_heights.append(ib["approach_height"])
                else:
                    assert ia == ib
                    np.testing.assert_array_equal(
                        a.unwrapped.data.qpos, b.unwrapped.data.qpos
                    )
                    for k in oa:
                        np.testing.assert_array_equal(oa[k], ob[k])
            if worker == 0:
                assert len(set(random_heights)) == 5
        finally:
            a.close()
            b.close()
