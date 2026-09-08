import mujoco
import numpy as np
import pytest
from robovision.closure_potential import GraspClosurePotential, closure_quality
from robovision.reverse_curriculum import ReverseGraspEnv
from robovision.joint_env import JointGraspEnv


def make(**kw):
    return ReverseGraspEnv(
        curriculum_level=17,
        replay_fraction=0,
        render_images=False,
        observation="state",
        **kw,
    )


def test_closure_feedback_at_cup_is_smooth_bounded_and_geometry_gated():
    qualities = [
        closure_quality([0, 0, 0], 1, [q, q])
        for q in [0.045, 0.044, 0.04, 0.035, 0.03, 0.0279]
    ]
    assert all((a < b for a, b in zip(qualities, qualities[1:])))
    assert qualities[-1] == pytest.approx(1.0)
    assert closure_quality([0, 0, 0.065], 1, [0.0279, 0.0279]) < 3e-05
    assert closure_quality([0, 0, 0.075], 1, [0.0279, 0.0279]) < 1e-06
    assert closure_quality([0.1, 0, 0], 1, [0.0279, 0.0279]) < 1e-06
    assert closure_quality([0, 0, 0], 0, [0.0279, 0.0279]) == 0
    assert closure_quality([0, 0, 0], 1, [0, 0]) < qualities[0]
    with pytest.raises(ValueError):
        closure_quality([0, np.nan, 0], 1, [0.03, 0.03])


def test_actual_env_closure_potential_and_air_gating():
    env = GraspClosurePotential(make())
    try:
        env.reset(seed=7)
        initial = env.potential()
        env.unwrapped.data.qpos[4:6] = 0.035
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        assert env.potential() > initial
        target = env.unwrapped.cup_position + [0, 0, 0.014 + 0.075]
        env.unwrapped.data.qpos[:4] = env.unwrapped.inverse_kinematics(target)
        env.unwrapped.data.qpos[4:6] = 0.0279
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        assert env.potential() < 1e-05
    finally:
        env.close()


def test_actual_deadline_terminal_zero_and_discounted_telescoping():
    env = GraspClosurePotential(make(max_steps=5))
    try:
        _, info = env.reset(seed=2)
        initial = info["closure_potential"]
        discounted = 0.0
        for t in range(5):
            _, _, done, truncated, info = env.step(np.zeros(5))
            discounted += (
                env.unwrapped.gamma**t * info["reward_components"]["closure_potential"]
            )
            if done or truncated:
                break
        assert done and env.previous_potential == 0
        assert info["closure_potential"] == 0
        assert discounted == pytest.approx(-initial)
    finally:
        env.close()


def test_success_also_zeroes_terminal_potential(monkeypatch):
    base = make()
    env = GraspClosurePotential(base)
    try:
        env.reset(seed=3)
        previous = env.previous_potential
        monkeypatch.setattr(
            base,
            "step",
            lambda action: (
                {},
                50.0,
                True,
                False,
                {
                    "is_success": True,
                    "reward_components": {},
                    "episode_reward_components": {},
                },
            ),
        )
        _, reward, done, _, info = env.step(np.zeros(5))
        assert done and info["is_success"]
        assert info["closure_potential"] == 0 and reward == pytest.approx(
            50.0 - previous
        )
    finally:
        env.close()


def test_disabled_wrapper_preserves_final_height_exactly():
    a = GraspClosurePotential(
        JointGraspEnv(stage=3, observation="state", render_images=False), scale=0.0
    )
    b = JointGraspEnv(stage=3, observation="state", render_images=False)
    try:
        oa, ia = a.reset(seed=42)
        ob, ib = b.reset(seed=42)
        assert ia == ib
        for k in oa:
            np.testing.assert_array_equal(oa[k], ob[k])
        for _ in range(5):
            ar = a.step(np.zeros(5))
            br = b.step(np.zeros(5))
            assert ar[1:] == br[1:]
            for k in ar[0]:
                np.testing.assert_array_equal(ar[0][k], br[0][k])
    finally:
        a.close()
        b.close()
