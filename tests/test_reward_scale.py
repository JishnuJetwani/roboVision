import gymnasium as gym
import numpy as np
import pytest
from robovision.reward_scale import RoleRewardScale


class Task(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position = np.array([0.0], dtype=np.float32)
        self.steps = 0
        self.total = 0.0
        return (self.position.copy(), {"seed": seed})

    def step(self, action):
        self.position += action
        self.steps += 1
        reward = float(self.position[0]) - 0.2
        self.total += reward
        self.last_info = {
            "is_success": self.steps == 3,
            "reward_components": {"position": float(self.position[0]), "time": -0.2},
        }
        if self.steps == 3:
            self.last_info["episode_reward_components"] = {"total": self.total}
        return (self.position.copy(), reward, self.steps == 3, False, self.last_info)


@pytest.mark.parametrize(
    "scale", [0, -1, float("nan"), float("inf"), -float("inf"), True, None, "invalid"]
)
def test_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        RoleRewardScale(Task(), scale)


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_only_rewards_change_and_original_components_preserved(scale):
    base, wrapped = (Task(), RoleRewardScale(Task(), scale))
    initial_base = base.reset(seed=7)
    initial_wrapped = wrapped.reset(seed=7)
    np.testing.assert_array_equal(initial_base[0], initial_wrapped[0])
    assert initial_wrapped[1]["seed"] == 7
    assert initial_wrapped[1]["training_reward_scale"] == scale
    for amount in [0.2, -0.1, 0.3]:
        action = np.array([amount], dtype=np.float32)
        original_action = action.copy()
        a, b = (base.step(action), wrapped.step(action))
        np.testing.assert_array_equal(action, original_action)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(base.position, wrapped.unwrapped.position)
        assert b[1] == pytest.approx(a[1] * scale)
        assert a[2:4] == b[2:4]
        assert a[4]["is_success"] == b[4]["is_success"]
        for key in ("reward_components", "episode_reward_components"):
            if key not in a[4]:
                continue
            assert b[4]["raw_" + key] == a[4][key]
            assert b[4][key] == {k: v * scale for k, v in a[4][key].items()}
            assert wrapped.unwrapped.last_info[key] == a[4][key]
            assert b[4][key] is not wrapped.unwrapped.last_info[key]
            assert b[4]["raw_" + key] is not wrapped.unwrapped.last_info[key]
        assert b[4]["training_reward_scale"] == scale
