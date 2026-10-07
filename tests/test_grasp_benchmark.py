import random
import numpy as np
import pytest
import torch
from robovision import grasp_benchmark as bench


def test_wilson_uncertainty_including_perfect_and_empty():
    assert bench.wilson_interval(0, 0) == [None, None]
    low, high = bench.wilson_interval(180, 200)
    assert low == pytest.approx(0.85059, abs=0.0001)
    assert high == pytest.approx(0.93433, abs=0.0001)
    assert bench.wilson_interval(200, 200)[0] < 1
    with pytest.raises(ValueError):
        bench.wilson_interval(201, 200)


def test_seed_validation_before_environment_creation():
    for seeds in [range(199), [1] * 200, [True] + list(range(1, 200))]:
        with pytest.raises(ValueError):
            bench.evaluate_hard_grasper(None, seeds)


def test_hard_configuration_ablations_rng_and_full_rows(monkeypatch):
    calls = []

    class Env:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.params = dict(
                spawn_x=0.3, spawn_y=0.01, cup_mass=0.1, grip_friction=0.7
            )

        def reset(self, seed):
            self.step_count = 0
            return (
                {"image": np.full((6, 2, 2), 4, np.uint8), "proprio": np.array([9.0])},
                {"clearance": 0.0},
            )

        def step(self, action):
            self.step_count += 1
            obs = {"image": np.full((6, 2, 2), 8, np.uint8), "proprio": np.array([9.0])}
            return (
                obs,
                1.0,
                self.step_count == 2,
                False,
                dict(
                    contacts=[True, True],
                    clearance=0.07,
                    is_success=True,
                    reason="success",
                ),
            )

        def close(self):
            calls.append("closed")

    class Policy:
        training = True

        def __init__(self):
            self.images = []

        def predict(self, obs, deterministic):
            self.images.append(float(obs["image"].mean()))
            assert obs["proprio"][0] == 9
            self.training = False
            torch.rand(1)
            np.random.rand()
            random.random()
            return (np.zeros(5), None)

        def train(self, mode):
            self.training = mode

    monkeypatch.setattr(bench, "JointGraspEnv", Env)
    policy = Policy()
    state = torch.random.get_rng_state().clone()
    nstate = np.random.get_state()
    pstate = random.getstate()
    result = bench.evaluate_hard_grasper(policy, range(200), stochastic=True)
    assert calls == [
        dict(
            stage=3,
            observation="pixels",
            render_images=True,
            gamma=0.995,
            max_steps=500,
        ),
        "closed",
    ]
    assert policy.training
    assert torch.equal(state, torch.random.get_rng_state())
    np.testing.assert_array_equal(nstate[1], np.random.get_state()[1])
    assert pstate == random.getstate()
    assert policy.images[:4] == [4, 8, 4, 8]
    assert policy.images[400:404] == [4, 4, 4, 4]
    assert policy.images[800:804] == [0, 0, 0, 0]
    assert len(result["evaluations"]) == 6
    for report in result["evaluations"].values():
        assert report["successes"] == report["episodes"] == 200
        assert (
            report["bilateral_contact"]
            == report["lifted_6cm_with_bilateral_contact"]
            == 200
        )
        assert report["strata"]["cup_mass"]["below"]["episodes"] == 0
        assert len(report["rows"]) == 200


def test_stateful_policy_resets_for_every_seed_ablation_and_sampling_mode(monkeypatch):
    episodes = []

    class Env:
        def __init__(self, **kwargs):
            assert kwargs == bench.ENV_CONFIG
            self.params = dict(
                spawn_x=0.3, spawn_y=0.01, cup_mass=0.1, grip_friction=0.7
            )

        def reset(self, seed):
            self.step_count = 0
            self.image_value = seed % 100 + 10
            episodes.append(seed)
            return (self.observation(), {"clearance": 0.0})

        def observation(self):
            return dict(
                image=np.full(
                    (6, 2, 2), self.image_value + 50 * self.step_count, np.uint8
                ),
                proprio=np.array([1.0 - self.step_count / 500.0], np.float32),
            )

        def step(self, action):
            assert action.shape == (5,)
            self.step_count += 1
            return (
                self.observation(),
                0.0,
                self.step_count == 2,
                False,
                dict(
                    contacts=[False, False],
                    clearance=0.0,
                    is_success=False,
                    reason="test_deadline",
                ),
            )

        def close(self):
            pass

    class StatefulPolicy:
        def __init__(self):
            self.reset_count = 0
            self.predictions = []

        def reset(self):
            self.reset_count += 1
            assert self.reset_count == len(episodes)
            self.local_actions = 0
            self.committed = False

        def predict(self, observation, deterministic):
            assert self.local_actions < 2, "Commitment leaked into the next episode"
            assert self.committed == bool(self.local_actions)
            self.predictions.append(
                (
                    episodes[-1],
                    deterministic,
                    self.local_actions,
                    float(observation["image"].mean()),
                )
            )
            assert observation["proprio"][0] == pytest.approx(
                1.0 - self.local_actions / 500.0
            )
            self.local_actions += 1
            self.committed = True
            return (np.zeros(5, np.float32), None)

    monkeypatch.setattr(bench, "JointGraspEnv", Env)
    policy = StatefulPolicy()
    result = bench.evaluate_hard_grasper(policy, range(200), stochastic=True)
    assert policy.reset_count == 200 * 3 * 2
    assert len(policy.predictions) == policy.reset_count * 2
    for mode_index, deterministic in enumerate((True, False)):
        for condition_index, condition in enumerate(("normal", "frozen", "black")):
            offset = (mode_index * 3 + condition_index) * 400
            for seed in range(200):
                first, second = policy.predictions[
                    offset + seed * 2 : offset + seed * 2 + 2
                ]
                image = seed % 100 + 10
                assert first[:3] == (seed, deterministic, 0)
                assert second[:3] == (seed, deterministic, 1)
                assert [first[3], second[3]] == (
                    [image, image + 50]
                    if condition == "normal"
                    else [image, image]
                    if condition == "frozen"
                    else [0.0, 0.0]
                )
    assert all((item["episodes"] == 200 for item in result["evaluations"].values()))
