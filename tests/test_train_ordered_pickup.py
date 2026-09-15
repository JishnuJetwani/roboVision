from types import SimpleNamespace
import copy
import json
import random
import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from robovision.ordered_pickup_policy import OrderedPickupEnv
import robovision.train_ordered_pickup as trainer


class PhysicalToy(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(0, 255, (6, 96, 96), np.uint8),
            "proprio": gym.spaces.Box(-1, 1, (18,), np.float32),
        }
    )
    action_space = gym.spaces.Box(-1, 1, (5,), np.float32)
    gamma = 0.999
    control_dt = 0.02
    max_steps = 500

    def __init__(self, *, horizon=3, **kwargs):
        self.horizon = horizon
        self.grasp_position = np.array([0.0, 0.0, 0.014])
        self.cup_position = np.zeros(3)
        self._grasp_sid = 0
        self.data = SimpleNamespace(
            site_xmat=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]])
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return (self.obs(), {})

    def obs(self):
        return dict(
            image=np.zeros((6, 96, 96), np.uint8), proprio=np.zeros(18, np.float32)
        )

    def step(self, action):
        self.step_count += 1
        done = self.step_count == self.horizon
        return (
            self.obs(),
            1.0,
            done,
            False,
            dict(
                is_success=False,
                centered_success=False,
                original_success_ever=done,
                reason="timeout" if done else "",
                contacts=[True, True],
                clearance=0.06,
                upright=1.0,
                cup_speed=0.0,
            ),
        )

    def specification(self):
        return dict(toy=True, physics_validation_claimed=False)


class TinyFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=8)
        self.linear = nn.Linear(18, 8)

    def forward(self, obs):
        return self.linear(obs["proprio"])


def toy_ppo(env=None, *, seed=7, gamma=0.999):
    return PPO(
        "MultiInputPolicy",
        env or PhysicalToy(),
        n_steps=32,
        batch_size=32,
        learning_rate=0.0001,
        gamma=gamma,
        seed=seed,
        device="cpu",
        policy_kwargs=dict(
            features_extractor_class=TinyFeatures,
            net_arch=[128, 128],
            share_features_extractor=False,
        ),
    )


@pytest.fixture
def pair():
    torch.set_num_threads(1)
    return [toy_ppo(seed=7), toy_ppo(seed=8)]


@pytest.mark.parametrize("duration", [5, 25])
def test_budget_reserves_terminal_commitments_not_continue_duration(duration):
    settings = trainer.ordered_duration_settings(duration)
    assert settings["manager_gamma"] == 0.999**duration
    assert settings["checkpoint_chunk_manager_steps"] == 128
    assert settings["checkpoint_chunk_physical_upper_bound"] == 64000
    assert trainer.planned_ordered_steps(63999) == 0
    assert trainer.planned_ordered_steps(64000) == 128
    assert trainer.planned_ordered_steps(1000000) == 128
    for remaining in range(0, 200000, 137):
        assert trainer.planned_ordered_steps(remaining) * 500 <= remaining
    for invalid in (-1, 64000.0, True):
        with pytest.raises(ValueError):
            trainer.planned_ordered_steps(invalid)


def test_recovery_bills_up_to_500_actions_for_each_lost_decision():
    bounds = trainer.ordered_recovery_bounds(
        dict(lost_interactions_lower_bound=4, lost_interactions_upper_bound=132),
        saved_physical_steps=100,
    )
    assert bounds["lost_physical_steps_lower_bound"] == 4
    assert bounds["lost_physical_steps_upper_bound"] == 66000
    with pytest.raises(RuntimeError, match="chunk intent"):
        trainer.ordered_recovery_bounds(
            dict(lost_interactions_lower_bound=0, lost_interactions_upper_bound=None),
            saved_physical_steps=100,
        )
    assert (
        trainer.ordered_recovery_bounds(
            dict(lost_interactions_lower_bound=0, lost_interactions_upper_bound=None),
            saved_physical_steps=0,
        )["lost_physical_steps_upper_bound"]
        == 0
    )


def test_fresh_prior_has_exact_probability_without_copying_heads_or_sharing_features(
    pair,
):
    env = OrderedPickupEnv(PhysicalToy(), pair)
    manager = toy_ppo(env, gamma=0.999**5)
    before_value = trainer._parameter_digest(manager.policy.value_net.state_dict())
    features = trainer.initialize_manager_features(manager, pair[0])
    prior = trainer.initialize_commit_prior(manager)
    assert features["independent_trainable_storage"] and prior["architectural_prior"]
    assert not manager.policy.optimizer.state
    assert before_value == trainer._parameter_digest(
        manager.policy.value_net.state_dict()
    )
    for _ in range(3):
        obs = PhysicalToy().obs()
        obs["proprio"] = np.random.uniform(-1, 1, 18).astype(np.float32)
        tensors, _ = manager.policy.obs_to_tensor(obs)
        probabilities = (
            manager.policy.get_distribution(tensors)
            .distribution.probs.detach()
            .cpu()
            .numpy()
        )
        np.testing.assert_allclose(probabilities, [[0.95, 0.05]], rtol=1e-06)
    for name in ("pi_features_extractor", "vf_features_extractor"):
        copied, source = (getattr(manager.policy, name), getattr(pair[0].policy, name))
        assert trainer._parameter_digest(
            copied.state_dict()
        ) == trainer._parameter_digest(source.state_dict())
        assert not {p.data_ptr() for p in copied.parameters()} & {
            p.data_ptr() for p in source.parameters()
        }
        assert all((p.requires_grad for p in copied.parameters()))
    trainer.validate_ordered_manager(manager, 5)
    with pytest.raises(ValueError, match="gamma"):
        trainer.validate_ordered_manager(manager, 25)
    with pytest.raises(ValueError, match="binary"):
        trainer.validate_ordered_manager(
            SimpleNamespace(gamma=0.999**5, action_space=gym.spaces.Discrete(3))
        )


@pytest.mark.parametrize("scale", [1.0, 0.01])
@pytest.mark.parametrize("reset_recipe", ["arrivals-v1", "full-start-v2"])
def test_registered_worker_families_use_exact_reset_ranges_and_pool(
    monkeypatch, pair, scale, reset_recipe
):
    built = []

    def base(**kwargs):
        built.append(kwargs)
        return PhysicalToy()

    monkeypatch.setattr(trainer, "CenteredGraspEnv", base)
    monkeypatch.setattr(trainer, "CenteredArrivalGraspEnv", base)
    pool = {"provenance": {"test": True}}
    for worker in range(4):
        env = trainer.make_ordered_env(
            worker,
            123,
            pair,
            pool,
            option_steps=25,
            render_images=False,
            training_reward_scale=scale,
            reset_recipe=reset_recipe,
        )
        assert env.action_space.n == 2 and env.unwrapped.max_steps == 500
        assert env.env.manager_gamma == 0.999**25
        assert env.env.specification()["training_reward_scale"] == scale
        env.close()
    assert built[0]["height_bands"] == ((0.1, 0.14, 1.0),)
    assert built[1]["height_bands"] == ((0.025, 0.1, 1.0),)
    if reset_recipe == "arrivals-v1":
        assert built[2]["arrival_pool"] is built[3]["arrival_pool"] is pool
    else:
        assert built[2]["height_bands"] == built[0]["height_bands"]
        assert built[3]["height_bands"] == built[1]["height_bands"]
        assert all(("arrival_pool" not in row for row in built))
    assert all(("xy_half_range" not in row for row in built if "height_bands" in row))
    assert all(
        (row["gamma"] == 0.999 and row["observation"] == "pixels" for row in built)
    )
    assert [row["seed"] for row in built] == [123, 124, 125, 126]


def test_arrival_loader_requires_hash_verified_approach_pool(tmp_path, monkeypatch):
    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return {
            "provenance": {
                "sha256": kwargs["expected_sha256"],
                "source_phase": "approach",
            }
        }

    monkeypatch.setattr(trainer, "load_centered_arrival_pool", load)
    source = dict(run="transfer", sha256="a" * 64)
    _, record, path = trainer.load_ordered_arrival(tmp_path, source)
    assert path == tmp_path / "transfer/approach-arrivals.pkl.gz"
    assert calls == [(path, dict(expected_sha256="a" * 64, expected_phase="approach"))]
    assert record["pool_provenance"]["source_phase"] == "approach"
    for invalid in (
        {"run": "../transfer", "sha256": "a" * 64},
        {"run": "transfer", "sha256": "A" * 64},
        {"run": "transfer"},
    ):
        with pytest.raises(ValueError):
            trainer.load_ordered_arrival(tmp_path, invalid)


@pytest.mark.parametrize("reset_recipe", ["arrivals-v1", "full-start-v2"])
def test_audit_keeps_original_success_separate_and_counts_terminal_physical_actions(
    reset_recipe,
):
    audit = trainer.OrderedAudit(option_steps=25, reset_recipe=reset_recipe)
    audit.model = SimpleNamespace(num_timesteps=4)
    base = dict(
        is_success=False,
        centered_success=False,
        original_success_ever=True,
        reason="timeout",
        episode_physics_steps=500,
        expert_action_counts=[25, 475],
        episode_physical_return=1.0,
        episode=dict(r=1.0, l=2, t=0.1),
    )
    infos = [
        dict(base, option_steps_executed=count, option_index=option)
        for count, option in ((25, 0), (500, 1), (2, 0), (475, 1))
    ]
    audit.locals = dict(infos=infos, dones=[False, True, True, True])
    assert audit._on_step()
    assert audit.physical_steps == 1002
    report = audit.report()
    assert report["manager_decisions"] == 4 and report["commit_decisions"] == 2
    assert report["completed_episodes"] == report["original_successes"] == 3
    assert report["centered_successes"] == report["task_successes"] == 0
    assert report["by_worker"][1]["physical_actions"] == 500
    assert report["worker_roles"] == list(trainer.ordered_worker_roles(reset_recipe))
    assert [row["role"] for row in report["by_worker"]] == report["worker_roles"]
    assert all(
        (
            row["worker_role"] == report["worker_roles"][row["worker"]]
            for row in audit.episodes
        )
    )
    assert (
        trainer.OrderedAudit(
            audit.state(), option_steps=25, reset_recipe=reset_recipe
        ).state()
        == audit.state()
    )
    other_recipe = "full-start-v2" if reset_recipe == "arrivals-v1" else "arrivals-v1"
    with pytest.raises(ValueError, match="reset recipe differs"):
        trainer.OrderedAudit(audit.state(), option_steps=25, reset_recipe=other_recipe)
    invalid = copy.deepcopy(audit.state())
    invalid["episodes"][0]["worker_role"] = "incorrect-reset-role"
    with pytest.raises(ValueError, match="episode role"):
        trainer.OrderedAudit(invalid, option_steps=25, reset_recipe=reset_recipe)
    audit.locals["dones"][1] = False
    with pytest.raises(RuntimeError, match="terminal contract"):
        audit._on_step()


@pytest.fixture
def toy_training(tmp_path, monkeypatch, pair):
    paths = []
    for index in range(2):
        path = tmp_path / f"expert-{index}.txt"
        path.write_text("Local toy source only; never a remote model download")
        paths.append(path)
    pool_path = tmp_path / "toy-pool.txt"
    pool_path.write_text("Local toy pool placeholder")
    pool_record = dict(
        run="toy-pool", sha256=trainer.sha256(pool_path), pool_provenance={"toy": True}
    )
    monkeypatch.setattr(
        trainer, "load_ordered_arrival", lambda *args: ({}, pool_record, pool_path)
    )

    def source(volume_root, spec, **kwargs):
        index = int(spec["run"])
        return (
            pair[index],
            dict(
                **spec,
                policy_sha256=trainer.sha256(paths[index]),
                metadata_sha256="a" * 64,
                inherited_model_steps=0,
            ),
            paths[index],
        )

    monkeypatch.setattr(trainer, "_source", source)
    monkeypatch.setattr(trainer, "NormalizedGraspCNN", TinyFeatures)
    monkeypatch.setattr(
        trainer,
        "make_ordered_env",
        lambda worker,
        seed,
        experts,
        pool,
        option_steps,
        training_reward_scale,
        reset_recipe="arrivals-v1": Monitor(
            trainer.OrderedTrainingRewardScale(
                OrderedPickupEnv(PhysicalToy(), experts, option_steps=option_steps),
                training_reward_scale,
            )
        ),
    )
    evaluations = []

    def evaluate(manager, experts, *, final=False, **kwargs):
        evaluations.append((manager.num_timesteps, final))
        return dict(rows=[], scores={}, final=final, evaluation_physical_steps=0)

    monkeypatch.setattr(trainer, "evaluate_ordered", evaluate)
    sources = [dict(run=str(index), checkpoint="toy.zip") for index in range(2)]
    return (
        sources,
        dict(run="toy-pool", sha256=trainer.sha256(pool_path)),
        pair,
        evaluations,
    )


def test_real_ppo_one_chunk_preserves_donors_and_complete_retry_collects_nothing(
    tmp_path, toy_training
):
    sources, arrival, experts, evaluations = toy_training
    hashes = [trainer._parameter_digest(expert.get_parameters()) for expert in experts]
    kwargs = dict(
        arrival_source=arrival, target_physics_steps=64000, seed=37, device="cpu"
    )
    result = trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
    assert result["new_manager_steps"] == 128 and result["new_physical_steps"] == 384
    assert result["actual_physical_upper_bound"] == 384
    assert result["unused_physical_budget"] == 63616
    assert result["training_scores"]["original_successes"] == 128
    assert result["training_scores"]["centered_successes"] == 0
    assert len(result["gates"]) == len(result["optimizer_reports"]) == 1
    assert result["optimizer_reports"][0]["optimizer_parameter_steps_max"] == 10
    assert evaluations == [(0, False), (128, False), (128, True)]
    assert hashes == [
        trainer._parameter_digest(expert.get_parameters()) for expert in experts
    ]
    assert (
        trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
        == result
    )
    assert evaluations == [(0, False), (128, False), (128, True)]


def test_preempted_rollout_reserves_64000_lost_actions_and_does_not_train_again(
    tmp_path, toy_training, monkeypatch
):
    sources, arrival, _, _ = toy_training
    original = trainer.OrderedAudit._on_step
    failed = []

    def interrupt(audit):
        result = original(audit)
        if not failed:
            failed.append(True)
            raise RuntimeError("Synthetic preemption during physical rollout")
        return result

    monkeypatch.setattr(trainer.OrderedAudit, "_on_step", interrupt)
    kwargs = dict(
        arrival_source=arrival, target_physics_steps=64000, seed=37, device="cpu"
    )
    with pytest.raises(RuntimeError, match="preemption"):
        trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
    result = trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
    assert result["new_manager_steps"] == result["new_physical_steps"] == 0
    assert result["actual_physical_upper_bound"] == 64000
    assert result["recoveries"][0]["lost_physical_steps_upper_bound"] == 64000
    assert result["recoveries"][0]["fresh_episodes"]
    assert result["recoveries"][0]["rollout_buffer_empty"]


def test_pending_final_gate_resume_preserves_optimizer_and_collects_no_more_physics(
    tmp_path, toy_training, monkeypatch
):
    sources, arrival, _, evaluations = toy_training
    original = trainer.evaluate_ordered
    failed = []
    before = []

    def interrupt(manager, experts, **kwargs):
        if manager.num_timesteps and (not kwargs["final"]) and (not failed):
            failed.append(True)
            before.append(trainer._parameter_digest(manager.get_parameters()))
            raise RuntimeError("Synthetic gate evaluation preemption")
        if manager.num_timesteps:
            assert trainer._parameter_digest(manager.get_parameters()) == before[0]
        return original(manager, experts, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_ordered", interrupt)
    kwargs = dict(
        arrival_source=arrival, target_physics_steps=64000, seed=37, device="cpu"
    )
    with pytest.raises(RuntimeError, match="preemption"):
        trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
    result = trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
    assert result["new_manager_steps"] == 128 and result["new_physical_steps"] == 384
    assert result["recoveries"][0]["lost_physical_steps_upper_bound"] == 0
    assert len(result["gates"]) == 1
    assert evaluations == [(0, False), (128, False), (128, True)]


def test_frozen_evaluation_preserves_rng_optimizer_and_labels_with_exact_physical_counts(
    monkeypatch, pair
):
    monkeypatch.setattr(
        trainer, "CenteredGraspEnv", lambda **kwargs: PhysicalToy(horizon=25)
    )
    manager = toy_ppo(OrderedPickupEnv(PhysicalToy(), pair), gamma=0.999**5)
    trainer.initialize_commit_prior(manager)
    state_before = trainer._parameter_digest(manager.get_parameters())
    random.seed(124)
    np.random.seed(124)
    torch.manual_seed(124)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    result = trainer.evaluate_ordered(manager, pair, final=True, seed=123)
    assert random.getstate() == python_rng
    assert all((np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state())))
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert state_before == trainer._parameter_digest(manager.get_parameters())
    assert len(result["rows"]) == 8 and result["evaluation_physical_steps"] == 200
    assert result["scores"]["deterministic"]["manager_decisions"] == 20
    assert result["scores"]["deterministic"]["task_successes"] == 0
    assert result["scores"]["deterministic"]["original_successes"] == 4
    assert all(
        (
            row["physical_steps"] == sum(row["expert_action_counts"]) == 25
            for row in result["rows"]
        )
    )
    assert result["evaluation_manager_decisions"] == sum(
        (row["manager_decisions"] for row in result["rows"])
    )
    assert all(
        (
            len(row["decision_trace"]) == row["manager_decisions"]
            for row in result["rows"]
        )
    )
    assert result["manager_probability_order"] == ["continue_approach", "commit_pickup"]
    for row in result["rows"]:
        for decision in row["decision_trace"]:
            np.testing.assert_allclose(
                decision["manager_probabilities"], [0.95, 0.05], rtol=1e-06
            )
    assert (
        result["policy_and_optimizer_sha256_before"]
        == result["policy_and_optimizer_sha256_after"]
    )


def test_frozen_checkpoint_entry_fresh40_separates_control_and_preserves_trained_adam(
    tmp_path, monkeypatch, pair
):
    class OriginalToy(PhysicalToy):
        def step(self, action):
            obs, reward, done, truncated, info = super().step(action)
            return (obs, reward, done, truncated, dict(info, is_success=done))

    monkeypatch.setattr(
        trainer, "CenteredGraspEnv", lambda **kwargs: PhysicalToy(horizon=2)
    )
    monkeypatch.setattr(
        trainer, "GeneralizationGraspEnv", lambda **kwargs: OriginalToy(horizon=2)
    )
    manager = toy_ppo(OrderedPickupEnv(PhysicalToy(), pair), gamma=0.999**5)
    trainer.initialize_commit_prior(manager)
    manager.learn(32)
    assert manager.policy.optimizer.state
    models = [manager, *pair]
    before = [trainer._parameter_digest(model.get_parameters()) for model in models]
    paths = []
    for index in range(3):
        path = tmp_path / f"toy-{index}.txt"
        path.write_text("Local synthetic model-source evidence")
        paths.append(path)

    def source(volume_root, spec, **kwargs):
        index = int(spec["run"])
        np.random.seed(37)
        torch.manual_seed(37)
        return (
            models[index],
            dict(**spec, policy_sha256=trainer.sha256(paths[index])),
            paths[index],
        )

    monkeypatch.setattr(trainer, "_source", source)
    sources = [dict(run=str(index), checkpoint="toy.zip") for index in range(3)]
    np.random.seed(112)
    torch.manual_seed(112)
    numpy_rng, torch_rng = (np.random.get_state(), torch.get_rng_state().clone())
    result = trainer.evaluate_ordered_checkpoint(
        tmp_path,
        sources[0],
        sources[1:],
        seed=321,
        profile="fresh40",
        original_task_control=True,
        device="cpu",
    )
    assert all((np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state())))
    assert torch.equal(torch_rng, torch.get_rng_state())
    assert before == [
        trainer._parameter_digest(model.get_parameters()) for model in models
    ]
    assert all(
        (
            len(decision["manager_probabilities"]) == 2
            for row in result["final"]["rows"]
            for decision in row["decision_trace"]
        )
    )
    assert result["optimizer_unchanged"] and result["source_checkpoint_bytes_unchanged"]
    assert (
        result["evaluation_physical_steps"]
        == result["control_evaluation_physical_steps"]
        == 160
    )
    assert (
        len(result["final"]["rows"])
        == len(result["original_task_control"]["rows"])
        == 80
    )
    assert (
        result["final"]["scene_cases"] == result["original_task_control"]["scene_cases"]
    )
    assert result["final"]["task"] == "centered_full_pickup"
    assert result["original_task_control"]["task"] == "original_full_pickup"
    assert result["final"]["scores"]["deterministic"]["task_successes"] == 0
    assert (
        result["original_task_control"]["scores"]["deterministic"]["task_successes"]
        == 40
    )
    for case in result["final"]["scene_cases"]:
        assert all(
            (
                abs(case["height"] - anchor) > 1e-06
                for anchor in (0.025, 0.065, 0.1, 0.14)
            )
        )
        assert case["offset"] == [0.0, 0.0]


@pytest.mark.parametrize("option_steps,horizon", [(5, 12), (25, 63)])
@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_uniform_scale_changes_only_each_discounted_option_reward(
    pair, option_steps, horizon, scale
):
    class ComponentsToy(PhysicalToy):
        def step(self, action):
            observation, reward, done, truncated, info = super().step(action)
            info.update(
                reward_components={"physical": reward},
                episode_reward_components={"physical": self.step_count * reward},
            )
            return (observation, reward, done, truncated, info)

    raw = OrderedPickupEnv(
        ComponentsToy(horizon=horizon), pair, option_steps=option_steps
    )
    scaled = trainer.OrderedTrainingRewardScale(
        OrderedPickupEnv(
            ComponentsToy(horizon=horizon), pair, option_steps=option_steps
        ),
        scale,
    )
    raw_obs, raw_info = raw.reset(seed=71)
    scaled_obs, scaled_info = scaled.reset(seed=71)
    assert raw_info == scaled_info
    for key in raw_obs:
        np.testing.assert_array_equal(raw_obs[key], scaled_obs[key])
    for action in (0, 0, 1):
        ro, rr, rd, rt, ri = raw.step(action)
        so, sr, sd, st, si = scaled.step(action)
        assert sr == rr * scale
        assert (rd, rt) == (sd, st)
        assert ri == si
        assert si["option_discounted_reward"] == rr
        for key in ro:
            np.testing.assert_array_equal(ro[key], so[key])
    assert raw.env.step_count == scaled.unwrapped.step_count == horizon
    assert ri["episode_physical_return"] == horizon
    assert ri["episode_reward_components"] == {"physical": horizon}
    assert ri["expert_action_counts"] == [2 * option_steps, horizon - 2 * option_steps]
    assert scaled.specification()["training_reward_scale"] == scale


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, 0.1, float("nan"), float("inf"), True, "0.01", None]
)
def test_invalid_reward_scale_rejected_before_any_model_or_environment_load(
    tmp_path, monkeypatch, scale
):
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "Invalid reward scale must fail before loading models or constructing environments"
        )

    monkeypatch.setattr(trainer, "load_ordered_arrival", unexpected)
    monkeypatch.setattr(trainer, "CenteredGraspEnv", unexpected)
    with pytest.raises(ValueError, match="training_reward_scale"):
        trainer.train_ordered_pickup(
            tmp_path,
            tmp_path / "run",
            [],
            arrival_source=None,
            training_reward_scale=scale,
        )
    with pytest.raises(ValueError, match="training_reward_scale"):
        trainer.make_ordered_env(0, 1, [], None, training_reward_scale=scale)


def test_reward_scale_is_explicit_in_identity_and_resume_rejects_cross_scale(
    tmp_path, toy_training, monkeypatch
):
    sources, arrival, _, _ = toy_training

    def stop_before_training(*args, **kwargs):
        raise RuntimeError("Synthetic interruption after durable initial checkpoint")

    monkeypatch.setattr(trainer, "evaluate_ordered", stop_before_training)
    kwargs = dict(
        arrival_source=arrival, target_physics_steps=64000, seed=37, device="cpu"
    )
    with pytest.raises(RuntimeError, match="interruption"):
        trainer.train_ordered_pickup(tmp_path, tmp_path / "run", sources, **kwargs)
    import json

    experiment = json.loads((tmp_path / "run/experiment.json").read_text())
    assert experiment["training_reward_scale"] == 1.0
    assert "not applied" in experiment["evaluation_reward_units"]
    assert all(
        (worker["training_reward_scale"] == 1.0 for worker in experiment["workers"])
    )
    with pytest.raises(ValueError, match="different recipe, source or configuration"):
        trainer.train_ordered_pickup(
            tmp_path, tmp_path / "run", sources, training_reward_scale=0.01, **kwargs
        )


def test_probability_probe_preserves_active_training_mode_adam_counters_and_rng(pair):
    manager = toy_ppo(OrderedPickupEnv(PhysicalToy(), pair), gamma=0.999**5)
    trainer.initialize_commit_prior(manager)
    manager.learn(32)
    manager.policy.train(True)
    obs = PhysicalToy().obs()
    before = trainer._parameter_digest(manager.get_parameters())
    counters = (manager.num_timesteps, manager._n_updates)
    random.seed(212)
    np.random.seed(212)
    torch.manual_seed(212)
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    probabilities = trainer.manager_decision_probabilities(manager, obs)
    assert len(probabilities) == 2 and np.isclose(sum(probabilities), 1.0)
    assert all((0.0 <= value <= 1.0 for value in probabilities))
    assert manager.policy.training
    assert before == trainer._parameter_digest(manager.get_parameters())
    assert counters == (manager.num_timesteps, manager._n_updates)
    assert random.getstate() == python_rng
    assert all((np.array_equal(a, b) for a, b in zip(numpy_rng, np.random.get_state())))
    assert torch.equal(torch_rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "values",
    [[[0.9, 0.2]], [[float("nan"), 0.5]], [[1.1, -0.1]], [[1.0]], [[0.2, 0.3, 0.5]]],
)
def test_probability_probe_rejects_invalid_binary_probabilities_and_restores_mode(
    pair, monkeypatch, values
):
    manager = toy_ppo(OrderedPickupEnv(PhysicalToy(), pair), gamma=0.999**5)
    manager.policy.train(True)

    def invalid_distribution(tensors):
        assert not manager.policy.training
        return SimpleNamespace(distribution=SimpleNamespace(probs=torch.tensor(values)))

    monkeypatch.setattr(manager.policy, "get_distribution", invalid_distribution)
    with pytest.raises(ValueError, match="two finite normalized"):
        trainer.manager_decision_probabilities(manager, PhysicalToy().obs())
    assert manager.policy.training


@pytest.fixture
def warm_source(tmp_path, toy_training):
    sources, arrival, experts, evaluations = toy_training
    source = trainer.train_ordered_pickup(
        tmp_path,
        tmp_path / "parent",
        sources,
        arrival_source=arrival,
        target_physics_steps=64000,
        seed=37,
        training_reward_scale=0.01,
        device="cpu",
    )
    return (sources, arrival, source, dict(run="parent", checkpoint="policy.zip"))


@pytest.mark.parametrize("reset_recipe", ["arrivals-v1", "full-start-v2"])
def test_warm_continuation_preserves_trained_actor_critic_adam_then_bills_only_new_work(
    tmp_path, toy_training, warm_source, monkeypatch, reset_recipe
):
    sources, arrival, parent, source = warm_source
    path = tmp_path / "parent/policy.zip"
    raw_hash = trainer.sha256(path)
    loaded = PPO.load(path, device="cpu")
    expected = trainer._parameter_digest(loaded.get_parameters())
    assert loaded.policy.optimizer.state and loaded.num_timesteps == 128
    expected_optimizer = trainer._parameter_digest(loaded.policy.optimizer.state_dict())
    seen_baseline = []
    old_evaluate = trainer.evaluate_ordered

    def evaluate(manager, experts, **kwargs):
        if manager.num_timesteps == 128 and (not kwargs["final"]):
            seen_baseline.append(True)
            assert trainer._parameter_digest(manager.get_parameters()) == expected
            assert (
                trainer._parameter_digest(manager.policy.optimizer.state_dict())
                == expected_optimizer
            )
            assert (
                manager._n_updates == loaded._n_updates
                and manager._episode_num == loaded._episode_num
            )
            assert manager.seed == 83 and manager.get_env()._seeds == [83, 84, 85, 86]
            assert manager._last_obs is None and manager.rollout_buffer.pos == 0
        return old_evaluate(manager, experts, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_ordered", evaluate)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Warm initialization must not overwrite features or learned action logits"
        )

    monkeypatch.setattr(trainer, "initialize_manager_features", forbidden)
    monkeypatch.setattr(trainer, "initialize_commit_prior", forbidden)
    kwargs = dict(
        arrival_source=arrival,
        manager_source=source,
        target_physics_steps=64000,
        seed=83,
        training_reward_scale=0.01,
        device="cpu",
        reset_recipe=reset_recipe,
    )
    result = trainer.train_ordered_pickup(
        tmp_path, tmp_path / "child", sources, **kwargs
    )
    assert seen_baseline == [True]
    assert (
        result["source_steps"]
        == result["starting_steps"]
        == result["inherited_manager_steps"]
        == 128
    )
    assert (
        result["final_steps"] == 256
        and result["new_steps"] == result["new_manager_steps"] == 128
    )
    assert result["new_physical_steps"] == result["actual_physical_upper_bound"] == 384
    assert result["inherited_ppo_updates"] == result["new_ppo_updates"] == 5
    assert result["inherited_model_episode_counter"] == loaded._episode_num
    assert result["new_completed_training_episodes"] == 128
    assert result["model_episode_counter_delta"] == 0
    assert result["training_scores"]["manager_decisions"] == 128
    assert result["optimizer_reports"][0]["optimizer_parameter_steps_min"] == 20.0
    assert result["commit_prior"] is None and result["feature_initialization"] is None
    assert result["initial_commit_probability"] is None
    assert {k: result["manager_source"][k] for k in ("run", "checkpoint")} == source
    assert result["requested_manager_source"] == source
    assert result["warm_manager_provenance"]["source_identity"] == parent["identity"]
    assert result["source_policy_sha256"] == raw_hash
    assert result["reset_recipe"] == result["identity"]["reset_recipe"] == reset_recipe
    assert (
        result["worker_roles"]
        == result["training_scores"]["worker_roles"]
        == list(trainer.ordered_worker_roles(reset_recipe))
    )
    assert result["arrival_source"] == parent["arrival_source"]
    assert result["arrival_resets_used"] == (reset_recipe == "arrivals-v1")
    transition = result["reset_curriculum_transition"]
    assert transition["source_worker_roles"] == list(trainer.WORKER_ROLES)
    assert transition["target_worker_roles"] == result["worker_roles"]
    assert transition["source_reset_recipe"] == "arrivals-v1"
    assert transition["target_reset_recipe"] == reset_recipe
    assert transition["intentional_distribution_change"] == (
        reset_recipe == "full-start-v2"
    )
    assert transition["reward_ppo_experts_unchanged"]
    warm = result["warm_initialization"]
    assert (
        warm["parameter_sha256_in_source"]
        == warm["parameter_sha256_after_load"]
        == warm["parameter_sha256_before_training"]
        == expected
    )
    assert warm["optimizer_sha256_before_training"] == expected_optimizer
    assert warm["actual_queued_worker_seeds"] == [83, 84, 85, 86]
    assert warm["exact_policy_and_adam_restore"] and warm["fresh_global_rng"]
    assert not warm["feature_initialization_repeated"] and (
        not warm["commit_prior_initialization_repeated"]
    )
    assert result["source_manager_bytes_unchanged"] and trainer.sha256(path) == raw_hash
    assert (
        trainer.train_ordered_pickup(tmp_path, tmp_path / "child", sources, **kwargs)
        == result
    )
    assert seen_baseline == [True]


@pytest.mark.parametrize(
    "mismatch",
    [
        "recipe",
        "experts",
        "expert_hash",
        "arrival",
        "duration",
        "scale",
        "ppo",
        "workers",
    ],
)
def test_warm_source_validation_rejects_incompatible_recipe_before_loading(
    tmp_path, warm_source, mismatch, monkeypatch
):
    sources, arrival, parent, source = warm_source
    metadata = copy.deepcopy(parent)
    if mismatch == "recipe":
        metadata["identity"]["recipe"] = "categorical-three-expert-manager"
    elif mismatch == "experts":
        metadata["experts"].reverse()
    elif mismatch == "expert_hash":
        metadata["experts"][0]["policy_sha256"] = "f" * 64
    elif mismatch == "arrival":
        metadata["arrival_source"]["sha256"] = "f" * 64
    elif mismatch == "duration":
        metadata["duration"]["option_steps"] = 25
    elif mismatch == "scale":
        metadata["training_reward_scale"] = 1.0
    elif mismatch == "ppo":
        metadata["ppo"]["gae_lambda"] = 0.95
    else:
        metadata["worker_roles"].reverse()
    (tmp_path / "parent/summary.json").write_text(json.dumps(metadata))

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Incompatible metadata must fail before loading the warm manager"
        )

    monkeypatch.setattr(trainer, "load_warm_ordered_manager", forbidden)
    with pytest.raises(ValueError, match="Warm"):
        trainer.train_ordered_pickup(
            tmp_path,
            tmp_path / "invalid",
            sources,
            arrival_source=arrival,
            manager_source=source,
            target_physics_steps=64000,
            seed=83,
            training_reward_scale=0.01,
            device="cpu",
        )
    assert not (tmp_path / "invalid").exists()


def test_warm_loader_checks_zip_config_and_detects_optimizer_loss(
    tmp_path, warm_source, toy_training, monkeypatch
):
    sources, arrival, parent, source = warm_source
    env = trainer.DummyVecEnv(
        [
            lambda: trainer.make_ordered_env(
                0, 83, toy_training[2], {}, option_steps=5, training_reward_scale=0.01
            )
            for _ in range(4)
        ]
    )
    path = tmp_path / "parent/policy.zip"
    try:
        incompatible = PPO.load(path, device="cpu")
        incompatible.gae_lambda = 0.95
        incompatible.save(tmp_path / "bad-config.zip")
        with pytest.raises(ValueError, match="gae_lambda"):
            trainer.load_warm_ordered_manager(
                tmp_path / "bad-config.zip",
                env,
                option_steps=5,
                seed=83,
                expected_manager_steps=128,
                device="cpu",
            )
        original_load = trainer.PPO.load

        def dropped_adam(*args, **kwargs):
            model = original_load(*args, **kwargs)
            model.policy.optimizer.state.clear()
            return model

        monkeypatch.setattr(trainer.PPO, "load", dropped_adam)
        with pytest.raises(RuntimeError, match="Adam tensors"):
            trainer.load_warm_ordered_manager(
                path,
                env,
                option_steps=5,
                seed=83,
                expected_manager_steps=128,
                device="cpu",
            )
    finally:
        env.close()


@pytest.mark.parametrize("reset_recipe", ["arrivals-v1", "full-start-v2"])
def test_warm_durable_resume_does_not_reload_or_reinitialize_parent(
    tmp_path, warm_source, monkeypatch, reset_recipe
):
    sources, arrival, parent, source = warm_source
    original_evaluate = trainer.evaluate_ordered
    interrupted = []
    digest_at_gate = []

    def evaluate(manager, experts, **kwargs):
        if manager.num_timesteps == 256 and (not kwargs["final"]) and (not interrupted):
            interrupted.append(True)
            digest_at_gate.append(trainer._parameter_digest(manager.get_parameters()))
            raise RuntimeError("Synthetic warm final-gate preemption")
        if manager.num_timesteps == 256:
            assert (
                trainer._parameter_digest(manager.get_parameters()) == digest_at_gate[0]
            )
        return original_evaluate(manager, experts, **kwargs)

    monkeypatch.setattr(trainer, "evaluate_ordered", evaluate)
    kwargs = dict(
        arrival_source=arrival,
        manager_source=source,
        target_physics_steps=64000,
        seed=83,
        training_reward_scale=0.01,
        device="cpu",
        reset_recipe=reset_recipe,
    )
    with pytest.raises(RuntimeError, match="warm final-gate"):
        trainer.train_ordered_pickup(tmp_path, tmp_path / "child", sources, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Durable resume must use child checkpoint, not the parent warm source"
        )

    monkeypatch.setattr(trainer, "load_warm_ordered_manager", forbidden)
    result = trainer.train_ordered_pickup(
        tmp_path, tmp_path / "child", sources, **kwargs
    )
    assert result["new_manager_steps"] == 128 and result["final_steps"] == 256
    assert (
        result["new_physical_steps"] == 384
        and result["recoveries"][0]["lost_physical_steps_upper_bound"] == 0
    )
    assert result["new_ppo_updates"] == 5 and result["inherited_ppo_updates"] == 5
    assert len(result["gates"]) == 1
    assert result["reset_recipe"] == reset_recipe
    assert result["training_scores"]["worker_roles"] == list(
        trainer.ordered_worker_roles(reset_recipe)
    )


def test_warm_source_same_output_or_changed_bytes_are_rejected(
    tmp_path, warm_source, monkeypatch
):
    sources, arrival, parent, source = warm_source
    kwargs = dict(
        arrival_source=arrival,
        manager_source=source,
        target_physics_steps=64000,
        seed=83,
        training_reward_scale=0.01,
        device="cpu",
    )
    with pytest.raises(ValueError, match="new output directory"):
        trainer.train_ordered_pickup(tmp_path, tmp_path / "parent", sources, **kwargs)
    original_evaluate = trainer.evaluate_ordered

    def corrupt_parent(manager, experts, **kwargs):
        result = original_evaluate(manager, experts, **kwargs)
        if kwargs["final"]:
            with (tmp_path / "parent/policy.zip").open("ab") as stream:
                stream.write(b"synthetic external corruption")
        return result

    monkeypatch.setattr(trainer, "evaluate_ordered", corrupt_parent)
    with pytest.raises(RuntimeError, match="source manager checkpoint bytes changed"):
        trainer.train_ordered_pickup(tmp_path, tmp_path / "child", sources, **kwargs)
    assert not (tmp_path / "child/summary.json").exists()


def test_warm_gate_source_resolves_its_own_counter_and_cannot_replace_child_resume_identity(
    tmp_path, warm_source, monkeypatch
):
    sources, arrival, parent, source = warm_source
    gate_source = dict(run="parent", checkpoint=parent["gates"][0]["checkpoint"])
    path, record = trainer.resolve_ordered_continuation(
        tmp_path,
        gate_source,
        parent["experts"],
        parent["arrival_source"],
        option_steps=5,
        training_reward_scale=0.01,
    )
    assert path.name == "step-128.zip" and record["expected_manager_steps"] == 128
    assert (
        record["source_new_physical_steps"]
        == parent["gates"][0]["new_physical_steps"]
        == 384
    )
    original_evaluate = trainer.evaluate_ordered

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic warm baseline interruption")

    monkeypatch.setattr(trainer, "evaluate_ordered", interrupt)
    kwargs = dict(
        arrival_source=arrival,
        target_physics_steps=64000,
        seed=83,
        training_reward_scale=0.01,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="baseline interruption"):
        trainer.train_ordered_pickup(
            tmp_path, tmp_path / "child", sources, manager_source=source, **kwargs
        )
    monkeypatch.setattr(trainer, "evaluate_ordered", original_evaluate)
    with pytest.raises(ValueError, match="different recipe, source or configuration"):
        trainer.train_ordered_pickup(
            tmp_path, tmp_path / "child", sources, manager_source=gate_source, **kwargs
        )


@pytest.mark.parametrize("active_provenance", ["experiment", "unfinished_summary"])
def test_warm_source_must_be_complete_even_when_its_gate_zip_is_already_valid(
    tmp_path, warm_source, active_provenance
):
    sources, arrival, parent, source = warm_source
    gate_source = dict(run="parent", checkpoint=parent["gates"][0]["checkpoint"])
    summary_path = tmp_path / "parent/summary.json"
    if active_provenance == "experiment":
        summary_path.unlink()
    else:
        summary_path.write_text(json.dumps(dict(parent, status="training")))
    with pytest.raises(ValueError, match="completed source run"):
        trainer.train_ordered_pickup(
            tmp_path,
            tmp_path / "child",
            sources,
            arrival_source=arrival,
            manager_source=gate_source,
            target_physics_steps=64000,
            seed=83,
            training_reward_scale=0.01,
            device="cpu",
        )
    assert not (tmp_path / "child").exists()


@pytest.mark.parametrize("invalid", ["unknown", "", None, True, ["arrivals-v1"]])
def test_unknown_reset_recipe_rejected_before_loading_any_sources(
    tmp_path, monkeypatch, invalid
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid reset recipe must not load a source")

    monkeypatch.setattr(trainer, "load_ordered_arrival", forbidden)
    with pytest.raises(ValueError, match="reset_recipe"):
        trainer.train_ordered_pickup(
            tmp_path, tmp_path / "child", [], arrival_source=None, reset_recipe=invalid
        )


def test_full_start_comparison_requires_warm_source_and_still_verifies_pool(
    tmp_path, monkeypatch
):
    calls = []

    def bad_pool(*args):
        calls.append(args)
        raise ValueError("Synthetic arrival pool hash mismatch")

    monkeypatch.setattr(trainer, "load_ordered_arrival", bad_pool)
    with pytest.raises(ValueError, match="completed warm manager"):
        trainer.train_ordered_pickup(
            tmp_path,
            tmp_path / "child",
            [],
            arrival_source=None,
            reset_recipe="full-start-v2",
        )
    assert not calls
    sources = [dict(run=str(i), checkpoint="policy.zip") for i in range(2)]
    with pytest.raises(ValueError, match="pool hash mismatch"):
        trainer.train_ordered_pickup(
            tmp_path,
            tmp_path / "child",
            sources,
            arrival_source=dict(run="pool", sha256="a" * 64),
            reset_recipe="full-start-v2",
            manager_source=dict(run="parent", checkpoint="policy.zip"),
        )
    assert len(calls) == 1 and (not (tmp_path / "child").exists())


@pytest.mark.parametrize("source_recipe", [None, "arrivals-v1", "full-start-v2"])
def test_warm_registered_parent_recipes_and_strict_legacy_role_inference(
    tmp_path, warm_source, source_recipe
):
    _, _, parent, source = warm_source
    metadata = copy.deepcopy(parent)
    if source_recipe is None:
        metadata.pop("reset_recipe")
        metadata["identity"].pop("reset_recipe")
    else:
        metadata["reset_recipe"] = metadata["identity"]["reset_recipe"] = source_recipe
        metadata["worker_roles"] = list(trainer.ordered_worker_roles(source_recipe))
    summary_path = tmp_path / "parent/summary.json"
    summary_path.write_text(json.dumps(metadata))
    for target in ("arrivals-v1", "full-start-v2"):
        _, record = trainer.resolve_ordered_continuation(
            tmp_path,
            source,
            parent["experts"],
            parent["arrival_source"],
            option_steps=5,
            training_reward_scale=0.01,
            reset_recipe=target,
        )
        inferred = source_recipe or "arrivals-v1"
        assert record["source_reset_recipe"] == inferred
        assert record["target_reset_recipe"] == target
        assert record["source_worker_roles"] == list(
            trainer.ordered_worker_roles(inferred)
        )
        assert record["target_worker_roles"] == list(
            trainer.ordered_worker_roles(target)
        )
        assert record["reset_distribution_changed"] == (inferred != target)
        assert record["exact_recipe_match"] == (inferred == target)
        assert record["exact_non_reset_recipe_match"]
    metadata.pop("reset_recipe", None)
    metadata["identity"].pop("reset_recipe", None)
    metadata["worker_roles"] = list(trainer.ordered_worker_roles("full-start-v2"))
    summary_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="worker roles"):
        trainer.resolve_ordered_continuation(
            tmp_path,
            source,
            parent["experts"],
            parent["arrival_source"],
            option_steps=5,
            training_reward_scale=0.01,
            reset_recipe="full-start-v2",
        )


def test_full_start_child_resume_cannot_change_reset_recipe(
    tmp_path, warm_source, monkeypatch
):
    sources, arrival, _, source = warm_source

    def interrupt(*args, **kwargs):
        raise RuntimeError("Synthetic curriculum baseline interruption")

    monkeypatch.setattr(trainer, "evaluate_ordered", interrupt)
    kwargs = dict(
        arrival_source=arrival,
        manager_source=source,
        target_physics_steps=64000,
        seed=83,
        training_reward_scale=0.01,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="baseline interruption"):
        trainer.train_ordered_pickup(
            tmp_path,
            tmp_path / "child",
            sources,
            reset_recipe="full-start-v2",
            **kwargs,
        )
    with pytest.raises(ValueError, match="different recipe, source or configuration"):
        trainer.train_ordered_pickup(
            tmp_path, tmp_path / "child", sources, reset_recipe="arrivals-v1", **kwargs
        )
