"""Frozen nominal confirmation guards, input propagation and work accounting."""

import copy
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import gymnasium as gym
import numpy as np
import pytest
import torch
import robovision.ordered_confirmation as confirmation


def space():
    return gym.spaces.Dict(
        image=gym.spaces.Box(0, 255, (6, 96, 96), np.uint8),
        proprio=gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
    )


class FrozenModel:
    def __init__(self, role):
        self.role = role
        self.observation_space = space()
        self.action_space = (
            gym.spaces.Discrete(2)
            if role == "manager"
            else gym.spaces.Box(-1, 1, (5,), np.float32)
        )
        self.gamma = 0.999**5
        self.policy = torch.nn.Sequential(torch.nn.Linear(18, 5), torch.nn.Dropout(0.2))
        self.policy[1].eval()
        self.policy[0].bias.requires_grad_(False)
        self.policy.optimizer = torch.optim.Adam(self.policy.parameters())
        parameter = self.policy[0].weight
        self.policy.optimizer.state[parameter] = dict(
            step=torch.tensor(7.0),
            exp_avg=torch.full_like(parameter, 0.25),
            exp_avg_sq=torch.full_like(parameter, 0.125),
        )
        self.num_timesteps, self._n_updates, self._episode_num = (123, 9, 4)
        self.rollout_buffer = SimpleNamespace(pos=0, full=False)
        self.calls = []
        self.hook = None

    def predict(self, obs, deterministic=True):
        self.calls.append(
            dict(
                step=int(obs["proprio"][0]),
                pixel=int(obs["image"][0, 0, 0]),
                proprio=obs["proprio"].copy(),
                deterministic=deterministic,
                grad=torch.is_grad_enabled(),
                training=self.policy.training,
            )
        )
        if self.hook:
            self.hook(self)
        if self.role == "manager":
            return (np.array(int(obs["proprio"][0] >= 5)), None)
        return (np.full(5, 0.1 if self.role == "approach" else -0.2, np.float32), None)

    def learn(self, *args, **kwargs):
        raise AssertionError("Frozen evaluation must never call learn")


class ToyEnv:
    control_dt = 0.02
    max_steps = 500
    fail_step = None
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.actions = []
        self.closed = False
        self.instances.append(self)

    def obs(self):
        proprio = np.linspace(-0.5, 0.5, 18, dtype=np.float32)
        proprio[0] = self.step_count
        proprio[-1] = 1 - self.step_count / 500
        return dict(
            image=np.full((6, 96, 96), 10 + self.step_count, np.uint8), proprio=proprio
        )

    def reset(self, *, seed):
        self.step_count = 0
        self.ended = False
        self.reset_random = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
        )
        return (self.obs(), {})

    def step(self, action):
        assert not self.ended, "No terminal overstep"
        if self.fail_step == self.step_count + 1:
            raise RuntimeError("Diagnostic simulator failure")
        np.testing.assert_array_equal(
            action, np.full(5, 0.1 if self.step_count < 5 else -0.2, np.float32)
        )
        self.actions.append(action.copy())
        self.step_count += 1
        self.ended = self.step_count == 7
        return (
            self.obs(),
            1.0,
            self.ended,
            False,
            dict(
                original_success_ever=self.step_count >= 2,
                original_is_success=self.step_count >= 2,
                centered_success=self.ended,
                contacts=[self.step_count >= 1, self.step_count >= 2],
                reason="success" if self.ended else "",
            ),
        )

    def close(self):
        self.closed = True


@pytest.fixture
def models(monkeypatch):
    torch.set_num_threads(1)
    ToyEnv.instances = []
    monkeypatch.setattr(confirmation, "CenteredGraspEnv", ToyEnv)
    return tuple((FrozenModel(role) for role in ("manager", "approach", "pickup")))


def test_protocol_scene_support_hash_and_reserved_seed_exclusion():
    state = np.random.get_state()
    a = confirmation.confirmation_cases(740000000)
    assert a == confirmation.confirmation_cases(740000000)
    assert confirmation._digest(a) != confirmation._digest(
        confirmation.confirmation_cases(740000001)
    )
    assert len(a) == len({r["seed"] for r in a}) == 200
    assert all(
        (0.025 <= r["height"] < 0.14 and r["cup_offset"] == [0.0, 0.0] for r in a)
    )
    assert (
        min((r["height"] for r in a)) < 0.03 and max((r["height"] for r in a)) > 0.135
    )
    assert confirmation.confirmation_cases(799999800)[-1]["seed"] == 799999999
    assert np.array_equal(state[1], np.random.get_state()[1])
    for seed in (
        True,
        740000000.0,
        699999999,
        799999801,
        800000000,
        800000199,
        900000000,
    ):
        with pytest.raises(ValueError, match="dedicated"):
            confirmation.confirmation_cases(seed)
    for conditions in ((), ("black",), ("normal", "normal"), ("normal", "unknown")):
        with pytest.raises(ValueError):
            confirmation.validate_protocol(740000000, conditions)


def test_image_condition_preserves_both_complete_frames_and_proprio_without_aliasing():
    image = np.arange(6 * 96 * 96, dtype=np.uint8).reshape(6, 96, 96)
    reset = np.flip(image, axis=0).copy()
    proprio = np.arange(18, dtype=np.float32)
    observation = dict(image=image.copy(), proprio=proprio.copy())
    for condition, expected in [
        ("normal", image),
        ("frozen", reset),
        ("black", np.zeros_like(image)),
    ]:
        changed = confirmation.image_condition(observation, reset, condition)
        np.testing.assert_array_equal(changed["image"], expected)
        np.testing.assert_array_equal(changed["proprio"], proprio)
        changed["image"][:] = 7
        changed["proprio"][:] = 7
        np.testing.assert_array_equal(observation["image"], image)
        np.testing.assert_array_equal(observation["proprio"], proprio)


def test_all_networks_get_paired_ablation_inputs_and_original_success_never_stops_episode(
    models,
):
    manager, *experts = models
    before = confirmation.frozen_state(models)
    modes = [[m.training for m in p.policy.modules()] for p in models]
    flags = [[q.requires_grad for q in p.policy.parameters()] for p in models]
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    reports = []
    result = confirmation.evaluate_confirmation(
        manager,
        experts,
        seed=740000000,
        conditions=("normal", "frozen", "black"),
        progress=lambda p: reports.append((p["status"], len(p["rows"]))),
    )
    assert len(result["rows"]) == 600
    assert (
        result["scores"]["physical_actions"]
        == result["scores"]["physical_action_attempts"]
        == 4200
    )
    assert result["scores"]["manager_decisions"] == 1200
    assert result["scores"]["nominal_stage"]["passed"] is True
    assert all(
        (
            r["first_original_success_step"] == 2
            and r["physical_actions"] == 7
            and (r["post_original_success_actions"] == 5)
            and (r["expert_action_counts"] == [5, 2])
            and (r["commit_step"] == 5)
            for r in result["rows"]
        )
    )
    assert len(ToyEnv.instances) == 600 and all(
        (env.closed for env in ToyEnv.instances)
    )
    for i in range(200):
        assert (
            ToyEnv.instances[i].kwargs
            == ToyEnv.instances[i + 200].kwargs
            == ToyEnv.instances[i + 400].kwargs
        )
        assert (
            ToyEnv.instances[i].reset_random
            == ToyEnv.instances[i + 200].reset_random
            == ToyEnv.instances[i + 400].reset_random
        )
    for model, calls_per_condition in zip(models, (400, 1000, 400)):
        assert len(model.calls) == 3 * calls_per_condition
        normal, frozen, black = (
            model.calls[i * calls_per_condition : (i + 1) * calls_per_condition]
            for i in range(3)
        )
        for a, b, c in zip(normal, frozen, black):
            assert (
                a["pixel"] == a["step"] + 10 and b["pixel"] == 10 and (c["pixel"] == 0)
            )
            np.testing.assert_array_equal(a["proprio"], b["proprio"])
            np.testing.assert_array_equal(a["proprio"], c["proprio"])
        assert all(
            (
                c["deterministic"] and (not c["training"]) and (not c["grad"])
                for c in model.calls
            )
        )
    assert before == confirmation.frozen_state(models) == result["frozen_after"]
    assert modes == [[m.training for m in p.policy.modules()] for p in models]
    assert flags == [[q.requires_grad for q in p.policy.parameters()] for p in models]
    assert rng[0] == random.getstate() and np.array_equal(
        rng[1][1], np.random.get_state()[1]
    )
    assert torch.equal(rng[2], torch.get_rng_state())
    assert reports[0] == ("episode_running", 0) and reports[-1] == ("evaluating", 600)


def test_failure_reports_attempted_work_restores_modes_and_does_not_continue(
    models, monkeypatch
):
    manager, *experts = models
    monkeypatch.setattr(ToyEnv, "fail_step", 3)
    modes = [[m.training for m in p.policy.modules()] for p in models]
    before = confirmation.frozen_state(models)
    updates = []
    with pytest.raises(RuntimeError, match="simulator failure"):
        confirmation.evaluate_confirmation(
            manager,
            experts,
            seed=740000000,
            progress=lambda p: updates.append(copy.deepcopy(p)),
        )
    assert len(ToyEnv.instances) == 1 and ToyEnv.instances[0].closed
    row = updates[-1]["rows"][0]
    assert (
        updates[-1]["status"] == "failed"
        and row["physical_actions"] == 2
        and (row["physical_action_attempts"] == 3)
    )
    assert before == confirmation.frozen_state(models)
    assert modes == [[m.training for m in p.policy.modules()] for p in models]


@pytest.mark.parametrize("mutation", ["weight", "adam", "counter"])
def test_frozen_integrity_detects_each_kind_of_mutation_even_on_failure(
    models, monkeypatch, mutation
):
    manager, *experts = models

    def corrupt(model):
        if mutation == "weight":
            with torch.no_grad():
                model.policy[0].weight.add_(1.0)
        elif mutation == "adam":
            next(iter(model.policy.optimizer.state.values()))["exp_avg"].add_(1.0)
        else:
            model.num_timesteps += 1
        raise RuntimeError("Stop after intentional corruption")

    manager.hook = corrupt
    with pytest.raises(RuntimeError, match="policy/Adam tensors or training counters"):
        confirmation.evaluate_confirmation(manager, experts, seed=740000000)


def test_original_deadline_and_force_frequency_are_mandatory(models, monkeypatch):
    manager, *experts = models
    monkeypatch.setattr(ToyEnv, "control_dt", 0.04)
    with pytest.raises(RuntimeError, match="50Hz"):
        confirmation.evaluate_confirmation(manager, experts, seed=740000000)
    assert not manager.calls


def sources_on_disk(tmp_path):
    sources = []
    for role in ("manager", "approach", "pickup"):
        directory = tmp_path / role
        directory.mkdir()
        path = directory / "policy.zip"
        path.write_bytes(f"Test marker, not a policy ZIP: {role}".encode())
        sources.append(
            dict(
                run=role,
                checkpoint="policy.zip",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    for source in sources:
        metadata = dict(policy_sha256=source["sha256"], final_steps=123)
        if source["run"] == "manager":
            metadata.update(
                version=confirmation.MANAGER_VERSION,
                duration=dict(option_steps=5),
                experts=[
                    dict(
                        run=s["run"],
                        checkpoint=s["checkpoint"],
                        policy_sha256=s["sha256"],
                    )
                    for s in sources[1:]
                ],
            )
        (tmp_path / source["run"] / "summary.json").write_text(json.dumps(metadata))
    return (sources[0], sources[1:])


def test_source_lock_precedes_case_generation_complete_reuse_and_no_silent_partial_resume(
    tmp_path, models, monkeypatch
):
    manager_source, expert_sources = sources_on_disk(tmp_path)
    out = tmp_path / "confirmation"
    monkeypatch.setattr(confirmation.PPO, "load", lambda *a, **k: models[0])
    monkeypatch.setattr(
        confirmation,
        "load_grasp_policy",
        lambda p, **k: models[1] if p.parent.name == "approach" else models[2],
    )
    original = confirmation.confirmation_cases

    def cases(seed):
        assert (out / "source-lock.json").exists(), (
            "Sources must be locked before cases are generated"
        )
        return original(seed)

    monkeypatch.setattr(confirmation, "confirmation_cases", cases)
    result = confirmation.run_ordered_confirmation(
        tmp_path, out, manager_source, expert_sources, seed=740000000, device="cpu"
    )
    assert result["status"] == "complete" and result["new_physical_training_steps"] == 0
    assert set((p.name for p in out.iterdir())) == {
        "source-lock.json",
        "experiment.json",
        "progress.json",
        "summary.json",
    }
    monkeypatch.setattr(
        confirmation.PPO,
        "load",
        lambda *a, **k: pytest.fail("Completed output must not load a model again"),
    )
    assert (
        confirmation.run_ordered_confirmation(
            tmp_path, out, manager_source, expert_sources, seed=740000000, device="cpu"
        )
        == result
    )
    saved = json.loads((out / "summary.json").read_text())
    saved["frozen_before"] = saved["frozen_after"] = [{}, {}, {}]
    (out / "summary.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="integrity verification"):
        confirmation.run_ordered_confirmation(
            tmp_path, out, manager_source, expert_sources, seed=740000000, device="cpu"
        )
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "progress.json").write_text("{}")
    with pytest.raises(ValueError, match="Partial confirmation"):
        confirmation.run_ordered_confirmation(
            tmp_path, partial, manager_source, expert_sources
        )


def test_explicit_source_hash_and_manager_expert_binding_checked_before_cases(
    tmp_path, monkeypatch
):
    manager, experts = sources_on_disk(tmp_path)
    monkeypatch.setattr(
        confirmation,
        "confirmation_cases",
        lambda seed: pytest.fail("Cannot generate cases before source checks"),
    )
    with pytest.raises(ValueError, match="explicitly selected SHA256"):
        confirmation.run_ordered_confirmation(
            tmp_path, tmp_path / "wrong-hash", {**manager, "sha256": "a" * 64}, experts
        )
    assert (tmp_path / "wrong-hash/failure.json").exists()
    p = tmp_path / "manager/summary.json"
    metadata = json.loads(p.read_text())
    metadata["experts"][0]["policy_sha256"] = "f" * 64
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="frozen during manager training"):
        confirmation.run_ordered_confirmation(
            tmp_path, tmp_path / "wrong-pair", manager, experts
        )
    assert not (tmp_path / "wrong-pair/source-lock.json").exists()


def test_source_bytes_checked_after_evaluation(tmp_path, models, monkeypatch):
    manager, experts = sources_on_disk(tmp_path)
    monkeypatch.setattr(confirmation.PPO, "load", lambda *a, **k: models[0])
    monkeypatch.setattr(
        confirmation,
        "load_grasp_policy",
        lambda p, **k: models[1] if p.parent.name == "approach" else models[2],
    )

    def change_file(model):
        (tmp_path / "pickup/policy.zip").write_bytes(b"Changed test marker")
        model.hook = None

    models[0].hook = change_file
    with pytest.raises(RuntimeError, match="checkpoint bytes changed"):
        confirmation.run_ordered_confirmation(
            tmp_path, tmp_path / "changed", manager, experts, device="cpu"
        )
    assert (tmp_path / "changed/failure.json").exists() and (
        not (tmp_path / "changed/summary.json").exists()
    )


def test_loaded_counter_must_match_source_record_before_any_physical_episode(
    tmp_path, models, monkeypatch
):
    manager, experts = sources_on_disk(tmp_path)
    models[0].num_timesteps += 1
    monkeypatch.setattr(confirmation.PPO, "load", lambda *a, **k: models[0])
    monkeypatch.setattr(
        confirmation,
        "load_grasp_policy",
        lambda p, **k: models[1] if p.parent.name == "approach" else models[2],
    )
    with pytest.raises(ValueError, match="counter differs"):
        confirmation.run_ordered_confirmation(
            tmp_path, tmp_path / "wrong-counter", manager, experts, device="cpu"
        )
    assert not ToyEnv.instances and (tmp_path / "wrong-counter/failure.json").exists()


def test_centered_threshold_requires180_normal_camera_successes(models):
    result = confirmation.evaluate_confirmation(models[0], models[1:], seed=740000000)
    rows = result["rows"]
    for row in rows[:20]:
        row["centered_success"] = False
    assert confirmation.summarize(rows, ("normal",))["nominal_stage"]["passed"] is True
    rows[20]["centered_success"] = False
    assert confirmation.summarize(rows, ("normal",))["nominal_stage"]["passed"] is False
    assert (
        confirmation.summarize(rows[:-1], ("normal",))["nominal_stage"]["passed"]
        is None
    )


def test_launcher_pins_three_hashes_before_dispatch_and_rejects_reserved_seed(
    tmp_path, monkeypatch
):
    class Image:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    modal = SimpleNamespace(
        App=lambda *a: SimpleNamespace(
            function=lambda **k: lambda f: f,
            local_entrypoint=lambda: lambda f: f,
            app_id="test-only",
        ),
        Volume=SimpleNamespace(
            from_name=lambda *a, **k: SimpleNamespace(commit=lambda: None)
        ),
        Image=SimpleNamespace(debian_slim=lambda **k: Image()),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    path = Path(__file__).resolve().parents[1] / "scripts/ordered_confirmation_modal.py"
    spec = importlib.util.spec_from_file_location("test_confirmation_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    calls = []

    def spawn(**kwargs):
        assert (tmp_path / "runs" / kwargs["name"] / "selection.json").exists()
        calls.append(kwargs)
        return SimpleNamespace(object_id="no-remote-call")

    monkeypatch.setattr(module.run, "spawn", spawn, raising=False)
    args = dict(
        manager_run="manager",
        manager_checkpoint="step-5248.zip",
        manager_sha256="a" * 64,
        approach_run="approach",
        approach_checkpoint="policy.zip",
        approach_sha256="b" * 64,
        pickup_run="pickup",
        pickup_checkpoint="step-1944064.zip",
        pickup_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="dedicated"):
        module.main("reserved", seed=800000000, **args)
    assert not calls and (not (tmp_path / "runs/reserved").exists())
    normal_import = builtins.__import__

    def lightweight_only(name, *args, **kwargs):
        if name.startswith(("torch", "stable_baselines3", "robovision")):
            raise AssertionError(
                "Local entrypoint must not import training dependencies"
            )
        return normal_import(name, *args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", lightweight_only)
        module.main("valid", conditions="normal,frozen,black", **args)
    assert calls[0]["conditions"] == ("normal", "frozen", "black")
    assert calls[0]["manager_source"]["sha256"] == "a" * 64
    assert [s["sha256"] for s in calls[0]["expert_sources"]] == ["b" * 64, "c" * 64]
    with pytest.raises(ValueError, match="duplicate"):
        module.main("valid", **args)
