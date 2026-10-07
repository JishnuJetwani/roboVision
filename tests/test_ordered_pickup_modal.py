"""CLI warm-source forwarding without importing Modal or launching any jobs."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    class Image:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    fake_modal = SimpleNamespace(
        App=lambda *args: SimpleNamespace(
            function=lambda **kwargs: lambda function: function,
            local_entrypoint=lambda: lambda function: function,
            app_id="app-test-no-remote-call",
        ),
        Volume=SimpleNamespace(
            from_name=lambda *args, **kwargs: SimpleNamespace(commit=lambda: None)
        ),
        Image=SimpleNamespace(debian_slim=lambda **kwargs: Image()),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    path = Path(__file__).resolve().parents[1] / "scripts/ordered_pickup_modal.py"
    spec = importlib.util.spec_from_file_location("test_ordered_modal_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    spawned = []

    def spawn(**kwargs):
        spawned.append(kwargs)
        return SimpleNamespace(object_id="fc-test-no-remote-call")

    monkeypatch.setattr(module.run, "spawn", spawn, raising=False)
    return (module, spawned)


def arguments():
    return dict(
        approach_source="approach",
        approach_checkpoint="policy.zip",
        pickup_source="pickup",
        pickup_checkpoint="step-123.zip",
        arrival_run="trusted-arrivals",
        arrival_sha256="a" * 64,
        target_physics_steps=196608,
        seed=470000,
        option_steps=5,
        training_reward_scale=0.01,
    )


def test_cli_fresh_and_warm_training_forward_only_explicit_manager_source(
    launcher, tmp_path
):
    module, spawned = launcher
    module.main("fresh", **arguments())
    module.main(
        "warm", manager_run="parent", manager_checkpoint="step-5248.zip", **arguments()
    )
    assert len(spawned) == 2
    assert spawned[0]["manager_source"] is None
    assert spawned[1]["manager_source"] == dict(
        run="parent", checkpoint="step-5248.zip"
    )
    assert spawned[1]["mode"] == "train" and spawned[1]["training_reward_scale"] == 0.01
    assert spawned[1]["arrival_source"] == dict(run="trusted-arrivals", sha256="a" * 64)
    assert spawned[0]["reset_recipe"] == spawned[1]["reset_recipe"] == "arrivals-v1"
    launch = json.loads((tmp_path / "runs/warm/launch.json").read_text())
    assert launch["manager_source"] == spawned[1]["manager_source"]
    assert launch["call"] == "fc-test-no-remote-call"
    assert list((tmp_path / "runs/warm").iterdir()) == [
        tmp_path / "runs/warm/launch.json"
    ]


@pytest.mark.parametrize(
    "changes",
    [
        dict(manager_run="parent"),
        dict(manager_checkpoint="policy.zip"),
        dict(manager_run="parent", manager_checkpoint="weights"),
        dict(manager_run="warm", manager_checkpoint="policy.zip"),
        dict(manager_run="../parent", manager_checkpoint="policy.zip"),
        dict(
            manager_run="parent",
            manager_checkpoint="policy.zip",
            arrival_run="",
            arrival_sha256="",
        ),
    ],
)
def test_cli_rejects_ambiguous_or_unsafe_warm_forms_before_spawn(
    launcher, tmp_path, changes
):
    module, spawned = launcher
    with pytest.raises(ValueError):
        module.main("warm", **dict(arguments(), **changes))
    assert not spawned and (not (tmp_path / "runs/warm").exists())


def test_frozen_evaluation_manager_flags_still_work(launcher):
    module, spawned = launcher
    kwargs = dict(
        arguments(), mode="evaluate", training_reward_scale=1.0, profile="fresh40"
    )
    module.main(
        "evaluation", manager_run="parent", manager_checkpoint="policy.zip", **kwargs
    )
    assert spawned[0]["mode"] == "evaluate" and spawned[0]["profile"] == "fresh40"
    assert spawned[0]["manager_source"] == dict(run="parent", checkpoint="policy.zip")
    with pytest.raises(ValueError, match="requires one"):
        module.main("missing-manager", **kwargs)


@pytest.mark.parametrize("reset_recipe", ["arrivals-v1", "full-start-v2"])
def test_remote_entry_passes_warm_source_to_training_function_without_launching(
    launcher, tmp_path, monkeypatch, reset_recipe
):
    module, spawned = launcher
    import robovision.train_ordered_pickup as trainer
    import robovision.torch_precision as precision

    monkeypatch.setattr(precision, "configure_policy_precision", lambda: None)
    project = tmp_path / "project"
    for filename in ("robovision/toy.py", "assets/cup_arm.xml"):
        path = project / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Local test-only source identity")
    actual_path = Path

    def redirected_path(value):
        if value == "/root/project":
            return project
        if value == "/runs":
            return tmp_path / "runs"
        return actual_path(value)

    monkeypatch.setattr(module, "Path", redirected_path)
    recorded = []

    def train(volume_root, out, experts, **kwargs):
        recorded.append((experts, kwargs))
        return dict(status="complete", new_steps=128, new_physical_steps=500)

    monkeypatch.setattr(trainer, "train_ordered_pickup", train)
    manager_source = dict(run="parent", checkpoint="step-5248.zip")
    module.run(
        "remote-function-test",
        [
            dict(run="approach", checkpoint="policy.zip"),
            dict(run="pickup", checkpoint="policy.zip"),
        ],
        arrival_source=dict(run="trusted-arrivals", sha256="a" * 64),
        manager_source=manager_source,
        training_reward_scale=0.01,
        reset_recipe=reset_recipe,
    )
    assert not spawned and recorded[0][1]["manager_source"] == manager_source
    assert recorded[0][1]["training_reward_scale"] == 0.01
    config = json.loads(
        (tmp_path / "runs/remote-function-test/launch-config.json").read_text()
    )
    assert config["manager_source"] == manager_source
    assert config["reset_recipe"] == recorded[0][1]["reset_recipe"] == reset_recipe


def test_cli_explicit_full_start_comparison_forwarding(launcher, tmp_path):
    module, spawned = launcher
    module.main(
        "full-start",
        manager_run="parent",
        manager_checkpoint="policy.zip",
        reset_recipe="full-start-v2",
        **arguments(),
    )
    assert len(spawned) == 1 and spawned[0]["reset_recipe"] == "full-start-v2"
    assert spawned[0]["arrival_source"] == dict(run="trusted-arrivals", sha256="a" * 64)
    config = json.loads((tmp_path / "runs/full-start/launch.json").read_text())
    assert config["reset_recipe"] == "full-start-v2"


@pytest.mark.parametrize(
    "changes",
    [
        dict(reset_recipe="unknown"),
        dict(reset_recipe="full-start-v2"),
        dict(
            reset_recipe="full-start-v2",
            manager_run="parent",
            manager_checkpoint="policy.zip",
            mode="evaluate",
            training_reward_scale=1.0,
        ),
        dict(
            reset_recipe="full-start-v2",
            manager_run="parent",
            manager_checkpoint="policy.zip",
            arrival_run="",
            arrival_sha256="",
        ),
    ],
)
def test_cli_rejects_ambiguous_reset_comparisons_before_spawn(
    launcher, tmp_path, changes
):
    module, spawned = launcher
    with pytest.raises(ValueError):
        module.main("bad-reset", **dict(arguments(), **changes))
    assert not spawned and (not (tmp_path / "runs/bad-reset").exists())
