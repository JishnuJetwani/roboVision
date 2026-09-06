from argparse import Namespace
from contextlib import contextmanager
import json

import numpy as np
from stable_baselines3 import PPO
import torch

from robovision.curriculum import ReachCurriculum
from robovision import train_grasp, train_target


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right


def test_curriculum_resumes_with_the_same_recent_window():
    curriculum = ReachCurriculum()
    curriculum.observe([True] * 2048)
    assert curriculum.advance(2047) is None
    assert curriculum.advance(2048)["tolerance"] == .04
    curriculum.observe([False] * 1200 + [True] * 848)
    assert curriculum.advance(4096) is None
    restored = ReachCurriculum(json.loads(json.dumps(curriculum.state_dict())))
    for controller in (curriculum, restored):
        controller.observe([True] * 1200)
    assert curriculum.advance(5296) == restored.advance(5296)
    assert curriculum.state_dict() == restored.state_dict()


def test_grasp_training_resumes_weights_optimizer_and_sampling(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)
    features = rng.normal(size=(40, 12)).astype(np.float32)
    path = tmp_path / "demonstrations.npz"
    np.savez(path, x=features, y=np.tanh(features[:, :2]),
             episode_id=np.repeat(np.arange(10), 4), validation=np.repeat(np.arange(10) >= 8, 4))

    def arguments(name, resume=False):
        return Namespace(data=path, out=tmp_path / name, seed=201, updates=10,
                         batch_size=8, learning_rate=3e-4, max_seconds=60.,
                         checkpoint_seconds=120., resume=resume)

    train_grasp.train(arguments("whole"))
    stopped = [False]
    original_step = torch.optim.Adam.step
    calls = 0

    @contextmanager
    def stop_control():
        yield stopped

    def stop_after_three_updates(optimizer, *args, **kwargs):
        nonlocal calls
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        stopped[0] = calls == 3
        return result

    with monkeypatch.context() as patch:
        patch.setattr(train_grasp, "stop_after_update", stop_control)
        patch.setattr(torch.optim.Adam, "step", stop_after_three_updates)
        train_grasp.train(arguments("split"))
    partial = torch.load(tmp_path / "split/checkpoint.pt", weights_only=True)
    assert partial["updates"] == 3
    train_grasp.train(arguments("split", resume=True))
    whole = torch.load(tmp_path / "whole/checkpoint.pt", weights_only=True)
    split = torch.load(tmp_path / "split/checkpoint.pt", weights_only=True)
    for key in ("weights", "optimizer", "numpy_rng", "torch_rng", "updates"):
        assert_equal(whole[key], split[key])


def test_ppo_resumes_after_a_complete_update(tmp_path, monkeypatch):
    environment_class = train_target.PrecisionReachEnv
    monkeypatch.setattr(train_target, "PrecisionReachEnv",
                        lambda **kwargs: environment_class(render_images=False, **kwargs))

    def arguments(name, resume=False):
        return Namespace(method="curriculum", run_dir=tmp_path / name, seed=101,
                         steps=16, device="cpu", envs=1, rollout_steps=8, batch_size=8,
                         epochs=2, learning_rate=3e-4, threads=1, max_seconds=60.,
                         checkpoint_seconds=120., resume=resume)

    train_target.train(arguments("whole"))
    stopped = [False]
    original_train = PPO.train

    @contextmanager
    def stop_control():
        yield stopped

    def stop_after_first_update(model):
        original_train(model)
        stopped[0] = True

    with monkeypatch.context() as patch:
        patch.setattr(train_target, "stop_after_update", stop_control)
        patch.setattr(PPO, "train", stop_after_first_update)
        train_target.train(arguments("split"))
    assert json.loads((tmp_path / "split/metadata.json").read_text())["ppo_steps"] == 8
    train_target.train(arguments("split", resume=True))

    def latest(name):
        directory = tmp_path / name
        pointer = json.loads((directory / "latest.json").read_text())
        checkpoint = directory / pointer["checkpoint"]
        return (PPO.load(checkpoint / "policy.zip", device="cpu"),
                torch.load(checkpoint / "state.pt", weights_only=True),
                json.loads((checkpoint / "metadata.json").read_text()))

    whole, whole_state, whole_metadata = latest("whole")
    split, split_state, split_metadata = latest("split")
    assert_equal(whole.policy.state_dict(), split.policy.state_dict())
    assert_equal(whole.policy.optimizer.state_dict(), split.policy.optimizer.state_dict())
    assert_equal(whole_state, split_state)
    assert whole_metadata["curriculum_state"] == split_metadata["curriculum_state"]
