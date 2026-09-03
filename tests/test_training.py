from argparse import Namespace
from contextlib import contextmanager

import numpy as np
import torch

from robovision import train_grasp


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
