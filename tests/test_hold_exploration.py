"""A noise schedule must preserve means, rollouts and resume state."""

from argparse import Namespace
import json
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from robovision.reverse_curriculum import ReverseGraspEnv, ReverseGraspCurriculum
from robovision.train_joint import anneal_hold_exploration, train
from robovision.evaluate_joint import resolve_checkpoint


def test_noise_scale_changes_only_variance_once_after_promotion():
    env = ReverseGraspEnv(observation="state", render_images=False)
    try:
        model = PPO(
            "MultiInputPolicy", env, n_steps=8, batch_size=8, device="cpu", seed=5
        )
        obs, _ = env.reset(seed=1)
        mean, _ = model.predict(obs, deterministic=True)
        weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
        metadata = {}
        curriculum = ReverseGraspCurriculum()
        assert not anneal_hold_exploration(model, curriculum, metadata, 0.25)
        curriculum.level = 1
        assert anneal_hold_exploration(model, curriculum, metadata, 0.25)
        np.testing.assert_allclose(
            model.policy.log_std.detach().exp(), weights["log_std"].exp() * 0.25
        )
        after, _ = model.predict(obs, deterministic=True)
        np.testing.assert_array_equal(mean, after)
        for key, value in model.policy.state_dict().items():
            if key != "log_std":
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        persisted = json.loads(json.dumps(metadata))
        curriculum.level = 5
        assert not anneal_hold_exploration(model, curriculum, persisted, 0.25)
        np.testing.assert_allclose(
            model.policy.log_std.detach().exp(), weights["log_std"].exp() * 0.25
        )
    finally:
        env.close()


def test_training_resume_does_not_apply_noise_reduction_twice(tmp_path):
    args = Namespace(
        run_dir=tmp_path / "run",
        observation="state",
        steps=8,
        envs=1,
        rollout_steps=8,
        batch_size=8,
        epochs=1,
        learning_rate=3e-05,
        gamma=0.995,
        seed=301,
        stage=0,
        curriculum=True,
        curriculum_level=1,
        curriculum_replay=0.0,
        max_episode_steps=8,
        device="cpu",
        threads=1,
        max_seconds=60.0,
        checkpoint_seconds=120.0,
        resume=False,
        hold_exploration_scale=0.25,
    )
    train(args)
    first = json.loads(
        resolve_checkpoint(args.run_dir).with_name("metadata.json").read_text()
    )
    assert first["hold_exploration_event"]["steps"] == 8
    args.resume = True
    args.steps = 16
    train(args)
    last = json.loads(
        resolve_checkpoint(args.run_dir).with_name("metadata.json").read_text()
    )
    assert last["hold_exploration_event"] == first["hold_exploration_event"]
    model = PPO.load(resolve_checkpoint(args.run_dir), device="cpu")
    np.testing.assert_allclose(
        model.policy.log_std.detach().exp(),
        first["hold_exploration_event"]["std_normalized_after"],
        rtol=0.01,
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, 1.01, float("nan")])
def test_invalid_noise_schedule_rejected(scale):
    with pytest.raises(ValueError, match="Hold exploration scale"):
        train(Namespace(curriculum=True, stage=0, hold_exploration_scale=scale))
