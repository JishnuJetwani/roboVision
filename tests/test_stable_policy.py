import numpy as np
import torch
from stable_baselines3 import PPO
from robovision.cnn import NormalizedGraspCNN
from robovision.joint_env import JointGraspEnv


def test_critic_gradient_does_not_modify_actor_features():
    env = JointGraspEnv(render_images=False)
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=8,
            batch_size=8,
            policy_kwargs=dict(
                features_extractor_class=NormalizedGraspCNN,
                share_features_extractor=False,
            ),
            device="cpu",
        )
        obs, _ = env.reset(seed=43)
        obs["image"][:] = 128
        tensors, _ = model.policy.obs_to_tensor(obs)
        model.policy.predict_values(tensors).square().mean().backward()
        assert all(
            (p.grad is None for p in model.policy.pi_features_extractor.parameters())
        )
        assert any(
            (
                p.grad is not None and torch.any(p.grad != 0)
                for p in model.policy.vf_features_extractor.parameters()
            )
        )
        with torch.no_grad():
            features = model.policy.extract_features(tensors)[0]
            assert torch.isfinite(features).all()
            first = model.policy.mlp_extractor.policy_net[:2](features)
            assert (first.abs() > 0.99).float().mean() < 0.1
    finally:
        env.close()
