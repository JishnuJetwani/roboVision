from argparse import Namespace
import json
import mujoco
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from robovision.cnn import GraspCNN
from robovision.joint_env import JointGraspEnv, VERSION
from robovision.env import VisionCupEnv
from robovision.train_joint import GraspCurriculum, train
from robovision.evaluate_joint import evaluate, resolve_checkpoint


@pytest.mark.parametrize("mode", ["pixels", "state"])
def test_joint_environment_contract(mode):
    env = JointGraspEnv(observation=mode, render_images=False, max_steps=3)
    try:
        check_env(env, warn=True, skip_render_check=True)
        obs, _ = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        for index in range(3):
            obs, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 1])
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert terminated or truncated
    finally:
        env.close()


def test_actions_are_direct_torques_and_forces_without_feedback(monkeypatch):
    env = JointGraspEnv(stage=2, render_images=False)
    try:
        env.reset(seed=8)
        np.testing.assert_array_equal(env.data.ctrl, 0)

        def forbidden(*args):
            raise AssertionError("IK must not run during policy control")

        monkeypatch.setattr(env, "inverse_kinematics", forbidden)
        env.step([0.5, -0.5, 0.25, -0.25, -1])
        expected = [20, -27.5, 11.25, -5, -10, -10]
        np.testing.assert_allclose(env.data.ctrl, expected)
        np.testing.assert_allclose(env.data.actuator_force, expected)
        assert np.all(env.model.actuator_forcelimited)
        assert not env.model.body_gravcomp.any()
        env.data.qpos[:4] += 0.01
        env.data.qvel[:6] = [0.2, -0.2, 0.1, -0.1, 0.03, -0.03]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env.data.actuator_force, expected)
        env.step([2, -2, 2, -2, 2])
        np.testing.assert_allclose(env.data.actuator_force, [40, -55, 45, -20, 10, 10])
        env.step(np.zeros(5))
        np.testing.assert_array_equal(env.data.actuator_force, 0)
        for bad in ([1, 2], [0, 0, 0, 0, np.nan]):
            with pytest.raises(ValueError):
                env.step(bad)
    finally:
        env.close()


def test_zero_torque_arm_falls_under_gravity():
    positions = []
    for gravity in (-9.81, 0.0):
        env = JointGraspEnv(stage=2, observation="state", render_images=False)
        try:
            env.model.opt.gravity[2] = gravity
            env.reset(seed=8)
            initial = env.grasp_position.copy()
            for _ in range(10):
                _, _, done, truncated, _ = env.step(np.zeros(5))
                np.testing.assert_array_equal(env.data.actuator_force, 0)
                if done or truncated:
                    break
            positions.append(env.grasp_position - initial)
        finally:
            env.close()
    assert positions[0][2] < -0.005
    np.testing.assert_allclose(positions[1], 0, atol=1e-06)


def test_finger_command_controls_force_not_opening_and_legacy_is_unchanged():
    env = JointGraspEnv(stage=2, render_images=False)
    legacy = VisionCupEnv(render_images=False)
    try:
        changes = []
        for force in (-0.2, 0.2):
            env.reset(seed=8)
            env.data.qpos[4:6] = 0.025
            mujoco.mj_forward(env.model, env.data)
            env.step([0, 0, 0, 0, force])
            changes.append(env.data.qpos[4:6] - 0.025)
        assert np.all(changes[0] < 0)
        assert np.all(changes[1] > 0)
        assert np.all(legacy.model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE)
        assert np.any(legacy.model.body_gravcomp)
    finally:
        env.close()
        legacy.close()


def test_pixel_policy_has_no_object_state_and_frames_update():
    env = JointGraspEnv(stage=2)
    try:
        first, _ = env.reset(seed=9)
        env.data.qpos[env._cup_qadr : env._cup_qadr + 2] = [0.38, 0.08]
        mujoco.mj_forward(env.model, env.data)
        second = env._observation()
        assert set(second) == {"image", "proprio"}
        np.testing.assert_array_equal(first["proprio"], second["proprio"])
        np.testing.assert_array_equal(first["image"][3:], second["image"][:3])
        assert not np.array_equal(first["image"][3:], second["image"][3:])
        extractor = GraspCNN(env.observation_space)
        pixels = torch.tensor(
            second["image"][None], dtype=torch.float32, requires_grad=True
        )
        features = extractor(
            dict(image=pixels / 255, proprio=torch.tensor(second["proprio"][None]))
        )
        features.square().sum().backward()
        assert pixels.grad[:, :3].abs().sum() > 0
        assert pixels.grad[:, 3:].abs().sum() > 0
        assert extractor.cnn[0].weight.grad.abs().sum() > 0
    finally:
        env.close()


def test_fixed_scene_and_height_changes_are_reset_only():
    env = JointGraspEnv(stage=0, render_images=False)
    try:
        env.reset(seed=12)
        pose = env.data.qpos.copy()
        color = env.model.geom_rgba[env._cup_geoms].copy()
        env.reset(seed=12)
        np.testing.assert_array_equal(pose, env.data.qpos)
        np.testing.assert_array_equal(color, env.model.geom_rgba[env._cup_geoms])
        env.set_stage(3)
        assert env.episode_stage == 0
        env.reset(seed=13)
        assert env.episode_stage == 3
        assert env.params["cup_mass"] == 0.08
        np.testing.assert_allclose(
            env.cup_position[:2],
            [env.params["spawn_x"], env.params["spawn_y"]],
            atol=0.001,
        )
        np.testing.assert_array_equal(color, env.model.geom_rgba[env._cup_geoms])
    finally:
        env.close()


def test_curriculum_requires_completed_successful_episodes():
    curriculum = GraspCurriculum()
    for i in range(255):
        assert not curriculum.observe({"stage": 0, "is_success": True}, i)
    assert curriculum.observe({"stage": 0, "is_success": True}, 256)
    assert curriculum.stage == 1
    assert not curriculum.observe({"stage": 0, "is_success": True}, 257)
    assert curriculum.episodes == 0
    restored = GraspCurriculum(state=json.loads(json.dumps(curriculum.state_dict())))
    assert restored.state_dict() == curriculum.state_dict()


def test_hold_requires_contact_and_toss_has_no_lift_shaping(monkeypatch):
    env = JointGraspEnv(render_images=False)
    try:
        env.reset(seed=10)
        stable = dict(contacts=[True, True], clearance=0.07, upright=1.0, cup_speed=0.0)
        state = stable.copy()
        monkeypatch.setattr(env, "_info", lambda: state.copy())
        monkeypatch.setattr(env, "_table_collision", lambda: False)
        monkeypatch.setattr(mujoco, "mj_step", lambda *a, **kw: None)
        for _ in range(env.hold_steps - 1):
            assert not env.step([0, 0, 0, 0, 1])[2]
        state["contacts"] = [False, False]
        assert not env.step([0, 0, 0, 0, 1])[2]
        assert env._grasp_scores(state)["lift"] == 0
        state["clearance"] = 0
        assert env._grasp_scores(state)["lift"] == 0
        state.update(stable)
        for _ in range(env.hold_steps - 1):
            assert not env.step([0, 0, 0, 0, 1])[2]
        _, _, terminated, truncated, info = env.step([0, 0, 0, 0, 1])
        assert terminated and (not truncated) and info["is_success"]
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["dummy", "subproc"])
def test_cnn_ppo_training_resume_evaluation_and_video(tmp_path, backend):
    args = Namespace(
        run_dir=tmp_path / "run",
        observation="pixels",
        steps=16,
        envs=1,
        rollout_steps=8,
        batch_size=8,
        epochs=1,
        learning_rate=0.0003,
        gamma=0.99,
        seed=301,
        stage=0,
        curriculum=True,
        max_episode_steps=8,
        device="cpu",
        threads=1,
        max_seconds=60.0,
        vec_env=backend,
        checkpoint_seconds=120.0,
        resume=False,
    )
    saved_paths = []
    train(args, on_checkpoint=saved_paths.append)
    assert len(saved_paths) >= 2
    assert all(((path / "policy.zip").exists() for path in saved_paths))
    start = PPO.load(saved_paths[0] / "policy.zip", device="cpu")
    expected_std = np.array([1 / 40, 2 / 55, 1.5 / 45, 0.6 / 20, 1 / 10])
    np.testing.assert_allclose(
        start.policy.log_std.detach().exp().numpy(), expected_std, rtol=1e-06
    )
    assert start.ent_coef == 0.0
    initial_path = resolve_checkpoint(args.run_dir)
    initial = PPO.load(initial_path, device="cpu")
    weights = initial.policy.features_extractor.cnn[0].weight.detach().clone()
    assert initial.num_timesteps == 16
    optimizer_rows = [
        json.loads(line)
        for line in (args.run_dir / "optimizer.jsonl").read_text().splitlines()
    ]
    assert optimizer_rows[-1]["steps"] == 16
    assert "train/approx_kl" in optimizer_rows[-1]
    args.steps = 24
    args.resume = True
    train(args)
    path = resolve_checkpoint(args.run_dir)
    restored = PPO.load(path, device="cpu")
    assert restored.num_timesteps == 24
    assert not torch.equal(weights, restored.policy.features_extractor.cnn[0].weight)
    for ablation in ("normal", "black", "frozen"):
        options = Namespace(
            model=args.run_dir,
            episodes=1,
            seed_start=91000,
            stage=2,
            ablation=ablation,
            device="cpu",
            out=tmp_path / f"{ablation}.json",
            video=tmp_path / "demo.mp4" if ablation == "normal" else None,
        )
        result = evaluate(options)
        assert result["episodes"] == 1
        assert result["rows"][0]["steps"] <= 8
    assert (tmp_path / "demo.mp4").stat().st_size > 1000
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["curriculum"]["kind"] == "reverse-grasp-v2"
    episode_rows = [
        json.loads(line)
        for line in (args.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert metadata["curriculum"]["episodes"] == len(episode_rows) >= 3
    assert metadata["reset_curriculum"]["final_evaluation_stage"] == 3
    episode_rows = [
        json.loads(line)
        for line in (args.run_dir / "episodes.jsonl").read_text().splitlines()
    ]
    assert all(
        (row["curriculum_level"] == 0 and row["stage"] is None for row in episode_rows)
    )
    metadata["version"] = "robovision-joint-v1"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incompatible environment/reward version"):
        evaluate(options)
    with pytest.raises(ValueError, match="Incompatible environment/reward version"):
        train(args)
    metadata["version"] = VERSION
    metadata_path.write_text(json.dumps(metadata))
    args.gamma = 0.95
    with pytest.raises(ValueError, match="matching configuration"):
        train(args)
