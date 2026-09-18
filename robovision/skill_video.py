"""Frozen single-episode skill video and exact action/physics trace; no training."""

import hashlib
from pathlib import Path
from .skill_hard_evaluation import basename, resolve_skill_checkpoint, sha256


def video_protocol(variant, stage, seed, mode):
    if variant not in (
        "near_pickup",
        "fine_near_pickup",
        "millimeter_near_pickup",
        "final_height",
    ):
        raise ValueError(
            "variant must be near_pickup, fine_near_pickup, millimeter_near_pickup or final_height"
        )
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if mode not in ("deterministic", "base"):
        raise ValueError("mode must be deterministic or base")
    if variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup"):
        from .near_pickup import (
            NearPickupEnv,
            FineNearPickupEnv,
            MillimeterNearPickupEnv,
        )

        HEIGHTS = {
            "near_pickup": NearPickupEnv,
            "fine_near_pickup": FineNearPickupEnv,
            "millimeter_near_pickup": MillimeterNearPickupEnv,
        }[variant].heights
        if type(stage) is not int or not 0 <= stage < len(HEIGHTS):
            raise ValueError("near_pickup requires an explicit valid stage")
        label = (
            f"Full pickup | start {HEIGHTS[stage] * 1000:.0f} mm above grasp | {mode}"
        )
    else:
        if stage is not None:
            raise ValueError("final_height has no curriculum stage override")
        label = f"Centered final-height pickup | {mode}"
    return dict(
        variant=variant,
        stage=stage,
        seed=seed,
        mode=mode,
        label=label,
        policy_sampling="deployment Gaussian; no training noise multipliers",
        success="Original full pickup and stable hold",
        control_hz=50,
    )


def make_video_env(variant, stage=None):
    if variant == "final_height":
        if stage is not None:
            raise ValueError("No final-height task stage override")
        from .joint_env import JointGraspEnv
        from .grasp_benchmark import ENV_CONFIG

        return JointGraspEnv(**ENV_CONFIG)
    if variant in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup"):
        from .near_pickup import (
            NearPickupEnv,
            FineNearPickupEnv,
            MillimeterNearPickupEnv,
        )

        cls = {
            "near_pickup": NearPickupEnv,
            "fine_near_pickup": FineNearPickupEnv,
            "millimeter_near_pickup": MillimeterNearPickupEnv,
        }[variant]
        return cls(subtask_stage=stage, observation="pixels")
    raise ValueError("Unknown video variant")


def record_skill_video(
    volume_root, name, source, checkpoint, *, variant, stage, seed, mode
):
    import json
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from .policy_loading import load_grasp_policy
    from .grasp_benchmark import preserved_rng

    spec = video_protocol(variant, stage, seed, mode)
    path, source_record, metadata, metadata_path = resolve_skill_checkpoint(
        volume_root, source, checkpoint
    )
    before = sha256(path)
    model = load_grasp_policy(path, device="cuda")

    def state_hash():
        digest = hashlib.sha256()
        for key, value in sorted(model.policy.state_dict().items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    state_before = state_hash()
    output = Path(volume_root) / basename(name)
    output.mkdir(exist_ok=False)
    video = output / "episode.mp4"
    rows = []
    env = None
    writer = None
    try:
        with preserved_rng():
            torch.manual_seed(seed)
            np.random.seed(seed)
            env = make_video_env(variant, stage)
            obs, info = env.reset(seed=seed)
            initial = dict(
                hand=env.grasp_position.tolist(),
                cup=env.cup_position.tolist(),
                joint_qpos=env.data.qpos[:6].tolist(),
                scene=dict(env.params),
            )
            writer = imageio.get_writer(video, fps=50)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except OSError:
                font = ImageFont.load_default(size=16)

            def frame(step, reason="", success=False):
                canvas = Image.fromarray(env.render())
                draw = ImageDraw.Draw(canvas)
                draw.rectangle((0, 0, canvas.width, 58), fill=(15, 20, 28))
                draw.text((10, 6), spec["label"], fill="white", font=font)
                text = f"t = {step * env.control_dt:.2f} s | seed {seed}"
                if reason:
                    text += f" | {('SUCCESS' if success else 'FAILURE')}: {reason}"
                draw.text(
                    (10, 31),
                    text,
                    fill=(150, 240, 170) if success else "white",
                    font=font,
                )
                writer.append_data(np.asarray(canvas))

            frame(0)
            while True:
                action, _ = model.predict(obs, deterministic=mode == "deterministic")
                obs, reward, done, truncated, info = env.step(action)
                rows.append(
                    dict(
                        step=env.step_count,
                        time_seconds=env.step_count * env.control_dt,
                        action=np.asarray(action).tolist(),
                        arm_torque_nm=info["arm_torque_nm"],
                        finger_force_n=info["finger_force_n"],
                        hand=env.grasp_position.tolist(),
                        cup=env.cup_position.tolist(),
                        joint_qpos=env.data.qpos[:6].tolist(),
                        contacts=list(info["contacts"]),
                        clearance=float(info["clearance"]),
                        upright=float(info["upright"]),
                        cup_speed=float(info["cup_speed"]),
                        stable_hold_steps=int(env._hold_steps),
                        required_hold_steps=int(env.hold_steps),
                        success=bool(info["is_success"]),
                        reason=info["reason"],
                        reward=float(reward),
                        terminated=bool(done),
                        truncated=bool(truncated),
                    )
                )
                frame(env.step_count, info["reason"], info["is_success"])
                if done or truncated:
                    break
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
    after = sha256(path)
    state_after = state_hash()
    if before != after or state_before != state_after:
        raise RuntimeError("Video recording changed frozen policy")
    root = Path(__file__).resolve().parents[1]
    files = [
        "assets/cup_arm.xml",
        "robovision/joint_env.py",
        "robovision/env.py",
        "robovision/grasp_reward.py",
        "robovision/near_pickup.py",
        "robovision/skill_video.py",
        "robovision/policy_loading.py",
    ]
    result = dict(
        protocol=spec,
        source_run=source,
        checkpoint=checkpoint,
        source_record=source_record,
        source_metadata_sha256=sha256(metadata_path),
        policy_sha256=before,
        policy_sha256_after=after,
        model_state_sha256_before=state_before,
        model_state_sha256_after=state_after,
        source_sha256={f: sha256(root / f) for f in files},
        initial=initial,
        rows=rows,
        success=rows[-1]["success"],
        reason=rows[-1]["reason"],
        steps=len(rows),
        video_path=str(video),
        video_sha256=sha256(video),
        model_downloaded=False,
    )
    (output / "trajectory.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
