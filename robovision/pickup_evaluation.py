"""Frozen curriculum gates for complete pickup at explicit start heights."""

import torch
import numpy as np
from .grasp_band import GraspBandEnv


def evaluate(model, heights, *, noisy_episodes=5, seed_start=160000):
    rows = []
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        for height in heights:
            env = GraspBandEnv(
                grasp_band_bonus=0,
                fixed_height=height,
                shaping=False,
                replay_fraction=0,
                observation="pixels",
                max_steps=500,
            )
            try:
                for deterministic in (True, False):
                    for i in range(1 if deterministic else noisy_episodes):
                        seed = seed_start + i
                        torch.manual_seed(seed)
                        obs, _ = env.reset(seed=seed)
                        initial_z = float(env.grasp_position[2])
                        first_contact = None
                        first_dz = None
                        contact_heights = []
                        best_band = 0.0
                        while True:
                            action, _ = model.predict(obs, deterministic=deterministic)
                            obs, _, done, truncated, info = env.step(action)
                            best_band = max(best_band, info["grasp_band_quality"])
                            if all(info["contacts"]):
                                contact_heights.extend(
                                    (
                                        h
                                        for h in info["grasp_contact_heights"]
                                        if h is not None
                                    )
                                )
                            if env.step_count == 5:
                                first_dz = float(env.grasp_position[2] - initial_z)
                            if any(info["contacts"]) and first_contact is None:
                                first_contact = env.step_count * 0.02
                            if done or truncated:
                                break
                        rows.append(
                            dict(
                                height=height,
                                deterministic=deterministic,
                                seed=seed,
                                success=bool(info["is_success"]),
                                reason=info["reason"],
                                first_contact=first_contact,
                                first_100ms_dz=first_dz,
                                best_grasp_band_quality=best_band,
                                mean_contact_height=float(np.mean(contact_heights))
                                if contact_heights
                                else None,
                                bilateral_steps=info["bilateral_contact_steps"],
                                peak_clearance=info["peak_clearance"],
                                duration=env.step_count * 0.02,
                            )
                        )
            finally:
                env.close()
    return rows
