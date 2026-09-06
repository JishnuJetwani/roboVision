"""Single-decision reaching task used to train the visual target policy."""
from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import spaces

from robovision.env import VisionCupEnv

VERSION = "robovision-reach-v1"
HARD_TOLERANCE = .006


class PrecisionReachEnv(VisionCupEnv):
    def __init__(self, seed=0, tolerance=HARD_TOLERANCE, render_images=True,
                 actuator_steps=400):
        super().__init__(seed=seed, render_images=render_images, max_steps=1)
        self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        self.tolerance = float(tolerance)
        self.actuator_steps = int(actuator_steps)
        self.task_rng = np.random.default_rng(seed)
        self.scenario_seed = 0

    def set_tolerance(self, tolerance):
        self.tolerance = float(tolerance)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.task_rng = np.random.default_rng(seed)
            self.scenario_seed = int(seed)
        else:
            self.scenario_seed = int(self.task_rng.integers(100000, 2**31 - 1))
        observation, info = super().reset(seed=self.scenario_seed)
        return observation, {**info, "task_version": VERSION, "tolerance": self.tolerance}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite target coordinates")
        action = np.clip(action, -1., 1.)
        desired = np.array([.32 + .075 * action[0], .095 * action[1], .45])
        self.data.ctrl[:4] = self.inverse_kinematics(desired)
        self.data.ctrl[4:6] = .045
        mujoco.mj_step(self.model, self.data, nstep=self.actuator_steps)
        self._check_simulation()
        self.last_action[:] = [action[0], action[1], 0., 1.]
        self.step_count = 1
        alignment = float(np.linalg.norm(self.grasp_position[:2] - self.cup_position[:2]))
        height_error = abs(float(self.grasp_position[2]) - .45)
        success = bool(alignment <= self.tolerance and height_error <= .006)
        info = self._info(success=success, reason="success" if success else "alignment_error")
        info.update(task_version=VERSION, tolerance=self.tolerance,
                    alignment_error=alignment, height_error=height_error,
                    tracking_error=float(np.linalg.norm(self.grasp_position - desired)),
                    scenario_seed=self.scenario_seed)
        return self._observation(), float(success), True, False, info
