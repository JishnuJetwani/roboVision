"""Full-episode joint-space grasping. Object state is used only for rewards/reset."""

from __future__ import annotations
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from .env import VisionCupEnv
from .grasp_reward import GraspReward

VERSION = "robovision-torque-v3"


class JointGraspEnv(VisionCupEnv):
    """Four direct joint torques and one symmetric finger force at 50 Hz.

    There is no position/velocity servo or gravity feed-forward. IK only sets reset poses.
    Stages increase starting height above a fixed cup. Physics and appearance are fixed.
    """

    control_dt = 0.02
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    torque_limits = np.array([40.0, 55.0, 45.0, 20.0])
    finger_force_limit = 10.0
    control_spec = dict(
        mode="direct_torque",
        frequency_hz=50,
        arm_torque_limits_nm=[40.0, 55.0, 45.0, 20.0],
        finger_force_limit_n=10.0,
        gravity_compensation=False,
    )

    def __init__(
        self, *, stage=0, observation="pixels", gamma=0.995, max_steps=500, **kwargs
    ):
        if observation not in ("pixels", "state"):
            raise ValueError("observation must be pixels or state")
        self.observation_mode = observation
        self.gamma = float(gamma)
        if not 0 < self.gamma <= 1 or max_steps < 1:
            raise ValueError("Require 0 < gamma <= 1 and a positive episode deadline")
        self.reward_function = GraspReward()
        self.set_stage(stage)
        super().__init__(max_steps=max_steps, **kwargs)
        self.model.body_gravcomp[:] = 0
        self.model.actuator_dyntype[:] = mujoco.mjtDyn.mjDYN_NONE
        self.model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
        self.model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
        self.model.actuator_gainprm[:] = 0
        self.model.actuator_gainprm[:, 0] = 1
        self.model.actuator_biasprm[:] = 0
        self.model.actuator_gear[:] = 0
        self.model.actuator_gear[:, 0] = 1
        limits = np.r_[
            self.torque_limits, self.finger_force_limit, self.finger_force_limit
        ]
        self.model.actuator_ctrlrange[:] = np.column_stack([-limits, limits])
        self.model.actuator_forcerange[:] = self.model.actuator_ctrlrange
        self.model.actuator_ctrllimited[:] = True
        self.model.actuator_forcelimited[:] = True
        self.physics_steps = int(round(self.control_dt / self.model.opt.timestep))
        self.hold_steps = int(round(0.5 / self.control_dt))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        proprio = spaces.Box(-np.inf, np.inf, shape=(18,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {"image": spaces.Box(0, 255, (6, 96, 96), np.uint8), "proprio": proprio}
            if observation == "pixels"
            else {
                "state": spaces.Box(-np.inf, np.inf, (13,), np.float32),
                "proprio": proprio,
            }
        )
        self.last_action = np.zeros(5, np.float32)
        self._robot_bodies = set(range(1, self._cup_bid))
        self._table_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "table")

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state

    def set_stage(self, stage):
        if isinstance(stage, bool) or int(stage) != stage or (not 0 <= stage <= 3):
            raise ValueError("stage must be 0, 1, 2, or 3")
        self.stage = int(stage)

    def _observation(self):
        q = (
            2
            * (self.data.qpos[:6] - self._joint_low)
            / (self._joint_high - self._joint_low)
            - 1
        )
        v = self.data.qvel[:6] / np.array([2.0, 2.0, 2.0, 2.0, 0.1, 0.1])
        previous_action = np.zeros(5, np.float32)
        previous_action[: len(self.last_action)] = self.last_action
        remaining = max(0.0, 1 - self.step_count / self.max_steps)
        proprio = np.r_[q, v, previous_action, remaining].astype(np.float32)
        if self.observation_mode == "state":
            state = np.r_[
                self.cup_position,
                self.data.qpos[self._cup_qadr + 3 : self._cup_qadr + 7],
                self.data.qvel[self._cup_vadr : self._cup_vadr + 6],
            ].astype(np.float32)
            return {"state": state, "proprio": proprio}
        frame = (
            self.render_camera()
            if self.render_images
            else np.zeros((96, 96, 3), np.uint8)
        )
        frame = frame.transpose(2, 0, 1).copy()
        previous = frame if self._previous_frame is None else self._previous_frame
        self._previous_frame = frame
        return {"image": np.concatenate([previous, frame]), "proprio": proprio}

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_stage = self.stage
        x, y = (0.32, 0.0)
        mass, friction = (0.08, 1.0)
        self.model.body_mass[self._cup_bid] = mass
        self.model.body_inertia[self._cup_bid] = self._base_inertia * mass / 0.08
        self.model.pair_friction[:, :2] = friction
        self.params = dict(
            spawn_x=float(x), spawn_y=float(y), cup_mass=mass, grip_friction=friction
        )
        self.model.geom_rgba[self._cup_geoms, :3] = [0.1, 0.65, 0.72]
        self.model.geom_rgba[self._table_id, :3] = [0.88, 0.9, 0.92]
        self.model.light_diffuse[0] = 0.8
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._cup_qadr : self._cup_qadr + 3] = [
            x,
            y,
            self.table_z + self.cup_height / 2 + 0.001,
        ]
        self.data.qpos[self._cup_qadr + 3 : self._cup_qadr + 7] = [1, 0, 0, 0]
        target = np.array([x, y, (0.34, 0.38, 0.415, 0.45)[self.episode_stage]])
        self.data.qpos[:4] = self.inverse_kinematics(target)
        self.data.qpos[4:6] = 0.045
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.last_action = np.zeros(5, np.float32)
        self.step_count = 0
        self._hold_steps = 0
        self._reward_totals = {}
        self._previous_frame = None
        mujoco.mj_forward(self.model, self.data)
        self._peak_clearance = max(0.0, self._clearance())
        return (
            self._observation(),
            {**self._info(), "env_version": VERSION, "stage": self.episode_stage},
        )

    def _grasp_scores(self, info):
        error = self.grasp_position - self.cup_position
        reach = float(np.exp(-np.linalg.norm(error) / 0.08))
        finger_axis = self.data.site_xmat[self._grasp_sid].reshape(3, 3)[:, 0]
        downward = float(np.clip(-finger_axis[2], 0, 1)) ** 4
        centered = float(
            np.exp(-np.linalg.norm(error[:2]) / 0.025 - abs(error[2]) / 0.035)
        )
        bilateral = float(all(info["contacts"]))
        opening = float(np.clip(np.mean(self.data.qpos[4:6]) / 0.025, 0, 1))
        alignment = centered * downward * max(opening, bilateral)
        grip = bilateral * float(np.clip(info["upright"], 0, 1)) ** 4
        lift = grip * float(np.clip(info["clearance"] / 0.06, 0, 1))
        lift *= float(np.clip(1 - info["cup_speed"] / 0.5, 0, 1))
        return dict(reach=reach, alignment=alignment, grip=grip, lift=lift)

    def _table_collision(self):
        for contact in self.data.contact:
            a, b = (int(contact.geom1), int(contact.geom2))
            other = b if a == self._table_id else a if b == self._table_id else None
            if (
                other is not None
                and int(self.model.geom_bodyid[other]) in self._robot_bodies
            ):
                return True
        return False

    def _early_failure(self, info):
        """Optional training-task termination; final-height task has none."""
        return ""

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (5,) or not np.isfinite(action).all():
            raise ValueError(
                "Expected four finite normalized torques and one finger-force command"
            )
        # Scale normalized actions into actuator forces; no position servo runs here.
        action = np.clip(action, -1, 1)
        previous_action = self.last_action.copy()
        self.data.ctrl[:4] = self.torque_limits * action[:4]
        self.data.ctrl[4:6] = self.finger_force_limit * action[4]
        mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        self._check_simulation()
        self.last_action = action.copy()
        self.step_count += 1
        info = self._info()
        stable = (
            all(info["contacts"])
            and info["clearance"] >= 0.06
            and (info["upright"] >= np.cos(np.deg2rad(20)))
            and (info["cup_speed"] < 0.1)
        )
        self._hold_steps = self._hold_steps + 1 if stable else 0
        cup = self.cup_position
        out = bool(
            cup[2] < 0.22 or cup[0] < 0.16 or cup[0] > 0.48 or (abs(cup[1]) > 0.22)
        )
        collision = self._table_collision()
        early_failure = self._early_failure(info)
        success = bool(
            self._hold_steps >= self.hold_steps
            and (not out)
            and (not collision)
            and (not early_failure)
        )
        deadline = self.step_count >= self.max_steps
        terminated = bool(success or out or collision or early_failure or deadline)
        truncated = False
        self._peak_clearance = max(self._peak_clearance, info["clearance"])
        reason = (
            "table_collision"
            if collision
            else "out_of_bounds"
            if out
            else early_failure
            if early_failure
            else "success"
            if success
            else "timeout"
            if deadline
            else ""
        )
        if (
            reason == "timeout"
            and self._peak_clearance >= 0.06
            and (info["clearance"] < 0.02)
        ):
            reason = "dropped"
        scores = self._grasp_scores(info)
        components = self.reward_function.components(
            scores,
            action,
            previous_action,
            dt=self.control_dt,
            gamma=self.gamma,
            remaining_steps=max(0, self.max_steps - self.step_count),
            success=success,
            failure=terminated and (not success),
        )
        reward = sum(components.values())
        for name, value in components.items():
            self._reward_totals[name] = self._reward_totals.get(name, 0.0) + value
        info.update(
            env_version=VERSION,
            is_success=success,
            reason=reason,
            stage=self.episode_stage,
            table_collision=collision,
            peak_clearance=self._peak_clearance,
            reward_components=components,
            grasp_scores=scores,
            arm_torque_nm=self.data.ctrl[:4].tolist(),
            finger_force_n=float(self.data.ctrl[4]),
        )
        if terminated:
            info["episode_reward_components"] = self._reward_totals.copy()
        return (self._observation(), float(reward), terminated, truncated, info)
