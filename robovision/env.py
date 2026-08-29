"""MuJoCo cup lifting with finite-force arm and gripper actuators."""
from __future__ import annotations

from pathlib import Path
import sys
import os
from typing import Any

if sys.platform == 'darwin':
    os.environ.setdefault('MUJOCO_GL', 'glfw')
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ENV_VERSION = 'robovision-cup-v1'


class VisionCupEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}
    control_dt = .05
    action_delta = .008
    table_z = .25
    cup_height = .09
    home = np.array([.32, 0., .45])

    def __init__(self, seed: int = 0, render_images: bool = True,
                 max_steps: int = 200):
        super().__init__()
        self.render_images = bool(render_images)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        path = Path(__file__).resolve().parents[1] / 'assets' / 'cup_arm.xml'
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self._cup_bid = self._id(mujoco.mjtObj.mjOBJ_BODY, 'cup')
        self._cup_jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, 'cup_free')
        self._cup_qadr = int(self.model.jnt_qposadr[self._cup_jid])
        self._cup_vadr = int(self.model.jnt_dofadr[self._cup_jid])
        self._grasp_sid = self._id(mujoco.mjtObj.mjOBJ_SITE, 'grasp_center')
        self._pad_ids = [self._id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('left_pad', 'right_pad')]
        self._finger_geoms = [{self._id(mujoco.mjtObj.mjOBJ_GEOM, f'{side}_{part}') for part in ('pad', 'shaft')}
                              for side in ('left', 'right')]
        self._cup_geoms = np.flatnonzero(self.model.geom_bodyid == self._cup_bid)
        self._cup_geom_set = set(map(int, self._cup_geoms))
        self._base_inertia = self.model.body_inertia[self._cup_bid].copy()
        self._joint_low = self.model.jnt_range[:6, 0].copy()
        self._joint_high = self.model.jnt_range[:6, 1].copy()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, shape=(6, 96, 96), dtype=np.uint8),
            'proprio': spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
        })
        self._camera_renderer = None
        self._demo_renderer = None
        self._previous_frame = None
        self.last_action = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self._hold_steps = 0
        self._last_potential = 0.
        self.params: dict[str, float] = {}

    def _id(self, kind, name):
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(name)
        return value

    @staticmethod
    def inverse_kinematics(grasp_position: np.ndarray) -> np.ndarray:
        x, y, z = map(float, grasp_position)
        radial = float(np.hypot(x, y))
        vertical = z + .075 - .42
        cosine = (radial * radial + vertical * vertical - .30 ** 2 - .28 ** 2) / (2 * .30 * .28)
        if not -1. <= cosine <= 1.:
            raise ValueError(f'Unreachable grasp position: {grasp_position}')
        elbow = float(np.arccos(cosine))
        shoulder = float(np.arctan2(-vertical, radial) - np.arctan2(.28 * np.sin(elbow), .30 + .28 * np.cos(elbow)))
        return np.array([np.arctan2(y, x), shoulder, elbow, np.pi / 2 - shoulder - elbow])

    @property
    def grasp_position(self) -> np.ndarray:
        return self.data.site_xpos[self._grasp_sid].copy()

    @property
    def cup_position(self) -> np.ndarray:
        """Object position for rewards, demonstration collection, and evaluation."""
        return self.data.xpos[self._cup_bid].copy()

    def _contacts(self) -> tuple[bool, bool]:
        forces = [0., 0.]
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            for side, finger in enumerate(self._finger_geoms):
                if (a in finger and b in self._cup_geom_set) or (b in finger and a in self._cup_geom_set):
                    mujoco.mj_contactForce(self.model, self.data, i, force)
                    forces[side] += max(0., float(force[0]))
        return (forces[0] > .05, forces[1] > .05)

    def _clearance(self) -> float:
        bottoms = []
        for gid in self._cup_geoms:
            matrix = self.data.geom_xmat[gid].reshape(3, 3)
            size = self.model.geom_size[gid]
            if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                vertical_extent = size[0] * np.linalg.norm(matrix[2, :2]) + size[1] * abs(matrix[2, 2])
            else:
                vertical_extent = float(np.abs(matrix[2]) @ size)
            bottoms.append(float(self.data.geom_xpos[gid, 2] - vertical_extent))
        return min(bottoms) - self.table_z

    def _check_simulation(self):
        bad = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
               mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL]
        for warning in bad:
            if self.data.warning[warning].number:
                raise RuntimeError(f'MuJoCo numerical warning: {warning.name}; invalid simulation state')
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Non-finite MuJoCo state')

    def _potential(self, info):
        distance = float(np.linalg.norm(self.grasp_position - self.cup_position))
        return -3. * distance + .3 * float(all(info['contacts'])) + 10. * float(np.clip(info['clearance'], 0, .12))

    def _info(self, success=False, reason='') -> dict[str, Any]:
        contacts = self._contacts()
        return {
            'env_version': ENV_VERSION, 'is_success': bool(success),
            'reason': reason, 'clearance': self._clearance(), 'contacts': list(contacts),
            'upright': float(self.data.xmat[self._cup_bid].reshape(3, 3)[2, 2]),
            'cup_speed': float(np.linalg.norm(self.data.qvel[self._cup_vadr:self._cup_vadr + 3])),
            'step': self.step_count, **self.params,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        x = float(self.rng.uniform(.25, .39))
        y = float(self.rng.uniform(-.09, .09))
        mass = .08
        friction = 1.
        self.params = {'spawn_x': x, 'spawn_y': y, 'cup_mass': mass, 'grip_friction': friction}
        self.model.body_mass[self._cup_bid] = mass
        self.model.body_inertia[self._cup_bid] = self._base_inertia * (mass / .08)
        self.model.pair_friction[:, :2] = friction
        color = np.array([.1, .65, .72])
        self.model.geom_rgba[self._cup_geoms, :3] = color
        brightness = 1.
        self.model.light_diffuse[0] = .8 * brightness
        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:4] = self.inverse_kinematics(self.home)
        self.data.qpos[4:6] = .045
        self.data.qpos[self._cup_qadr:self._cup_qadr + 3] = [x, y, self.table_z + self.cup_height / 2 + .001]
        self.data.qpos[self._cup_qadr + 3:self._cup_qadr + 7] = [1., 0., 0., 0.]
        self.data.ctrl[:] = self.data.qpos[:6]
        mujoco.mj_forward(self.model, self.data)
        # Settle only with the physical simulator, using the same fixed robot home.
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self._check_simulation()
        self.step_count = 0
        self._hold_steps = 0
        self.last_action[:] = [0., 0., 0., 1.]
        self._previous_frame = None
        info = self._info()
        self._last_potential = self._potential(info)
        return self._observation(), info

    def _observation(self):
        frame = self.render_camera() if self.render_images else np.zeros((96, 96, 3), dtype=np.uint8)
        chw = np.transpose(frame, (2, 0, 1)).copy()
        previous = chw if self._previous_frame is None else self._previous_frame
        image = np.concatenate([previous, chw], axis=0)
        self._previous_frame = chw
        normalized_q = 2 * (self.data.qpos[:6] - self._joint_low) / (self._joint_high - self._joint_low) - 1
        normalized_v = self.data.qvel[:6] / np.array([2., 2., 2., 2., .1, .1])
        proprio = np.concatenate([normalized_q, normalized_v, self.last_action]).astype(np.float32)
        return {'image': image, 'proprio': proprio}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('action must contain four finite numbers')
        action = np.clip(action, -1, 1)
        desired = np.clip(self.grasp_position + self.action_delta * action[:3], [.22, -.12, .29], [.42, .12, .51])
        self.data.ctrl[:4] = self.inverse_kinematics(desired)
        self.data.ctrl[4:6] = .0225 * (float(action[3]) + 1.)
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
        self._check_simulation()
        self.last_action = action.copy()
        self.step_count += 1
        info = self._info()
        stable = (all(info['contacts']) and info['clearance'] >= .06
                  and info['upright'] >= np.cos(np.deg2rad(20)) and info['cup_speed'] < .10)
        self._hold_steps = self._hold_steps + 1 if stable else 0
        success = self._hold_steps >= 10
        cup = self.cup_position
        out = bool(cup[2] < .22 or cup[0] < .16 or cup[0] > .48 or abs(cup[1]) > .22)
        unstable = not bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        terminated = bool(success or out or unstable)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        reason = 'success' if success else 'unstable' if unstable else 'out_of_bounds' if out else 'timeout' if truncated else ''
        info.update(is_success=bool(success), reason=reason)
        # Reward progress, rather than time spent near the cup.
        potential = self._potential(info)
        reward = potential - self._last_potential - .01 - .002 * float(action[:3] @ action[:3])
        reward += 25. if success else -5. if out or unstable else 0.
        self._last_potential = potential
        return self._observation(), float(reward), terminated, truncated, info

    def render_camera(self):
        if self._camera_renderer is None:
            self._camera_renderer = mujoco.Renderer(self.model, height=96, width=96)
        self._camera_renderer.update_scene(self.data, camera='policy')
        return self._camera_renderer.render().copy()

    def render(self):
        if self._demo_renderer is None:
            self._demo_renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._demo_renderer.update_scene(self.data, camera='demo')
        return self._demo_renderer.render().copy()

    def close(self):
        for key in ('_camera_renderer', '_demo_renderer'):
            renderer = getattr(self, key, None)
            if renderer is not None:
                renderer.close()
                setattr(self, key, None)
