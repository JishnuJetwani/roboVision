"""Exact simulator snapshots and explicit physical handoffs between PPO skills.

Snapshots are trusted, process-local Python objects (also pickleable for a local
arrival-state pool), not JSON reports. They contain simulator state, not learned
policy weights. Exact restore reinstates the same phase and its complete Python
history. Handoff keeps physics, the episode clock, force history and both image
frames, but starts the receiving phase's own reward/stability bookkeeping.

Only unwrapped JointGraspEnv descendants and fileless SB3 Monitor wrappers are
supported. Unknown wrappers are rejected instead of silently losing history.
Monitor wall-clock timings naturally advance with real time; simulator state,
observations, rewards and counters remain deterministic.
"""

import copy
import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3.common.monitor import Monitor
from .hierarchical_skills import GraspSkillEnv, LiftSkillEnv
from .joint_env import JointGraspEnv

VERSION = "exact-skill-state-v1"
INTEGRATION_STATE = mujoco.mjtState.mjSTATE_INTEGRATION
BASE_RESOURCES = {"model", "data", "_camera_renderer", "_demo_renderer"}
WRAPPER_RESOURCES = {"env", "results_writer"}
PHYSICAL_EPISODE_FIELDS = (
    "step_count",
    "last_action",
    "_previous_frame",
    "_hold_steps",
    "_reward_totals",
    "_peak_clearance",
    "params",
    "episode_stage",
    "episode_reset_height",
    "episode_cup_offset",
    "episode_height_band_index",
)


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _unwrap(env):
    wrappers = []
    while isinstance(env, gym.Wrapper):
        if type(env) is not Monitor:
            raise ValueError(f"Unsupported stateful wrapper: {_class_name(env)}")
        if env.results_writer is not None:
            raise ValueError("Snapshot restore does not rewind a Monitor output file")
        wrappers.append(env)
        env = env.env
    if not isinstance(env, JointGraspEnv):
        raise ValueError(
            "Skill snapshots require the original force-control environment family"
        )
    return (env, wrappers)


def _integration(data, model):
    state = np.empty(mujoco.mj_stateSize(model, INTEGRATION_STATE))
    mujoco.mj_getState(model, data, state, INTEGRATION_STATE)
    return state


def _python_state(instance, excluded):
    return {
        name: copy.deepcopy(value)
        for name, value in vars(instance).items()
        if name not in excluded
    }


def _restore_python_state(instance, state, excluded):
    for name in set(vars(instance)) - set(state) - excluded:
        delattr(instance, name)
    for name, value in state.items():
        setattr(instance, name, copy.deepcopy(value))


def _validate_observation(env, observation):
    if not isinstance(observation, dict) or set(observation) != set(
        env.observation_space.spaces
    ):
        raise ValueError("Snapshot requires the current complete actor observation")
    for key, space in env.observation_space.spaces.items():
        if np.shape(observation[key]) != space.shape:
            raise ValueError(f"Snapshot observation shape mismatch: {key}")
    q = (
        2.0 * (env.data.qpos[:6] - env._joint_low) / (env._joint_high - env._joint_low)
        - 1.0
    )
    velocity = env.data.qvel[:6] / np.array([2.0, 2.0, 2.0, 2.0, 0.1, 0.1])
    proprio = np.r_[
        q, velocity, env.last_action, max(0.0, 1.0 - env.step_count / env.max_steps)
    ].astype(np.float32)
    if not np.array_equal(observation["proprio"], proprio):
        raise ValueError(
            "Snapshot observation is stale: proprioception or clock differs"
        )
    if "image" in observation and (
        env._previous_frame is None
        or not np.array_equal(observation["image"][3:], env._previous_frame)
    ):
        raise ValueError(
            "Snapshot observation does not match the latest framebuffer history"
        )


def capture_skill_state(env, observation, *, info=None):
    """Copy all simulator/model state and supported environment/wrapper history.

    Pass the step's ``info`` when the snapshot will be used as a successful
    handoff. No simulator forward call or rendering occurs during capture.
    """
    base, wrappers = _unwrap(env)
    _validate_observation(base, observation)
    if info is not None and (
        info.get("step") != base.step_count
        or info.get("phase", getattr(base, "phase", None))
        != getattr(base, "phase", None)
    ):
        raise ValueError(
            "Snapshot info must describe the current environment step and phase"
        )
    return dict(
        version=VERSION,
        mujoco_version=mujoco.__version__,
        environment_class=_class_name(base),
        model=copy.copy(base.model),
        data=copy.copy(base.data),
        integration_state=_integration(base.data, base.model),
        python_state=_python_state(base, BASE_RESOURCES),
        wrappers=[
            dict(
                wrapper_class=_class_name(wrapper),
                state=_python_state(wrapper, WRAPPER_RESOURCES),
            )
            for wrapper in wrappers
        ],
        observation=copy.deepcopy(observation),
        info=copy.deepcopy(info),
        phase=getattr(base, "phase", None),
        phase_success=None if info is None else bool(info.get("phase_success", False)),
        step_count=base.step_count,
    )


def _validate_snapshot(base, snapshot):
    if (
        snapshot.get("version") != VERSION
        or snapshot.get("mujoco_version") != mujoco.__version__
    ):
        raise ValueError("Incompatible skill snapshot or MuJoCo version")
    if not isinstance(snapshot.get("model"), mujoco.MjModel) or not isinstance(
        snapshot.get("data"), mujoco.MjData
    ):
        raise ValueError("Snapshot must contain full MuJoCo model and data copies")
    model = snapshot["model"]
    signature = lambda value: (
        value.nq,
        value.nv,
        value.nu,
        value.nbody,
        value.ngeom,
        value.nsite,
        bytes(value.names),
    )
    if signature(base.model) != signature(model):
        raise ValueError(
            "Snapshot and receiving environment have different model topology"
        )
    if not np.array_equal(
        _integration(snapshot["data"], model), snapshot["integration_state"]
    ):
        raise ValueError("Snapshot integration state was modified after capture")
    if snapshot["step_count"] != snapshot["python_state"]["step_count"]:
        raise ValueError("Snapshot clock metadata disagrees with episode history")
    if (
        base.max_steps != snapshot["python_state"]["max_steps"]
        or base.control_dt != 0.02
    ):
        raise ValueError(
            "Handoff must retain the original deadline and control frequency"
        )


def _restore_physics(base, snapshot):
    base.close()
    base.model = copy.copy(snapshot["model"])
    base.data = copy.copy(snapshot["data"])


def restore_skill_state(env, snapshot):
    """Restore an exact same-phase continuation, including both RGB frames."""
    base, wrappers = _unwrap(env)
    _validate_snapshot(base, snapshot)
    if _class_name(base) != snapshot["environment_class"]:
        raise ValueError(
            "Exact restore requires the same environment class; use handoff for a new phase"
        )
    if [_class_name(wrapper) for wrapper in wrappers] != [
        item["wrapper_class"] for item in snapshot["wrappers"]
    ]:
        raise ValueError("Exact restore requires the same wrapper stack")
    _restore_physics(base, snapshot)
    _restore_python_state(base, snapshot["python_state"], BASE_RESOURCES)
    for wrapper, saved in zip(wrappers, snapshot["wrappers"]):
        _restore_python_state(wrapper, saved["state"], WRAPPER_RESOURCES)
    observation = copy.deepcopy(snapshot["observation"])
    _validate_observation(base, observation)
    info = (
        copy.deepcopy(snapshot["info"])
        if snapshot["info"] is not None
        else base._info()
    )
    return (observation, info)


def restore_skill_handoff(env, snapshot, *, require_success=True):
    """Transfer an actual arrival into the next reset-initialized specialist.

    Call the receiving environment's ordinary reset first. Its RNG and future
    reset distribution remain its own. The source's physical state, elapsed
    clock, last action, full-pickup history and image history are transferred.
    The source's phase reward and success counters are deliberately not reused.
    """
    base, wrappers = _unwrap(env)
    _validate_snapshot(base, snapshot)
    phases = {"approach": GraspSkillEnv, "grasp": LiftSkillEnv}
    if snapshot["phase"] not in phases or not isinstance(
        base, phases[snapshot["phase"]]
    ):
        raise ValueError("Supported handoffs are approach to grasp and grasp to lift")
    if (
        base.observation_mode != snapshot["python_state"]["observation_mode"]
        or base.render_images != snapshot["python_state"]["render_images"]
    ):
        raise ValueError(
            "Handoff must retain the source observation and rendering mode"
        )
    if require_success and snapshot["phase_success"] is not True:
        raise ValueError(
            "A successful handoff requires phase_success evidence from the captured step"
        )
    if (
        not hasattr(base, "episode_skill_stage")
        or base.step_count != 0
        or any((wrapper.needs_reset or wrapper.rewards for wrapper in wrappers))
    ):
        raise ValueError("Reset the receiving environment before restoring a handoff")
    if snapshot["step_count"] >= base.max_steps:
        raise ValueError("No original episode time remains for the receiving phase")
    _restore_physics(base, snapshot)
    for name in PHYSICAL_EPISODE_FIELDS:
        if name in snapshot["python_state"]:
            setattr(base, name, copy.deepcopy(snapshot["python_state"][name]))
    base._full_pickup_success_ever = bool(
        snapshot["python_state"].get("_full_pickup_success_ever", False)
    )
    base.episode_skill_stage = base.skill_stage
    if isinstance(base, GraspSkillEnv):
        base._lesson = base.lessons[base.skill_stage]
        base._phase_stable_steps = 0
        base._phase_reward_totals = {}
        metrics = base._phase_metrics(base._info())
        base._reset_arrival_velocity = np.asarray(metrics["relative_velocity_m_s"])
        info = base._skill_info(
            {
                **base._generalization_info(base._info()),
                "is_success": False,
                "full_pickup_success": False,
            },
            metrics,
        )
    else:
        info = base._lift_info(base._info())
    info.update(
        handoff_from_phase=snapshot["phase"],
        handoff_step=base.step_count,
        handoff_remaining_steps=base.max_steps - base.step_count,
        handoff_clock_reset=False,
        handoff_velocity_reset=False,
        handoff_source_success=snapshot["phase_success"],
    )
    observation = copy.deepcopy(snapshot["observation"])
    _validate_observation(base, observation)
    return (observation, info)
