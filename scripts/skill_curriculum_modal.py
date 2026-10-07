"""Bounded bridge experiment; returns JSON metrics only, leaves weights on Modal."""

import json
import math
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("robovision-skill-curriculum")
volume = modal.Volume.from_name("robovision-training", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libegl1", "libgl1", "libgles2", "libopengl0", "libglib2.0-0")
    .pip_install(
        "mujoco==3.3.7",
        "gymnasium==1.2.1",
        "stable-baselines3==2.7.0",
        "torch==2.8.0",
        "numpy==2.2.6",
        "matplotlib==3.10.7",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "pillow==11.3.0",
    )
    .env(
        {
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "PYTHONPATH": "/root/project",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    .add_local_dir(
        ROOT / "robovision", "/root/project/robovision", ignore=["__pycache__/**"]
    )
    .add_local_file(ROOT / "assets/cup_arm.xml", "/root/project/assets/cup_arm.xml")
)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=7200,
    scaledown_window=2,
    volumes={"/runs": volume},
)
def run_experiment(
    name,
    source,
    variant,
    budget,
    noise_scale,
    early_close_failure=False,
    gate_interval=60.0,
    frontier_reward_scale=1.0,
    strict_descent=False,
    role_normalization=False,
    source_checkpoint=None,
    arm_noise_scale=1.0,
    learning_rate_scale=1.0,
    bootstrap_height_jitter=0.0,
    decoupled=False,
    critic_learning_rate=0.0003,
    context_exploration=False,
    context_std_scales=None,
    context_role_normalization=None,
    allow_context_reconfigure=False,
    role_gradient_balance=None,
    gradient_balance_max_weight=None,
    pickup_potential_scale=None,
    pickup_curriculum=None,
    pickup_initial_level=12,
    pickup_curriculum_kind=None,
    pickup_initial_opening_index=0,
    finger_exploration_max=None,
    actor_update_scope=None,
    wide_finger_residual=None,
    pickup_freeze_opening=None,
    pickup_attempt_limit_steps=None,
    frontier_fixed_height=None,
    stop_after_passed_gate=False,
    start_stage=None,
    arm_exploration_ramp=False,
    promote_passed_baseline=True,
    fixed_evaluation_interval=0.0,
    frontier_attempt_limit_steps=None,
    target_new_steps=0,
    gate_step_interval=0,
    arm_residual_mode=None,
):
    from robovision.train_skill_curriculum import train

    return train(
        Path("/runs") / source,
        Path("/runs") / name,
        variant,
        budget,
        volume.commit,
        noise_scale=noise_scale,
        early_close_failure=early_close_failure,
        gate_interval=gate_interval,
        frontier_reward_scale=frontier_reward_scale,
        strict_descent=strict_descent,
        role_normalization=role_normalization,
        source_checkpoint=source_checkpoint,
        arm_noise_scale=arm_noise_scale,
        learning_rate_scale=learning_rate_scale,
        bootstrap_height_jitter=bootstrap_height_jitter,
        decoupled=decoupled,
        critic_learning_rate=critic_learning_rate,
        context_exploration=context_exploration,
        context_std_scales=context_std_scales,
        context_role_normalization=context_role_normalization,
        allow_context_reconfigure=allow_context_reconfigure,
        role_gradient_balance=role_gradient_balance,
        gradient_balance_max_weight=gradient_balance_max_weight,
        pickup_potential_scale=pickup_potential_scale,
        pickup_curriculum=pickup_curriculum,
        pickup_initial_level=pickup_initial_level,
        pickup_curriculum_kind=pickup_curriculum_kind,
        pickup_initial_opening_index=pickup_initial_opening_index,
        finger_exploration_max=finger_exploration_max,
        actor_update_scope=actor_update_scope,
        wide_finger_residual=wide_finger_residual,
        pickup_freeze_opening=pickup_freeze_opening,
        pickup_attempt_limit_steps=pickup_attempt_limit_steps,
        frontier_fixed_height=frontier_fixed_height,
        stop_after_passed_gate=stop_after_passed_gate,
        stage=start_stage,
        arm_exploration_ramp=arm_exploration_ramp,
        promote_passed_baseline=promote_passed_baseline,
        fixed_evaluation_interval=fixed_evaluation_interval,
        frontier_attempt_limit_steps=frontier_attempt_limit_steps,
        target_new_steps=target_new_steps,
        gate_step_interval=gate_step_interval,
        arm_residual_mode=arm_residual_mode,
    )


@app.local_entrypoint()
def main(
    name: str,
    variant: str = "subtask",
    source: str = "modal-approach-bridge-20261004-v1",
    budget: float = 300.0,
    noise_scale: float = 1.0,
    early_close_failure: bool = False,
    gate_interval: float = 60.0,
    frontier_reward_scale: float = 1.0,
    strict_descent: bool = False,
    role_normalization: bool = False,
    source_checkpoint: str = "",
    arm_noise_scale: float = 1.0,
    learning_rate_scale: float = 1.0,
    bootstrap_height_jitter: float = 0.0,
    decoupled: bool = False,
    critic_learning_rate: float = 0.0003,
    context_exploration: bool = False,
    context_std_scales: str = "",
    context_role_normalization: str = "inherit",
    allow_context_reconfigure: bool = False,
    role_gradient_balance: str = "inherit",
    gradient_balance_max_weight: float = 0.0,
    pickup_potential_scale: str = "",
    pickup_curriculum: str = "inherit",
    pickup_initial_level: int = 12,
    pickup_curriculum_kind: str = "",
    pickup_initial_opening_index: int = 0,
    finger_exploration_max: str = "",
    actor_update_scope: str = "inherit",
    wide_finger_residual: str = "inherit",
    pickup_freeze_opening: str = "inherit",
    pickup_attempt_limit_steps: int = -1,
    frontier_fixed_height: str = "inherit",
    stop_after_passed_gate: bool = False,
    start_stage: int = -1,
    arm_exploration_ramp: bool = False,
    promote_passed_baseline: bool = True,
    fixed_evaluation_interval: float = 0.0,
    frontier_attempt_limit_steps: int = -1,
    target_new_steps: int = 0,
    gate_step_interval: int = 0,
    arm_residual_mode: str = "inherit",
):
    if arm_residual_mode not in ("inherit", "wide", "always"):
        raise ValueError("Invalid arm residual mode")
    if not -1 <= frontier_attempt_limit_steps <= 500 or (
        frontier_attempt_limit_steps > 0
        and variant not in ("near_pickup", "fine_near_pickup", "millimeter_near_pickup")
    ):
        raise ValueError("Invalid frontier attempt limit")
    for value in (target_new_steps, gate_step_interval):
        if value < 0 or value % 2048:
            raise ValueError("Interaction budgets/gates must be multiples of2048")
    if gate_step_interval and (not target_new_steps):
        raise ValueError("Step gates require interaction budget")
    if fixed_evaluation_interval and variant not in (
        "fine_near_pickup",
        "millimeter_near_pickup",
    ):
        raise ValueError("Fixed height evaluation requires fine/millimeter near pickup")
    if not math.isfinite(fixed_evaluation_interval) or fixed_evaluation_interval < 0:
        raise ValueError("Invalid fixed evaluation interval")
    if start_stage < -1:
        raise ValueError("start-stage must be -1 (resume/default) or nonnegative")
    if frontier_fixed_height not in ("inherit", "off"):
        height = float(frontier_fixed_height)
        if not math.isfinite(height) or not 0 <= height <= 0.14:
            raise ValueError(
                "frontier-fixed-height must be inherit, off, or meters in [0, .14]"
            )
        if variant not in (
            "open_bootstrap",
            "hover_bootstrap",
            "fine_hover_bootstrap",
            "fine_descent_bootstrap",
        ):
            raise ValueError(
                "frontier-fixed-height requires an open or hover bootstrap variant"
            )
    if pickup_attempt_limit_steps < -1:
        raise ValueError(
            "pickup-attempt-limit-steps must be -1 (inherit), 0 (disable), or positive"
        )
    if pickup_freeze_opening not in ("inherit", "on", "off"):
        raise ValueError("pickup-freeze-opening must be inherit, on, or off")
    if wide_finger_residual not in ("inherit", "on", "off"):
        raise ValueError("wide-finger-residual must be inherit, on, or off")
    if actor_update_scope not in (
        "inherit",
        "all",
        "finger_head",
        "arm_head",
        "wide_finger",
        "arm_residual",
        "grasp_residual",
    ):
        raise ValueError(
            "actor-update-scope must be inherit, all, finger_head, arm_head, or wide_finger"
        )
    if context_role_normalization not in ("inherit", "on", "off"):
        raise ValueError("context-role-normalization must be inherit, on, or off")
    if role_gradient_balance not in ("inherit", "on", "off"):
        raise ValueError("role-gradient-balance must be inherit, on, or off")
    if pickup_curriculum not in ("inherit", "on", "off"):
        raise ValueError("pickup-curriculum must be inherit, on, or off")
    adaptive = None if pickup_curriculum == "inherit" else pickup_curriculum == "on"
    balance = (
        None if role_gradient_balance == "inherit" else role_gradient_balance == "on"
    )
    normalization = (
        None
        if context_role_normalization == "inherit"
        else context_role_normalization == "on"
    )
    if Path(name).name != name:
        raise ValueError("Run name must be a basename")
    output = ROOT / "runs" / name
    if (output / "launch.json").exists():
        raise ValueError(
            "Run already has a launch handle; inspect it before launching another"
        )
    output.mkdir(parents=True, exist_ok=True)
    call = run_experiment.spawn(
        name,
        source,
        variant,
        budget,
        noise_scale,
        early_close_failure,
        gate_interval,
        frontier_reward_scale,
        strict_descent,
        role_normalization,
        source_checkpoint or None,
        arm_noise_scale,
        learning_rate_scale,
        bootstrap_height_jitter,
        decoupled,
        critic_learning_rate,
        context_exploration,
        [float(v) for v in context_std_scales.split(",")]
        if context_std_scales
        else None,
        normalization,
        allow_context_reconfigure,
        balance,
        gradient_balance_max_weight or None,
        float(pickup_potential_scale) if pickup_potential_scale else None,
        adaptive,
        pickup_initial_level,
        pickup_curriculum_kind or None,
        pickup_initial_opening_index,
        [float(v) for v in finger_exploration_max.split(",")]
        if finger_exploration_max
        else None,
        None if actor_update_scope == "inherit" else actor_update_scope,
        None if wide_finger_residual == "inherit" else wide_finger_residual == "on",
        None if pickup_freeze_opening == "inherit" else pickup_freeze_opening == "on",
        None if pickup_attempt_limit_steps == -1 else pickup_attempt_limit_steps,
        frontier_fixed_height,
        stop_after_passed_gate,
        None if start_stage == -1 else start_stage,
        arm_exploration_ramp,
        promote_passed_baseline,
        fixed_evaluation_interval,
        None if frontier_attempt_limit_steps == -1 else frontier_attempt_limit_steps,
        target_new_steps,
        gate_step_interval,
        None if arm_residual_mode == "inherit" else arm_residual_mode,
    )
    launch = dict(
        arm_residual_mode=arm_residual_mode,
        frontier_attempt_limit_steps=frontier_attempt_limit_steps,
        target_new_steps=target_new_steps,
        gate_step_interval=gate_step_interval,
        arm_exploration_ramp=arm_exploration_ramp,
        promote_passed_baseline=promote_passed_baseline,
        fixed_evaluation_interval=fixed_evaluation_interval,
        requested_start_stage=None if start_stage == -1 else start_stage,
        stop_after_passed_gate=stop_after_passed_gate,
        frontier_fixed_height=frontier_fixed_height,
        name=name,
        source=source,
        source_checkpoint=source_checkpoint or "policy.zip",
        variant=variant,
        budget=budget,
        actor_update_scope=actor_update_scope,
        wide_finger_residual=wide_finger_residual,
        pickup_freeze_opening=pickup_freeze_opening,
        pickup_attempt_limit_steps=pickup_attempt_limit_steps,
        call=call.object_id,
        app=app.app_id,
    )
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps(launch), flush=True)
