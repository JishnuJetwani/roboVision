import pytest
from robovision.skill_video import video_protocol, make_video_env
from robovision.near_pickup import NearPickupEnv
from robovision.joint_env import JointGraspEnv


def test_explicit_protocol_prevents_task_or_mode_ambiguity():
    p = video_protocol("near_pickup", 0, 190000, "deterministic")
    assert "10 mm" in p["label"] and p["control_hz"] == 50
    assert video_protocol("final_height", None, 190000, "base")["stage"] is None
    for args in [
        ("final_height", 0, 1, "base"),
        ("near_pickup", None, 1, "base"),
        ("near_pickup", 51, 1, "base"),
        ("near_pickup", 0, -1, "base"),
        ("near_pickup", 0, 1, "scheduled"),
    ]:
        with pytest.raises(ValueError):
            video_protocol(*args)


def test_environment_selection_is_original_task_not_teacher():
    hard = make_video_env("final_height")
    near = make_video_env("near_pickup", 10)
    try:
        assert type(hard) is JointGraspEnv and hard.stage == 3
        assert (
            hard.max_steps == 500
            and hard.gamma == 0.995
            and (hard.observation_mode == "pixels")
        )
        assert isinstance(near, NearPickupEnv) and near.subtask_stage == 10
        assert hard.action_space.shape == near.action_space.shape == (5,)
        assert not hard.model.body_gravcomp.any()
    finally:
        hard.close()
        near.close()


def test_fine_video_uses_actual_height_not_old_stage_index():
    from robovision.near_pickup import FineNearPickupEnv

    assert "56 mm" in video_protocol("fine_near_pickup", 46, 190000, "base")["label"]
    assert "80 mm" in video_protocol("near_pickup", 46, 190000, "base")["label"]
    env = make_video_env("fine_near_pickup", 46)
    try:
        _, info = env.reset(seed=190000)
        assert type(env) is FineNearPickupEnv
        assert info["near_pickup_height"] == 0.056
        assert env.action_space.shape == (5,) and (not env.model.body_gravcomp.any())
    finally:
        env.close()
    with pytest.raises(ValueError):
        video_protocol("fine_near_pickup", 79, 190000, "base")


def test_millimeter_video_has_explicit_81mm_stage_and_unchanged_fine_stage():
    from robovision.near_pickup import MillimeterNearPickupEnv

    assert (
        "81 mm" in video_protocol("millimeter_near_pickup", 71, 190000, "base")["label"]
    )
    assert "85 mm" in video_protocol("fine_near_pickup", 71, 190000, "base")["label"]
    assert (
        "140 mm"
        in video_protocol("millimeter_near_pickup", 130, 190000, "base")["label"]
    )
    for stage in (None, -1, 131, True):
        with pytest.raises(ValueError):
            video_protocol("millimeter_near_pickup", stage, 190000, "base")
    env = make_video_env("millimeter_near_pickup", 71)
    env.render_images = False
    try:
        _, info = env.reset(seed=190000)
        assert type(env) is MillimeterNearPickupEnv
        assert info["near_pickup_height"] == 0.081
        assert env.frontier_attempt_limit is None and env.max_steps == 500
        assert env.action_space.shape == (5,) and (not env.model.body_gravcomp.any())
    finally:
        env.close()


def test_millimeter_cli_validation_without_remote_or_filesystem_actions():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/skill_video_modal.py"
    function = next(
        (
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
    )
    stop = next(
        (i for i, n in enumerate(function.body) if isinstance(n, ast.ImportFrom))
    )
    function.body = function.body[:stop]
    function.decorator_list = []
    scope = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    validate = scope["main"]
    args = dict(
        name="never-launched",
        source="source",
        checkpoint="policy.zip",
        seed=190000,
        variant="millimeter_near_pickup",
    )
    validate(**args, stage=71)
    with pytest.raises(ValueError, match="requires --stage"):
        validate(**args)
