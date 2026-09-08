import pytest
from robovision.train_skill_curriculum import resolve_source_lesson


def test_explicit_descent_fork_does_not_change_default_resume():
    source = {"variant": "fine_hover_bootstrap", "final_lesson": 11}
    checkpoint = {"lesson": 9}
    assert resolve_source_lesson("fine_hover_bootstrap", source, checkpoint) == (
        9,
        None,
    )
    assert resolve_source_lesson("fine_hover_bootstrap", source, checkpoint, 22) == (
        22,
        None,
    )
    assert resolve_source_lesson("fine_hover_bootstrap", source, source) == (11, None)
    assert resolve_source_lesson("fine_hover_bootstrap", source, checkpoint, 0) == (
        0,
        None,
    )


def test_invalid_requested_or_resumed_stages_rejected():
    source = {"variant": "fine_hover_bootstrap"}
    for invalid in [-1, 42, True, 2.5, "22"]:
        with pytest.raises(ValueError):
            resolve_source_lesson(
                "fine_hover_bootstrap", source, {"lesson": 9}, invalid
            )
        with pytest.raises(ValueError):
            resolve_source_lesson("fine_hover_bootstrap", source, {"lesson": invalid})
    with pytest.raises(ValueError):
        resolve_source_lesson("dense", {}, {}, 1)


def test_exact_crossvariant_guard_still_rejects_unmatched_override():
    source = {"variant": "hover_bootstrap"}
    assert (
        resolve_source_lesson("fine_hover_bootstrap", source, {"lesson": 2}, 4)[0] == 4
    )
    with pytest.raises(ValueError, match="conflicts"):
        resolve_source_lesson("fine_hover_bootstrap", source, {"lesson": 2}, 22)
