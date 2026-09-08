from robovision.train_skill_curriculum import gate_transition


def test_pass_stops_same_lesson_and_never_learns_next_lesson():
    stage = 8
    trained = []
    saved = []
    for passed in [False, True, True]:
        trained.append(stage)
        saved.append((stage, passed))
        stage, stop = gate_transition(stage, 41, passed, stop_after_passed_gate=True)
        if stop:
            break
    assert trained == [8, 8] and saved == [(8, False), (8, True)]
    assert stage == 8 and stop


def test_baseline_pass_promotes_without_stopping_then_training_pass_stops():
    stage, stop = gate_transition(
        8, 41, True, stop_after_passed_gate=True, baseline=True
    )
    assert (stage, stop) == (9, False)
    assert gate_transition(stage, 41, True, stop_after_passed_gate=True) == (9, True)


def test_default_preserves_promotion_and_terminal_stage_and_dense_behavior():
    assert gate_transition(8, 41, True) == (9, False)
    assert gate_transition(8, 41, False, stop_after_passed_gate=True) == (8, False)
    assert gate_transition(41, 41, True) == (41, False)
    assert gate_transition(41, 41, True, stop_after_passed_gate=True) == (41, True)
    assert gate_transition(0, None, True) == (0, False)
    assert gate_transition(0, None, True, stop_after_passed_gate=True) == (0, True)
