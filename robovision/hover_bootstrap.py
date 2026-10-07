"""Longer open-hand stabilization lessons before descent, trained only by PPO."""

from dataclasses import asdict
from .open_bootstrap import OpenBootstrapEnv, BootstrapLesson, LESSONS as OPEN_LESSONS

VERSION = "open-hover-bootstrap-v1"
HOVER_STEPS = (4, 5, 8, 12, 18, 25)
LESSONS = tuple(
    [
        BootstrapLesson(0.075, 1.0, stable_steps=n, reset_height=0.075)
        for n in HOVER_STEPS
    ]
    + list(OPEN_LESSONS[3:])
)
FIRST_DESCENT_LESSON = len(HOVER_STEPS)
FULL_APPROACH_LESSON = next(
    (i for i, l in enumerate(LESSONS) if l.target_height == 0.0)
)
FULL_PICKUP_LESSON = next(
    (i for i, l in enumerate(LESSONS) if l.approach_fraction == 0.0)
)
FINAL_LESSON = len(LESSONS) - 1


class HoverBootstrapEnv(OpenBootstrapEnv):
    lessons = LESSONS
    first_descent_lesson = FIRST_DESCENT_LESSON
    fixed_descent_reset_lessons = (FIRST_DESCENT_LESSON, FIRST_DESCENT_LESSON + 1)

    @staticmethod
    def specification():
        spec = OpenBootstrapEnv.specification()
        spec.update(
            version=VERSION,
            lessons=[asdict(l) for l in LESSONS],
            first_descent_lesson=FIRST_DESCENT_LESSON,
            full_approach_lesson=FULL_APPROACH_LESSON,
            full_pickup_lesson=FULL_PICKUP_LESSON,
            final_lesson=FINAL_LESSON,
            hover_steps=list(HOVER_STEPS),
        )
        spec["optional_bootstrap_height_jitter"].update(
            lessons=list(range(FIRST_DESCENT_LESSON)),
            strict_descent_additional_lessons=[
                FIRST_DESCENT_LESSON,
                FIRST_DESCENT_LESSON + 1,
            ],
        )
        spec["optional_strict_descent"].update(
            start_lesson=FIRST_DESCENT_LESSON,
            fixed_training_reset_lessons=[
                FIRST_DESCENT_LESSON,
                FIRST_DESCENT_LESSON + 1,
            ],
        )
        return spec


FINE_HOVER_STEPS = tuple(range(4, 26))
FINE_LESSONS = tuple(
    [
        BootstrapLesson(0.075, 1.0, stable_steps=n, reset_height=0.075)
        for n in FINE_HOVER_STEPS
    ]
    + list(OPEN_LESSONS[3:])
)
FINE_FIRST_DESCENT_LESSON = len(FINE_HOVER_STEPS)
FINE_FULL_APPROACH_LESSON = next(
    (i for i, l in enumerate(FINE_LESSONS) if l.target_height == 0.0)
)
FINE_FULL_PICKUP_LESSON = next(
    (i for i, l in enumerate(FINE_LESSONS) if l.approach_fraction == 0.0)
)
FINE_FINAL_LESSON = len(FINE_LESSONS) - 1


class FineHoverBootstrapEnv(HoverBootstrapEnv):
    lessons = FINE_LESSONS
    first_descent_lesson = FINE_FIRST_DESCENT_LESSON
    fixed_descent_reset_lessons = (
        FINE_FIRST_DESCENT_LESSON,
        FINE_FIRST_DESCENT_LESSON + 1,
    )

    @staticmethod
    def specification():
        spec = HoverBootstrapEnv.specification()
        spec.update(
            version="fine-open-hover-bootstrap-v1",
            lessons=[asdict(l) for l in FINE_LESSONS],
            first_descent_lesson=FINE_FIRST_DESCENT_LESSON,
            full_approach_lesson=FINE_FULL_APPROACH_LESSON,
            full_pickup_lesson=FINE_FULL_PICKUP_LESSON,
            final_lesson=FINE_FINAL_LESSON,
            hover_steps=list(FINE_HOVER_STEPS),
        )
        spec["optional_bootstrap_height_jitter"].update(
            lessons=list(range(FINE_FIRST_DESCENT_LESSON)),
            strict_descent_additional_lessons=[
                FINE_FIRST_DESCENT_LESSON,
                FINE_FIRST_DESCENT_LESSON + 1,
            ],
        )
        spec["optional_strict_descent"].update(
            start_lesson=FINE_FIRST_DESCENT_LESSON,
            fixed_training_reset_lessons=[
                FINE_FIRST_DESCENT_LESSON,
                FINE_FIRST_DESCENT_LESSON + 1,
            ],
        )
        return spec


_FINE_TWENTY_INDEX = next(
    (i for i, l in enumerate(FINE_LESSONS) if l.target_height == 0.02)
)
_FINE_MIXED_INDEX = next(
    (i for i, l in enumerate(FINE_LESSONS) if l.approach_fraction < 1.0)
)
FINE_DESCENT_LESSONS = tuple(
    list(FINE_LESSONS[: _FINE_TWENTY_INDEX + 1])
    + [BootstrapLesson(mm / 1000.0, 1.0) for mm in range(19, -1, -1)]
    + list(FINE_LESSONS[_FINE_MIXED_INDEX:])
)


class FineDescentBootstrapEnv(FineHoverBootstrapEnv):
    lessons = FINE_DESCENT_LESSONS

    @staticmethod
    def specification():
        spec = FineHoverBootstrapEnv.specification()
        spec.update(
            version="fine-descent-bootstrap-v1",
            lessons=[asdict(l) for l in FINE_DESCENT_LESSONS],
            full_approach_lesson=next(
                (
                    i
                    for i, l in enumerate(FINE_DESCENT_LESSONS)
                    if l.target_height == 0.0
                )
            ),
            full_pickup_lesson=next(
                (
                    i
                    for i, l in enumerate(FINE_DESCENT_LESSONS)
                    if l.approach_fraction == 0.0
                )
            ),
            final_lesson=len(FINE_DESCENT_LESSONS) - 1,
            fine_descent_start_height=0.02,
            fine_descent_increment_m=0.001,
        )
        return spec
