"""Success-based adjustment of the reaching tolerance."""
from __future__ import annotations

from collections import deque


class ReachCurriculum:
    tolerances = (.06, .04, .025, .015, .01, .006)
    window_size = 2048
    minimum_steps = 2048
    threshold = .5

    def __init__(self, state=None):
        state = state or {}
        self.level = state.get("level", 0)
        self.stage_start = state.get("stage_start", 0)
        self.recent_successes = deque(state.get("recent_successes", []), maxlen=self.window_size)
        self.promotions = list(state.get("promotions", []))

    @property
    def tolerance(self):
        return self.tolerances[self.level]

    def observe(self, successes):
        self.recent_successes.extend(bool(success) for success in successes)

    def advance(self, steps):
        if (self.level == len(self.tolerances) - 1
                or steps - self.stage_start < self.minimum_steps
                or len(self.recent_successes) < self.window_size):
            return None
        success_rate = sum(self.recent_successes) / len(self.recent_successes)
        if success_rate < self.threshold:
            return None
        self.level += 1
        self.stage_start = steps
        promotion = {"step": steps, "tolerance": self.tolerance, "success_rate": success_rate}
        self.promotions.append(promotion)
        self.recent_successes.clear()
        return promotion

    def state_dict(self):
        return {"level": self.level, "stage_start": self.stage_start,
                "recent_successes": list(self.recent_successes),
                "promotions": list(self.promotions)}
