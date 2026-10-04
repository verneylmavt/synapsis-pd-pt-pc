from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HysteresisState:
    """Confirm initial presence and later transitions from consecutive observations."""

    inside: bool = False
    in_streak: int = 0
    out_streak: int = 0
    initialized: bool = False

    def reset_pending(self) -> None:
        """A missed frame breaks confirmation without changing latched presence."""
        self.in_streak = self.out_streak = 0

    def update(self, now_inside: bool, k: int = 3) -> str | None:
        if k < 1:
            raise ValueError("Confirmation must be positive")
        if now_inside:
            self.in_streak = min(k, self.in_streak + 1)
            self.out_streak = 0
            streak = self.in_streak
        else:
            self.out_streak = min(k, self.out_streak + 1)
            self.in_streak = 0
            streak = self.out_streak
        if streak < k:
            return None
        if not self.initialized:
            self.initialized = True
            self.inside = now_inside
            return None
        if now_inside == self.inside:
            return None
        self.inside = now_inside
        return "enter" if now_inside else "exit"
