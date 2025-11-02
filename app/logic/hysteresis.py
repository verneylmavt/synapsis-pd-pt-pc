from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class HysteresisState:
    """
    Maintain a stable inside/outside state using hysteresis logic.

    Purpose:
        - Used to smooth noisy binary signals (inside/outside detections).
        - Prevents rapid toggling of states when readings fluctuate near boundaries.
        - Requires several consecutive consistent readings (controlled by parameter `k`)
        before confirming a change of state.

    Attributes:
        inside (bool): The current stable (latched) state.
            - True  → currently considered "inside"
            - False → currently considered "outside"

        in_streak (int): Number of consecutive "inside" readings since the last "outside".
        out_streak (int): Number of consecutive "outside" readings since the last "inside".
    """
    inside: bool = False
    in_streak: int = 0
    out_streak: int = 0

    def update(self, now_inside: bool, k: int = 3) -> Optional[str]:
        """
        Update the hysteresis state with a new inside/outside observation.

        Args:
            now_inside (bool): The latest reading (True if inside, False if outside).
            k (int): Number of consecutive consistent readings required to confirm
                    a state change. Default is 3.

        Returns:
            Optional[str]:
                - "enter" → triggered when transitioning from outside → inside
                (after k consecutive "inside" readings)
                - "exit"  → triggered when transitioning from inside → outside
                (after k consecutive "outside" readings)
                - None    → returned when no stable transition occurs

        Logic summary:
            The function tracks how many consecutive frames/updates the signal
            has stayed inside or outside. Only when a streak reaches k frames
            does it commit a stable state change.
        """

        # Case 1: The new reading says the object is inside the area
        if now_inside:
            # Increment inside streak counter
            self.in_streak += 1
            # Reset outside streak since we’re now detecting inside
            self.out_streak = 0

            # If the previous stable state was "outside" and we’ve seen
            # at least k consecutive "inside" readings → trigger "enter"
            if not self.inside and self.in_streak >= k:
                self.inside = True  # latch the new stable state
                return "enter"

            # Otherwise, no stable change yet → return None
            return None

        # Case 2: The new reading says the object is outside the area
        else:
            # Increment outside streak counter
            self.out_streak += 1
            # Reset inside streak since we’re now detecting outside
            self.in_streak = 0

            # If the previous stable state was "inside" and we’ve seen
            # at least k consecutive "outside" readings → trigger "exit"
            if self.inside and self.out_streak >= k:
                self.inside = False  # latch the new stable state
                return "exit"

            # Otherwise, no stable change yet → return None
            return None