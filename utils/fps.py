from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class FPSCounter:
    """Compute an average frames-per-second value."""

    start_time: float = field(default_factory=time.perf_counter)
    frame_count: int = 0

    def update(self) -> float:
        """Advance the counter and return the current FPS."""

        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.frame_count / elapsed

    def reset(self) -> None:
        """Reset the FPS counter."""

        self.start_time = time.perf_counter()
        self.frame_count = 0
