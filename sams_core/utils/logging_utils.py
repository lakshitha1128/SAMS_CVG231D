"""Console progress reporting for the long-running image-processing
pipeline, so the user can see each stage (greyscale, binarization, ...)
as it happens.

Owned by: Project Lead / Architecture role.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass
class ProgressReporter:
    total_steps: int
    _current: int = field(default=0, init=False)
    _start: float = field(default_factory=time.time, init=False)

    def step(self, message: str) -> None:
        self._current += 1
        elapsed = time.time() - self._start
        sys.stdout.write(
            f"[{self._current}/{self.total_steps}] {message} "
            f"({elapsed:.2f}s elapsed)\n"
        )
        sys.stdout.flush()

    def done(self, message: str = "Done") -> None:
        elapsed = time.time() - self._start
        sys.stdout.write(f"-- {message} in {elapsed:.2f}s --\n")
        sys.stdout.flush()
