"""Wall-clock progress estimates for model-training tasks."""

from __future__ import annotations

from collections import defaultdict
from statistics import median


class TrainingEtaEstimator:
    """Estimate remaining time from completed tasks in the same model block."""

    def __init__(self, pending_tasks: list[tuple[str, str]]) -> None:
        self.pending_tasks = pending_tasks
        self.completed_durations: dict[str, list[float]] = defaultdict(list)
        self.completed_count = 0

    def record(self, block_name: str, duration_seconds: float) -> None:
        """Record the elapsed wall time for one completed training task."""
        if duration_seconds < 0:
            raise ValueError("Task duration cannot be negative.")
        if self.completed_count >= len(self.pending_tasks):
            raise ValueError("All planned training tasks are already complete.")
        expected_block, _ = self.pending_tasks[self.completed_count]
        if expected_block != block_name:
            raise ValueError(
                f"Completed task block {block_name!r} does not match "
                f"planned block {expected_block!r}."
            )
        self.completed_durations[block_name].append(duration_seconds)
        self.completed_count += 1

    def estimate_remaining(self) -> tuple[float | None, float]:
        """Return remaining seconds and the fraction estimated from block data."""
        remaining_tasks = self.pending_tasks[self.completed_count :]
        if not remaining_tasks:
            return 0.0, 1.0
        if not self.completed_durations:
            return None, 0.0

        all_durations = [
            duration
            for durations in self.completed_durations.values()
            for duration in durations
        ]
        global_median = median(all_durations)
        estimated_seconds = 0.0
        block_estimates = 0

        for block_name, _ in remaining_tasks:
            block_durations = self.completed_durations.get(block_name)
            if block_durations:
                estimated_seconds += median(block_durations)
                block_estimates += 1
            else:
                estimated_seconds += global_median

        confidence_coverage = block_estimates / len(remaining_tasks)
        return estimated_seconds, confidence_coverage


__all__ = ["TrainingEtaEstimator"]
