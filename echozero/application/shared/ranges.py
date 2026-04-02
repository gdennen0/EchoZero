"""Small shared value objects for application-layer time semantics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeRange:
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"TimeRange.start must be >= 0, got {self.start}")
        if self.end < self.start:
            raise ValueError(
                f"TimeRange.end must be >= start, got start={self.start}, end={self.end}"
            )

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, value: float) -> bool:
        return self.start <= value <= self.end
