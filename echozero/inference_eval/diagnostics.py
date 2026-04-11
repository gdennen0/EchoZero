from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    path: str
    message: str
    severity: str


@dataclass(slots=True)
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, code: str, path: str, message: str) -> None:
        self.errors.append(
            ValidationIssue(code=code, path=path, message=message, severity="error")
        )

    def add_warning(self, code: str, path: str, message: str) -> None:
        self.warnings.append(
            ValidationIssue(code=code, path=path, message=message, severity="warning")
        )

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        return self
