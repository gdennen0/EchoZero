"""Minimal MA3 adapter placeholder for infrastructure wiring.

Concrete implementation and protocol behavior come later.
"""

from typing import Protocol


class MA3Adapter(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_status(self) -> dict: ...
