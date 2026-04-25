"""Stable Calix registry import surface (defaults to v1)."""

from chesspoint72.aiengines.jonathan.v1.registry import (  # noqa: F401
    ModuleDescriptor,
    find_capability,
    scan_modules,
)

__all__ = [
    "ModuleDescriptor",
    "find_capability",
    "scan_modules",
]

