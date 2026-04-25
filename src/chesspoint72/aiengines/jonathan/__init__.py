"""Jonathan engines package.

This package provides a stable import surface for Calix:

- `chesspoint72.aiengines.jonathan.*` points to the default Calix version.
- Versioned implementations live under `chesspoint72.aiengines.jonathan.v1`,
  `v2`, `v3`, …

For now, the default surface re-exports **v1** (the baseline Calix rules
engine used by the test suite).
"""

from chesspoint72.aiengines.jonathan.v1.agent import AgentContext, EngineConfig, build_context, select_modules, with_runtime_position
from chesspoint72.aiengines.jonathan.v1.main import CalixController, build_controller, main
from chesspoint72.aiengines.jonathan.v1.registry import ModuleDescriptor, find_capability, scan_modules

__all__ = [
    "AgentContext",
    "EngineConfig",
    "ModuleDescriptor",
    "CalixController",
    "build_context",
    "select_modules",
    "with_runtime_position",
    "scan_modules",
    "find_capability",
    "build_controller",
    "main",
]

