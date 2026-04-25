"""Module registry scanner for the Calix engine.

Walks ``src/chesspoint72`` at startup and builds a list of
``ModuleDescriptor`` records, one per Python module that declares a
``# @capability: <name>`` comment. The Module Selector Agent reads these
descriptors to decide which modules to activate.

Discovery contract:
- A module is a candidate iff it contains at least one
  ``# @capability: <name>`` line in the first 40 lines.
- Multiple ``# @capability:`` lines stack: the descriptor's
  ``capabilities`` list contains every name found.
- A module's default config fields are parsed from a ``default_*_config``
  factory function defined immediately below the capability tags. The
  parser only looks for ``key=value`` keyword arguments inside the
  ``replace(...)``- or dataclass-style returned struct; everything else is
  best-effort and may be empty.

The scanner is intentionally tolerant: any file it cannot parse cleanly is
skipped silently. This keeps Calix's startup robust to refactors elsewhere
in the tree.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Tag prefix used in source files. ``//`` was the originally specified marker
# but Python source uses ``#``; both forms are accepted so the same convention
# can be reused if a non-Python module is ever added.
_CAPABILITY_PREFIXES: tuple[str, ...] = ("# @capability:", "// @capability:")


@dataclass
class ModuleDescriptor:
    """One row in the registry: a single discovered module."""

    name: str
    path: str
    capabilities: list[str] = field(default_factory=list)
    config_fields: dict[str, Any] = field(default_factory=dict)
    enabled: bool = False

    def has_capability(self, capability: str) -> bool:
        return capability in self.capabilities


def _read_capability_tags(file_path: Path) -> list[str]:
    """Return the ``@capability:`` names declared in the first 40 lines."""
    tags: list[str] = []
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            for i, raw in enumerate(fh):
                if i >= 40:
                    break
                line = raw.strip()
                for prefix in _CAPABILITY_PREFIXES:
                    if line.startswith(prefix):
                        name = line[len(prefix):].strip()
                        if name:
                            tags.append(name)
                        break
    except OSError:
        return []
    return tags


def _parse_config_fields(file_path: Path) -> dict[str, Any]:
    """Best-effort parse of a ``default_*_config`` factory's keyword args.

    Walks the file's AST looking for a top-level function whose name starts
    with ``default_`` and ends with ``_config``. The first ``Return`` inside
    that function whose value is a ``Call`` node is inspected; each keyword
    argument with a literal value contributes one entry to the returned dict.
    Anything non-literal (callbacks, attribute lookups, complex expressions)
    is silently ignored.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    fields: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not (node.name.startswith("default_") and node.name.endswith("_config")):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Call):
                for kw in sub.value.keywords:
                    if kw.arg is None:
                        continue
                    try:
                        fields[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        continue
                break
    return fields


def _module_name_from_path(root: Path, file_path: Path) -> str:
    """Convert ``src/chesspoint72/foo/bar.py`` into ``chesspoint72.foo.bar``."""
    rel = file_path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def scan_modules(root: str | os.PathLike[str] | None = None) -> list[ModuleDescriptor]:
    """Walk *root* recursively and return one descriptor per tagged module.

    *root* defaults to the ``chesspoint72`` package directory inferred from
    this file's location. Duplicates (same module name) are folded into a
    single descriptor whose ``capabilities`` is the union of both.
    """
    if root is None:
        # Infer the `.../src/chesspoint72` package directory robustly.
        # (The Calix package lives under `chesspoint72/aiengines/jonathan/v*/`,
        # so fixed parent indexing is fragile.)
        here = Path(__file__).resolve()
        root = next((p for p in here.parents if p.name == "chesspoint72"), here.parents[0])
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    by_name: dict[str, ModuleDescriptor] = {}
    for dirpath, _dirnames, filenames in os.walk(root_path):
        if "__pycache__" in dirpath:
            continue
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            file_path = Path(dirpath) / fname
            tags = _read_capability_tags(file_path)
            if not tags:
                continue
            mod_name = _module_name_from_path(root_path, file_path)
            existing = by_name.get(mod_name)
            if existing is None:
                by_name[mod_name] = ModuleDescriptor(
                    name=mod_name,
                    path=str(file_path),
                    capabilities=list(tags),
                    config_fields=_parse_config_fields(file_path),
                )
            else:
                for tag in tags:
                    if tag not in existing.capabilities:
                        existing.capabilities.append(tag)

    return sorted(by_name.values(), key=lambda d: d.name)


def find_capability(
    descriptors: list[ModuleDescriptor], capability: str
) -> ModuleDescriptor | None:
    """Return the first descriptor advertising *capability*, or None."""
    for desc in descriptors:
        if desc.has_capability(capability):
            return desc
    return None
