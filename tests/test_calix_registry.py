"""Phase 7 / test 1 — registry scanner discovers tagged modules."""
from __future__ import annotations

from chesspoint72.aiengines.jonathan.registry import scan_modules


def test_scanner_returns_non_empty_list():
    descriptors = scan_modules()
    assert descriptors, "registry returned no descriptors at all"


def test_no_duplicate_module_names():
    names = [d.name for d in scan_modules()]
    assert len(names) == len(set(names))


def test_every_descriptor_has_at_least_one_capability():
    for desc in scan_modules():
        assert desc.capabilities, f"{desc.name} has no capabilities"


def test_core_capabilities_are_present():
    """The Calix agent depends on these capabilities being discoverable."""
    descriptors = scan_modules()
    flat = {cap for d in descriptors for cap in d.capabilities}
    for required in ("search", "pruning", "move_ordering", "evaluator", "uci", "board"):
        assert required in flat, f"missing capability {required!r}"


def test_pruning_config_fields_parsed():
    """The pruning_config module exposes a default factory we can introspect."""
    descriptors = scan_modules()
    cfg_mod = next(
        (d for d in descriptors if d.name.endswith("pruning.config")), None
    )
    assert cfg_mod is not None
    assert cfg_mod.config_fields, "pruning config defaults were not parsed"
    assert "nmp_enabled" in cfg_mod.config_fields
