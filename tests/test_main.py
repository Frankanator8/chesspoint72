from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import chess

import chesspoint72.main as main_module


def _args(**overrides: object) -> Namespace:
    defaults: dict[str, object] = {
        "engine": "/tmp/engine",
        "evaluator": "hce",
        "hce_modules": "classic",
        "depth": 5,
        "engine_color": "white",
        "movetime": 0.3,
        "square_size": 72,
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_build_game_config_uses_all_current_fields() -> None:
    config = main_module._build_game_config(_args())

    assert config.engine_path == "/tmp/engine"
    assert config.evaluator == "hce"
    assert config.hce_modules == "classic"
    assert config.depth == 5
    assert config.engine_color == chess.WHITE
    assert config.think_time == 0.3
    assert config.square_size == 72
    assert config.initial_fen == "8/8/8/8/8/8/8/8 w - - 0 1"


def test_build_game_config_ignores_unknown_fields_on_legacy_config(monkeypatch) -> None:
    @dataclass
    class LegacyGameConfig:
        engine_path: str | None = None
        engine_color: bool = chess.BLACK
        think_time: float = 0.2
        square_size: int = 96
        initial_fen: str | None = None

    monkeypatch.setattr(main_module, "GameConfig", LegacyGameConfig)

    config = main_module._build_game_config(_args(evaluator="material", hce_modules="pst", depth=3))

    assert isinstance(config, LegacyGameConfig)
    assert config.engine_path == "/tmp/engine"
    assert config.engine_color == chess.WHITE
    assert config.think_time == 0.3
    assert config.square_size == 72
    assert config.initial_fen == "8/8/8/8/8/8/8/8 w - - 0 1"

