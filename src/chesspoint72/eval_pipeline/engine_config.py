"""EngineConfig — named engine configurations and in-process engine builder.

Every stage in the eval pipeline compares two EngineConfig instances.
``build_engine_for_test`` converts a config into a live (search, board) pair
ready for in-process game play.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chesspoint72.engine.search.negamax.negamax import NegamaxSearch
    from chesspoint72.engine.boards.pychess import PyChessBoard


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #

@dataclass
class EngineConfig:
    name: str
    evaluator: str = "hce"
    hce_modules: str | None = "classic"
    ordering: str = "movesorter"      # stub | movesorter | movepicker
    use_tt: bool = True
    tt_policy: str = "always_replace" # always_replace | depth_preferred
    nmp_enabled: bool = True
    lmr_enabled: bool = True
    razoring_enabled: bool = True
    futility_enabled: bool = True
    aspiration_windows: bool = False
    depth: int = 4
    time_limit: float = 1.0


# --------------------------------------------------------------------------- #
# Preset configurations used throughout the pipeline
# --------------------------------------------------------------------------- #

STUB_CONFIG = EngineConfig(
    name="stub",
    evaluator="hce", hce_modules="classic",
    ordering="stub",
    use_tt=True,
)

BASELINE_B = EngineConfig(
    name="baseline_b",
    evaluator="hce", hce_modules="classic",
    ordering="movesorter",
    use_tt=True,
)

# Stage 5 A/B pairs — each differs from BASELINE_B in exactly one dimension

NO_ORDERING = EngineConfig(
    name="no_ordering",
    evaluator="hce", hce_modules="classic",
    ordering="stub",
    use_tt=True,
)

NO_TT = EngineConfig(
    name="no_tt",
    evaluator="hce", hce_modules="classic",
    ordering="movesorter",
    use_tt=False,
)

HCE_MATERIAL_ONLY = EngineConfig(
    name="hce_material_only",
    evaluator="material",
    ordering="movesorter",
    use_tt=True,
)

HCE_FULL = EngineConfig(
    name="hce_full",
    evaluator="hce", hce_modules="all",
    ordering="movesorter",
    use_tt=True,
)

NO_NMP = EngineConfig(
    name="no_nmp",
    evaluator="hce", hce_modules="classic",
    ordering="movesorter",
    use_tt=True,
    nmp_enabled=False,
)

NO_LMR = EngineConfig(
    name="no_lmr",
    evaluator="hce", hce_modules="classic",
    ordering="movesorter",
    use_tt=True,
    lmr_enabled=False,
)

MOVEPICKER_CONFIG = EngineConfig(
    name="movepicker",
    evaluator="hce", hce_modules="classic",
    ordering="movepicker",
    use_tt=True,
)

ASPIRATION_CONFIG = EngineConfig(
    name="aspiration",
    evaluator="hce", hce_modules="classic",
    ordering="movesorter",
    use_tt=True,
    aspiration_windows=True,
)

DEPTH_PREFERRED_TT = EngineConfig(
    name="depth_preferred_tt",
    evaluator="hce", hce_modules="classic",
    ordering="movesorter",
    use_tt=True,
    tt_policy="depth_preferred",
)

# Stage 8 tournament ladder configs (cumulative stack)

TOURNAMENT_CONFIGS: dict[str, EngineConfig] = {
    "A_baseline_b":   BASELINE_B,
    "B_plus_tt":      EngineConfig(
        name="B_plus_tt",
        evaluator="hce", hce_modules="classic",
        ordering="movesorter",
        use_tt=True, tt_policy="depth_preferred",
    ),
    "C_plus_pruning": EngineConfig(
        name="C_plus_pruning",
        evaluator="hce", hce_modules="classic",
        ordering="movesorter",
        use_tt=True, tt_policy="depth_preferred",
        nmp_enabled=True, lmr_enabled=True,
    ),
    "D_plus_asp":     EngineConfig(
        name="D_plus_asp",
        evaluator="hce", hce_modules="classic",
        ordering="movesorter",
        use_tt=True, tt_policy="depth_preferred",
        nmp_enabled=True, lmr_enabled=True,
        aspiration_windows=True,
    ),
    "E_plus_hce":     EngineConfig(
        name="E_plus_hce",
        evaluator="hce", hce_modules="all",
        ordering="movesorter",
        use_tt=True, tt_policy="depth_preferred",
        nmp_enabled=True, lmr_enabled=True,
        aspiration_windows=True,
    ),
    "F_nnue_variant": EngineConfig(
        name="F_nnue_variant",
        evaluator="nnue",
        ordering="movesorter",
        use_tt=True, tt_policy="depth_preferred",
        nmp_enabled=True, lmr_enabled=True,
        aspiration_windows=True,
    ),
}


# --------------------------------------------------------------------------- #
# Engine instance
# --------------------------------------------------------------------------- #

class EngineInstance:
    """A live (search, board) pair that can play moves in-process."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.search, self.board = _build(config)

    def get_best_move(self, fen: str):
        """Set the board to *fen* and return the best Move, or None."""
        self.board.set_position_from_fen(fen)
        return self.search.find_best_move(
            self.board,
            self.config.depth,
            self.config.time_limit,
        )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def build_engine_for_test(config: EngineConfig) -> "EngineInstance":
    return EngineInstance(config)


def _build(config: EngineConfig):
    from chesspoint72.engine.boards.pychess import PyChessBoard
    from chesspoint72.engine.core.transposition import (
        TranspositionTable, DepthPreferredTT, NullTranspositionTable,
    )
    from chesspoint72.engine.factory import (
        build_evaluator,
        MoveSorterPolicy,
        _StubMoveOrderingPolicy,
    )
    from chesspoint72.engine.ordering import KillerMoveTable, HistoryTable
    from chesspoint72.engine.ordering.picker_policy import MovePickerPolicy
    from chesspoint72.engine.pruning import ForwardPruningPolicy

    evaluator = build_evaluator(config.evaluator, config.hce_modules)
    board = PyChessBoard()

    # --- Ordering policy ---
    if config.ordering == "movesorter":
        killer_table  = KillerMoveTable()
        history_table = HistoryTable()
        ordering_policy = MoveSorterPolicy(killer_table, history_table)
    elif config.ordering == "movepicker":
        ordering_policy = MovePickerPolicy()
        killer_table  = KillerMoveTable()
        history_table = HistoryTable()
    else:
        ordering_policy = _StubMoveOrderingPolicy()
        killer_table  = KillerMoveTable()
        history_table = HistoryTable()

    # --- Transposition table ---
    if not config.use_tt:
        tt = NullTranspositionTable()
    elif config.tt_policy == "depth_preferred":
        tt = DepthPreferredTT()
    else:
        tt = TranspositionTable()

    # --- Pruning config ---
    from chesspoint72.engine.pruning.config import default_pruning_config
    from dataclasses import replace as dc_replace
    pruning_cfg = default_pruning_config()
    pruning_cfg = dc_replace(
        pruning_cfg,
        nmp_enabled=config.nmp_enabled,
        lmr_enabled=config.lmr_enabled,
        razoring_enabled=config.razoring_enabled,
        futility_enabled=config.futility_enabled,
    )
    pruning_policy = ForwardPruningPolicy(pruning_cfg)

    # --- Search ---
    if config.aspiration_windows:
        from chesspoint72.engine.search.negamax.aspiration import AspirationNegamaxSearch
        search = AspirationNegamaxSearch(
            evaluator, tt, ordering_policy, pruning_policy, pruning_cfg,
        )
    else:
        from chesspoint72.engine.search.negamax import NegamaxSearch
        search = NegamaxSearch(
            evaluator, tt, ordering_policy, pruning_policy, pruning_cfg,
        )

    # Share killer/history tables with the search
    search.killer_table  = killer_table
    search.history_table = history_table

    return search, board
