from __future__ import annotations

from dataclasses import dataclass

from chesspoint72.engine.core.types import Move, NodeType


@dataclass
class TranspositionEntry:
    zobrist_hash: int
    depth: int
    score: int
    node_type: NodeType
    best_move: Move | None


class TranspositionTable:
    """Zobrist-keyed cache of evaluated positions.

    Default policy is always-replace with FIFO eviction once the rough
    in-memory footprint exceeds ``max_memory_size`` megabytes. Subclasses can
    override ``store_position`` for depth-priority or two-tier schemes.
    """

    estimated_entry_bytes: int = 64

    hash_map: dict[int, TranspositionEntry]
    max_memory_size: int

    def __init__(self, max_memory_size: int = 64) -> None:
        self.hash_map = {}
        self.max_memory_size = max_memory_size

    def store_position(
        self,
        zobrist_hash: int,
        depth_searched: int,
        score: int,
        node_type: NodeType,
        best_move: Move | None,
    ) -> None:
        if zobrist_hash not in self.hash_map and self._is_full():
            self.hash_map.pop(next(iter(self.hash_map)))
        self.hash_map[zobrist_hash] = TranspositionEntry(
            zobrist_hash=zobrist_hash,
            depth=depth_searched,
            score=score,
            node_type=node_type,
            best_move=best_move,
        )

    def retrieve_position(self, zobrist_hash: int) -> TranspositionEntry | None:
        return self.hash_map.get(zobrist_hash)

    def clear_table(self) -> None:
        self.hash_map.clear()

    def _is_full(self) -> bool:
        return len(self.hash_map) * self.estimated_entry_bytes >= self.max_memory_size * 1024 * 1024
