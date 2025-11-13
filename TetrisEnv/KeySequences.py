from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .Moves import Keys
from .Pieces import Piece, PieceType
from .Placements import Placement, PlacementFinder
from .RotationSystem import RotationSystem
from .helpers import overlaps


@dataclass(frozen=True)
class Goal:
    """Desired final placement (rotation index, resting row, column)."""

    rotation: int
    row: int
    col: int


@dataclass(frozen=True)
class State:
    """Piece configuration used during BFS while the piece is floating."""

    row: int
    col: int
    rotation: int


class KeySequenceFinder:
    """Compute minimal key sequences that reach specified resting placements."""

    _ALLOWED_KEYS: Tuple[int, ...] = (
        Keys.TAP_LEFT,
        Keys.TAP_RIGHT,
        Keys.DAS_LEFT,
        Keys.DAS_RIGHT,
        Keys.CLOCKWISE,
        Keys.ANTICLOCKWISE,
        Keys.ROTATE_180,
        Keys.SOFT_DROP,
    )

    def __init__(self, rotation_system: RotationSystem | None = None) -> None:
        self._rotation_system = rotation_system or RotationSystem()
        self._placement_finder = PlacementFinder(self._rotation_system)

    def find_sequences(
        self,
        board: np.ndarray,
        piece: Piece,
        goals: Sequence[Goal],
    ) -> List[Optional[List[int]]]:
        """Return minimal sequences (Keys ints) that reach each goal placement."""

        self._validate_board(board)
        if not goals:
            return []

        rotation_system = self._rotation_system
        start_piece = self._copy_piece(piece)
        start_piece.cells = np.array(
            rotation_system.orientations[piece.piece_type][piece.r],
            dtype=np.int32,
            copy=True,
        )

        if overlaps(start_piece.cells, start_piece.loc, board):
            raise ValueError("Starting piece overlaps the board.")

        goal_map = self._normalize_goals(goals)
        results: List[Optional[List[int]]] = [None] * len(goals)

        queue: deque[State] = deque()
        start_state = self._state_from_piece(start_piece)
        queue.append(start_state)
        parents: MutableMapping[State, Tuple[Optional[State], Optional[int]]] = {
            start_state: (None, None)
        }

        remaining = set(goal_map.keys())
        while queue and remaining:
            current_state = queue.popleft()
            current_piece = self._piece_from_state(
                current_state, start_piece.piece_type, rotation_system
            )

            drop_row = self._hard_drop_row(current_piece, board)
            goal_key = (current_state.rotation % 4, drop_row, current_state.col)
            if goal_key in remaining:
                sequence = (
                    [Keys.START]
                    + self._reconstruct_sequence(current_state, parents)
                    + [Keys.HARD_DROP]
                )
                results[goal_map[goal_key]] = sequence
                remaining.remove(goal_key)
                if not remaining:
                    break

            for key in self._ALLOWED_KEYS:
                next_piece = self._apply_key(current_piece, key, board, rotation_system)
                if next_piece is None:
                    continue
                next_state = self._state_from_piece(next_piece)
                if next_state in parents:
                    continue
                parents[next_state] = (current_state, key)
                queue.append(next_state)

        return results

    def find_all(
        self,
        board: np.ndarray,
        piece: Piece,
        return_timing: bool = False,
    ) -> (
        Tuple[List[Tuple[Placement, List[int]]], Dict[str, float]]
        | List[Tuple[Placement, List[int]]]
    ):
        """Enumerate every reachable placement and its minimal key sequence."""

        t0 = time.perf_counter()
        placements = self._placement_finder.find(board, piece.piece_type)
        t1 = time.perf_counter()

        goals = [
            Goal(rotation=placement.rotation, row=placement.row, col=placement.col)
            for placement in placements
        ]
        raw_sequences = self.find_sequences(board, piece, goals)
        t2 = time.perf_counter()

        sequences = [
            sequence if sequence is not None else [-1] for sequence in raw_sequences
        ]
        results = list(zip(placements, sequences))

        if return_timing:
            timing = {
                "placements": t1 - t0,
                "paths": t2 - t1,
                "total": t2 - t0,
            }
            return results, timing

        return results

    @staticmethod
    def _validate_board(board: np.ndarray) -> None:
        if board.ndim != 2:
            raise ValueError("Board must be a 2-D array.")

    @staticmethod
    def _copy_piece(piece: Piece) -> Piece:
        return Piece(
            piece_type=piece.piece_type,
            loc=np.array(piece.loc, dtype=np.int32, copy=True),
            r=int(piece.r),
            cells=np.array(piece.cells, dtype=np.int32, copy=True),
        )

    @staticmethod
    def _apply_rotation(
        piece: Piece,
        delta_r: int,
        board: np.ndarray,
        rotation_system: RotationSystem,
    ) -> Optional[Piece]:
        new_r = (piece.r + delta_r) % 4
        cells = rotation_system.orientations[piece.piece_type][new_r]
        loc = piece.loc

        rotated = KeySequenceFinder._copy_piece(piece)
        if not overlaps(cells=cells, loc=loc, board=board):
            rotated.r = new_r
            rotated.cells = cells
            return rotated

        kick_table = (
            rotation_system.i_kicks
            if piece.piece_type == PieceType.I
            else rotation_system.kicks
        )
        kicked_piece = KeySequenceFinder._copy_piece(piece)
        kicked, cells, loc, r, _, _ = rotation_system.kick_piece(
            kick_table=kick_table,
            piece=kicked_piece,
            cells=cells,
            new_r=new_r,
            delta_r=delta_r,
            board=board,
        )
        if not kicked:
            return None

        kicked_piece.cells = cells
        kicked_piece.loc = loc
        kicked_piece.r = r
        return kicked_piece

    @staticmethod
    def _apply_translation(
        piece: Piece,
        delta_col: int,
        board: np.ndarray,
    ) -> Optional[Piece]:
        if delta_col == 0:
            return KeySequenceFinder._copy_piece(piece)

        direction = -1 if delta_col < 0 else 1
        limit = abs(delta_col)
        last_valid_loc = None
        for step in range(1, limit + 1):
            try_loc = piece.loc + np.array([0, direction * step], dtype=np.int32)
            if overlaps(piece.cells, try_loc, board):
                break
            last_valid_loc = try_loc

        if last_valid_loc is None:
            return None

        shifted = KeySequenceFinder._copy_piece(piece)
        shifted.loc = last_valid_loc
        return shifted

    @classmethod
    def _apply_key(
        cls,
        piece: Piece,
        key: int,
        board: np.ndarray,
        rotation_system: RotationSystem,
    ) -> Optional[Piece]:
        if key == Keys.CLOCKWISE:
            return cls._apply_rotation(piece, +1, board, rotation_system)
        if key == Keys.ANTICLOCKWISE:
            return cls._apply_rotation(piece, -1, board, rotation_system)
        if key == Keys.ROTATE_180:
            return cls._apply_rotation(piece, +2, board, rotation_system)
        if key == Keys.TAP_LEFT:
            return cls._apply_translation(piece, -1, board)
        if key == Keys.TAP_RIGHT:
            return cls._apply_translation(piece, +1, board)
        if key == Keys.DAS_LEFT:
            return cls._apply_translation(piece, -100, board)
        if key == Keys.DAS_RIGHT:
            return cls._apply_translation(piece, +100, board)
        if key == Keys.SOFT_DROP:
            dropped = cls._copy_piece(piece)
            loc = np.array(piece.loc, dtype=np.int32, copy=True)
            while True:
                next_loc = loc + np.array([1, 0], dtype=np.int32)
                if overlaps(dropped.cells, next_loc, board):
                    break
                loc = next_loc
            if np.array_equal(loc, dropped.loc):
                return None
            dropped.loc = loc
            return dropped
        if key == Keys.START:
            return cls._copy_piece(piece)
        if key == Keys.HARD_DROP:
            raise ValueError("Hard drop should not be applied during BFS exploration.")
        if key == Keys.HOLD:
            raise ValueError("Hold key is not permitted for active piece movement.")
        raise ValueError(f"Unsupported key: {key}")

    @staticmethod
    def _state_from_piece(piece: Piece) -> State:
        return State(int(piece.loc[0]), int(piece.loc[1]), int(piece.r) % 4)

    @staticmethod
    def _piece_from_state(
        state: State,
        piece_type: PieceType,
        rotation_system: RotationSystem,
    ) -> Piece:
        cells = rotation_system.orientations[piece_type][state.rotation]
        return Piece(
            piece_type=piece_type,
            loc=np.array([state.row, state.col], dtype=np.int32),
            r=state.rotation,
            cells=np.array(cells, dtype=np.int32, copy=True),
        )

    @staticmethod
    def _hard_drop_row(piece: Piece, board: np.ndarray) -> int:
        loc = np.array(piece.loc, dtype=np.int32, copy=True)
        while True:
            next_loc = loc + np.array([1, 0], dtype=np.int32)
            if overlaps(piece.cells, next_loc, board):
                break
            loc = next_loc
        return int(loc[0])

    @staticmethod
    def _reconstruct_sequence(
        end_state: State,
        parents: Mapping[State, Tuple[Optional[State], Optional[int]]],
    ) -> List[int]:
        keys: List[int] = []
        state = end_state
        while True:
            parent_state, key = parents[state]
            if parent_state is None or key is None:
                break
            keys.append(key)
            state = parent_state
        keys.reverse()
        return keys

    @staticmethod
    def _normalize_goals(goals: Sequence[Goal]) -> Dict[Tuple[int, int, int], int]:
        mapping: Dict[Tuple[int, int, int], int] = {}
        for idx, goal in enumerate(goals):
            key = (goal.rotation % 4, int(goal.row), int(goal.col))
            mapping[key] = idx
        return mapping


__all__ = [
    "Goal",
    "State",
    "KeySequenceFinder",
]
