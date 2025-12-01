from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from .KeySequences import KeySequenceFinder
from .Moves import Keys
from .Pieces import Piece, PieceType
from .RotationSystem import RotationSystem
from .helpers import overlaps


class BitboardKeySequenceFinder(KeySequenceFinder):
    _VISIBLE_ROWS = 20
    _BOARD_COLS = 10
    _ROTATIONS = 4
    _KICK_STATES = 2

    def __init__(self, rotation_system: RotationSystem | None = None) -> None:
        super().__init__(rotation_system)
        self._bitpiece_cache: Dict[PieceType, Dict[str, np.ndarray]] = {}
        self._col_bits = (
            np.uint16(1) << np.arange(self._BOARD_COLS, dtype=np.uint16)
        ).astype(np.uint16)

    def find_all(
        self,
        board: np.ndarray,
        piece: Piece,
        max_len: int,
        is_hold: bool,
        return_timing: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
        self._validate_board(board)

        t_start = time.perf_counter()

        rotation_system = self._rotation_system
        start_piece = self._copy_piece(piece)
        start_piece.cells = np.array(
            rotation_system.orientations[piece.piece_type][piece.r],
            dtype=np.int32,
            copy=True,
        )

        if overlaps(start_piece.cells, start_piece.loc, board):
            result = self._empty_sequences_array(board, max_len)
            timing = self._timing_dict(t_start)
            return (result, timing) if return_timing else result

        sequences_array = self._empty_sequences_array(board, max_len)

        board_rows, board_cols = board.shape
        assert board_cols == self._BOARD_COLS

        board_mask = self._board_to_bitmask(board)
        piece_data = self._bitpiece_data(start_piece.piece_type)
        min_col_offsets = piece_data["min_cols"]
        rotation_bounds = piece_data["row_bounds"]

        state_count = board_rows * board_cols * self._ROTATIONS
        visible_start = board_rows - self._VISIBLE_ROWS
        visible_end = visible_start + self._VISIBLE_ROWS
        total_positions = self._ROTATIONS * board_cols * self._KICK_STATES

        visited = np.zeros(state_count, dtype=np.bool_)
        parents = np.full(state_count, -1, dtype=np.int32)
        parent_keys = np.full(state_count, -1, dtype=np.int16)
        depths = np.full(state_count, -1, dtype=np.int16)
        queue = np.empty(state_count, dtype=np.int32)
        filled_slots = np.zeros(total_positions, dtype=np.bool_)
        entered_via_kick = np.zeros(state_count, dtype=np.bool_)

        piece_type = start_piece.piece_type
        kick_table = (
            rotation_system.i_kicks
            if piece_type == PieceType.I
            else rotation_system.kicks
        )

        start_row = int(start_piece.loc[0])
        start_col = int(start_piece.loc[1])
        start_rotation = int(start_piece.r) % 4
        start_state = self._encode_state(
            start_row, start_col, start_rotation, board_cols, min_col_offsets
        )
        if start_state is None:
            timing = self._timing_dict(t_start)
            return (sequences_array, timing) if return_timing else sequences_array

        queue_head = 0
        queue_tail = 0

        queue[queue_tail] = start_state
        queue_tail += 1

        visited[start_state] = True
        depths[start_state] = 0

        max_sequence_len = max_len if is_hold else max_len - 1
        if max_sequence_len < 2:
            raise ValueError(
                "max_len must be at least 2 for hold sequences and at least 3 when "
                "is_hold is False."
            )
        max_path_length = max_sequence_len - 2

        while queue_head != queue_tail:
            current_state = queue[queue_head]
            queue_head += 1

            current_row, current_col, current_rotation = self._decode_state(
                current_state, board_cols, min_col_offsets
            )
            current_depth = int(depths[current_state])

            drop_row = self._hard_drop_row(
                board_mask,
                piece_data,
                current_rotation,
                current_row,
                current_col,
                rotation_bounds,
            )
            if visible_start <= drop_row < visible_end:
                has_kick = bool(entered_via_kick[current_state])
                idx = self._placement_index(
                    current_rotation,
                    current_col,
                    board_cols,
                    min_col_offsets,
                    has_kick,
                )
                if idx is not None and not filled_slots[idx]:
                    sequence = self._materialize_sequence(
                        current_state, parents, parent_keys, is_hold
                    )
                    if len(sequence) <= max_sequence_len:
                        padded = np.full(max_len, Keys.PAD, dtype=np.int32)
                        padded[: len(sequence)] = sequence
                        sequences_array[idx] = padded
                        filled_slots[idx] = True

            if current_depth >= max_path_length:
                continue

            for key in self._ALLOWED_KEYS:
                next_state_tuple = self._apply_key_state(
                    board_mask,
                    piece_data,
                    rotation_bounds,
                    kick_table,
                    current_row,
                    current_col,
                    current_rotation,
                    key,
                    board_rows,
                )
                if next_state_tuple is None:
                    continue

                next_row, next_col, next_rotation, used_kick = next_state_tuple
                next_state = self._encode_state(
                    next_row,
                    next_col,
                    next_rotation,
                    board_cols,
                    min_col_offsets,
                )
                if next_state is None:
                    continue

                if visited[next_state]:
                    continue

                next_depth = current_depth + 1
                if next_depth > max_path_length:
                    continue

                visited[next_state] = True
                parents[next_state] = current_state
                parent_keys[next_state] = key
                depths[next_state] = next_depth
                entered_via_kick[next_state] = used_kick

                queue[queue_tail] = next_state
                queue_tail += 1

        timing = self._timing_dict(t_start)
        return (sequences_array, timing) if return_timing else sequences_array

    def _bitpiece_data(self, piece_type: PieceType) -> Dict[str, np.ndarray]:
        cached = self._bitpiece_cache.get(piece_type)
        if cached is not None:
            return cached

        rotation_system = self._rotation_system
        orientations = rotation_system.orientations[piece_type]
        row_masks = []
        row_offsets = []
        min_cols = []
        max_cols = []
        row_bounds = []

        for rotation_cells in orientations:
            rel_rows = rotation_cells[:, 0]
            rel_cols = rotation_cells[:, 1]
            unique_rows = np.unique(rel_rows)
            masks = np.zeros(len(unique_rows), dtype=np.uint16)
            for idx, rel_row in enumerate(unique_rows):
                cols = rel_cols[rel_rows == rel_row]
                mask = np.uint16(0)
                for col in cols:
                    mask |= np.uint16(1 << int(col))
                masks[idx] = mask
            row_offsets.append(unique_rows.astype(np.int32))
            row_masks.append(masks)
            min_cols.append(int(rel_cols.min()))
            max_cols.append(int(rel_cols.max()))
            row_bounds.append(
                np.array([int(rel_rows.min()), int(rel_rows.max())], dtype=np.int16)
            )

        data = {
            "row_offsets": row_offsets,
            "row_masks": row_masks,
            "min_cols": np.array(min_cols, dtype=np.int16),
            "max_cols": np.array(max_cols, dtype=np.int16),
            "row_bounds": np.stack(row_bounds, axis=0),
        }
        self._bitpiece_cache[piece_type] = data
        return data

    def _board_to_bitmask(self, board: np.ndarray) -> np.ndarray:
        occupied = (board != 0).astype(np.uint16)
        mask_rows = (
            (occupied * self._col_bits).sum(axis=1, dtype=np.uint32).astype(np.uint16)
        )
        return mask_rows

    @staticmethod
    def _empty_sequences_array(board: np.ndarray, max_len: int) -> np.ndarray:
        rotations = BitboardKeySequenceFinder._ROTATIONS
        kick_states = BitboardKeySequenceFinder._KICK_STATES
        board_rows, board_cols = board.shape

        if max_len < 2:
            raise ValueError(
                "max_len must be at least 2 to include START and HARD_DROP keys."
            )
        if board_cols != BitboardKeySequenceFinder._BOARD_COLS:
            raise ValueError("Board must have exactly 10 columns.")
        if board_rows < BitboardKeySequenceFinder._VISIBLE_ROWS:
            raise ValueError("Board must have at least 20 rows.")

        total_positions = rotations * board_cols * kick_states
        return np.full((total_positions, max_len), Keys.PAD, dtype=np.int32)

    @staticmethod
    def _timing_dict(t_start: float) -> Dict[str, float]:
        duration = time.perf_counter() - t_start
        return {"placements": 0.0, "paths": duration, "total": duration}

    @staticmethod
    def _encode_state(
        row: int,
        col: int,
        rotation: int,
        board_cols: int,
        min_col_offsets: np.ndarray,
    ) -> Optional[int]:
        norm_col = col + int(min_col_offsets[rotation % 4])
        if norm_col < 0 or norm_col >= board_cols:
            return None
        return ((row * board_cols) + norm_col) * 4 + (rotation % 4)

    @staticmethod
    def _decode_state(
        state: int, board_cols: int, min_col_offsets: np.ndarray
    ) -> Tuple[int, int, int]:
        rotation = state % 4
        base = state // 4
        row = base // board_cols
        norm_col = base % board_cols
        col = norm_col - int(min_col_offsets[rotation])
        return row, col, rotation

    @staticmethod
    def _placement_index(
        rotation: int,
        col: int,
        board_cols: int,
        min_col_offsets: np.ndarray,
        has_kick: bool,
    ) -> Optional[int]:
        rotation_idx = rotation % 4
        actual_col = col + int(min_col_offsets[rotation_idx])
        if actual_col < 0 or actual_col >= board_cols:
            return None
        kick_idx = 1 if has_kick else 0
        stride = BitboardKeySequenceFinder._KICK_STATES
        return rotation_idx * board_cols * stride + actual_col * stride + kick_idx

    def _apply_key_state(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        row_bounds: np.ndarray,
        kick_table: Dict[Tuple[int, int], np.ndarray],
        row: int,
        col: int,
        rotation: int,
        key: int,
        board_rows: int,
    ) -> Optional[Tuple[int, int, int, bool]]:
        if key == Keys.CLOCKWISE:
            return self._rotate_state(
                board_mask, piece_data, row_bounds, kick_table, row, col, rotation, +1
            )
        if key == Keys.ANTICLOCKWISE:
            return self._rotate_state(
                board_mask, piece_data, row_bounds, kick_table, row, col, rotation, -1
            )
        if key == Keys.ROTATE_180:
            return self._rotate_state(
                board_mask, piece_data, row_bounds, kick_table, row, col, rotation, +2
            )
        if key == Keys.TAP_LEFT:
            shifted = self._shift_once(
                board_mask, piece_data, row, col, rotation, -1, row_bounds
            )
            if shifted is None:
                return None
            return shifted[0], shifted[1], shifted[2], False
        if key == Keys.TAP_RIGHT:
            shifted = self._shift_once(
                board_mask, piece_data, row, col, rotation, +1, row_bounds
            )
            if shifted is None:
                return None
            return shifted[0], shifted[1], shifted[2], False
        if key == Keys.DAS_LEFT:
            shifted = self._shift_das(
                board_mask, piece_data, row, col, rotation, -1, row_bounds
            )
            if shifted is None:
                return None
            return shifted[0], shifted[1], shifted[2], False
        if key == Keys.DAS_RIGHT:
            shifted = self._shift_das(
                board_mask, piece_data, row, col, rotation, +1, row_bounds
            )
            if shifted is None:
                return None
            return shifted[0], shifted[1], shifted[2], False
        if key == Keys.SOFT_DROP:
            dropped = self._soft_drop_state(
                board_mask, piece_data, row, col, rotation, board_rows, row_bounds
            )
            if dropped is None:
                return None
            return dropped[0], dropped[1], dropped[2], False
        if key == Keys.START:
            return (row, col, rotation, False)
        if key in (Keys.HARD_DROP, Keys.HOLD):
            raise ValueError("HARD_DROP and HOLD are not valid movement keys.")
        raise ValueError(f"Unsupported key: {key}")

    def _rotate_state(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        row_bounds: np.ndarray,
        kick_table: Dict[Tuple[int, int], np.ndarray],
        row: int,
        col: int,
        rotation: int,
        delta_r: int,
    ) -> Optional[Tuple[int, int, int, bool]]:
        new_rotation = (rotation + delta_r) % 4
        if self._can_occupy(board_mask, piece_data, new_rotation, row, col, row_bounds):
            return row, col, new_rotation, False

        kicks = kick_table.get((rotation, new_rotation))
        if kicks is None:
            return None

        for dr, dc in kicks:
            new_row = row + int(dr)
            new_col = col + int(dc)
            if self._can_occupy(
                board_mask, piece_data, new_rotation, new_row, new_col, row_bounds
            ):
                return new_row, new_col, new_rotation, True
        return None

    def _shift_once(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        row: int,
        col: int,
        rotation: int,
        delta_col: int,
        row_bounds: np.ndarray,
    ) -> Optional[Tuple[int, int, int]]:
        new_col = col + delta_col
        if self._can_occupy(board_mask, piece_data, rotation, row, new_col, row_bounds):
            return row, new_col, rotation
        return None

    def _shift_das(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        row: int,
        col: int,
        rotation: int,
        direction: int,
        row_bounds: np.ndarray,
    ) -> Optional[Tuple[int, int, int]]:
        new_col = col
        moved = False
        while self._can_occupy(
            board_mask, piece_data, rotation, row, new_col + direction, row_bounds
        ):
            new_col += direction
            moved = True
        if not moved:
            return None
        return row, new_col, rotation

    def _soft_drop_state(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        row: int,
        col: int,
        rotation: int,
        board_rows: int,
        row_bounds: np.ndarray,
    ) -> Optional[Tuple[int, int, int]]:
        new_row = row
        moved = False
        while self._can_occupy(
            board_mask, piece_data, rotation, new_row + 1, col, row_bounds
        ):
            new_row += 1
            moved = True
            if new_row + row_bounds[rotation][1] >= board_rows - 1:
                break
        if not moved:
            return None
        return new_row, col, rotation

    def _hard_drop_row(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        rotation: int,
        row: int,
        col: int,
        row_bounds: np.ndarray,
    ) -> int:
        drop_row = row
        while self._can_occupy(
            board_mask, piece_data, rotation, drop_row + 1, col, row_bounds
        ):
            drop_row += 1
        return drop_row

    def _can_occupy(
        self,
        board_mask: np.ndarray,
        piece_data: Dict[str, np.ndarray],
        rotation: int,
        row: int,
        col: int,
        row_bounds: np.ndarray,
    ) -> bool:
        min_row, max_row = row_bounds[rotation]
        if row + min_row < 0 or row + max_row >= len(board_mask):
            return False
        min_col = piece_data["min_cols"][rotation]
        max_col = piece_data["max_cols"][rotation]
        if col + min_col < 0 or col + max_col >= self._BOARD_COLS:
            return False

        rel_rows = piece_data["row_offsets"][rotation]
        masks = piece_data["row_masks"][rotation]
        for rel_row, mask in zip(rel_rows, masks):
            board_row = row + int(rel_row)
            if col >= 0:
                shifted_mask = np.uint16(int(mask) << col)
            else:
                shifted_mask = np.uint16(int(mask) >> (-col))
            if board_mask[board_row] & shifted_mask:
                return False
        return True

    @staticmethod
    def _materialize_sequence(
        end_state: int,
        parents: np.ndarray,
        parent_keys: np.ndarray,
        is_hold: bool,
    ) -> np.ndarray:
        keys: list[int] = []
        state = end_state
        while parents[state] != -1 and parent_keys[state] != -1:
            keys.append(int(parent_keys[state]))
            state = int(parents[state])
        keys.reverse()

        sequence = [Keys.START]
        if is_hold:
            sequence.append(Keys.HOLD)
        sequence.extend(keys)
        sequence.append(Keys.HARD_DROP)
        return np.array(sequence, dtype=np.int32)
