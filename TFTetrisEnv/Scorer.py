from enum import Enum
from .Pieces import Piece, PieceType
from .Moves import Keys
import numpy as np

class Spins(Enum):
    # TODO
    # More spins
    NO_SPIN = 0
    T_SPIN_MINI = 1
    T_SPIN = 2

class Scorer():

    def __init__(self):

        self._b2b = 0
        self._combo = 0
        self._spin = Spins.NO_SPIN
        
        self._pf_attack = np.array([0, 5, 6, 7, 9], dtype=np.float32)
        self._ts_attack = np.array([0, 2, 4, 6, 0], dtype=np.float32)
        self._tsm_attack = np.array([0, 0, 1, 2, 0], dtype=np.float32)
        self._clear_attack = np.array([0, 0, 1, 2, 4], dtype=np.float32)

        self._step_reward = np.array(0.1, dtype=np.float32)
        self._hole_penalty = np.array(-0.025, dtype=np.float32)
        self._max_hole_penalty = np.array(-0.099, dtype=np.float32)
        self._death_penalty = np.array(-1.0, dtype=np.float32)

    def reset(self):
        self._b2b = 0
        self._combo = 0
        self._spin = Spins.NO_SPIN
    
    def _get_holes(self, board: np.ndarray, heights: np.ndarray) -> int:
        # Count holes in the board
        holes = np.sum(heights - np.sum(board, axis=0))
        return holes

    def _get_heights(self, board: np.ndarray) -> np.ndarray:
        # Get heights of each column in the board

        height_matrix = np.arange(board.shape[0]-1, -1, -1)[..., None]
        heights = np.max(board * height_matrix, axis=0)

        return heights

    def _supp_reward(self, board: np.ndarray, ended: bool) -> float:
        # TODO
        # More supplemental rewards

        # Get heights of each column in the board
        heights = self._get_heights(board)

        # Get number of holes in the board
        holes = self._get_holes(board, heights)

        # Compute rewards
        hole_penalty = np.maximum(holes * self._hole_penalty, self._max_hole_penalty)
        death_penalty = self._death_penalty if ended else 0
        return self._step_reward, hole_penalty, death_penalty

    def judge(self, piece: Piece, board: np.ndarray, key: int, clears: int, ended: bool) -> float:

        # TODO
        # all-mini+ immobile spin detection

        attack = 0

        if piece.piece_type == PieceType.T:
            if piece.delta_r != 0:
                # Check for corners clockwise from top-left
                # Corners are filled if cell is nonzero or out of board
                corner_cells = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=np.int32)

                # Check out of bounds corners
                corner_inds = piece.loc + corner_cells # 4, 2
                corners = np.any([np.any(corner_inds >= board.shape, axis=-1),
                                  np.any(corner_inds < 0, axis=-1)], axis=0)

                # Check in bounds corners
                corner_inds = np.maximum(0, np.minimum(corner_inds, [board.shape[0] - 1,
                                                                     board.shape[1] - 1]))
                corners = np.any([corners,
                                  board[corner_inds[:, 0],
                                        corner_inds[:, 1]] != 0], axis=0)

                # Find back cell in same order as corners. Corners[back] is
                # is the corner anticlockwise relative to the cell
                back = None
                for i, cell in enumerate(np.array([[0, 1], [1, 2], [2, 1], [1, 0]], np.int32)):
                    if not np.any(np.all(piece.cells == cell, axis=-1)):
                        back = i
                        break

                front_corners = np.sum([corners[(back + 2) % 4], corners[(back + 3) % 4]])
                back_corners = np.sum([corners[(back + 0) % 4], corners[(back + 1) % 4]])

                if front_corners == 2 and back_corners >= 1:
                    # Proper T spin
                    self._spin = Spins.T_SPIN

                elif front_corners == 1 and back_corners == 2:
                    if np.sum(np.abs(piece.delta_loc)) > 2:
                        # Piece was kicked far enough, so this is also a T spin
                        self._spin = Spins.T_SPIN
                    else:
                        # Not kicked far, T spin mini
                        self._spin = Spins.T_SPIN_MINI
                else:
                    # Fails to pass corner rules, not a T spin
                    self._spin = Spins.NO_SPIN
            elif np.any(piece.delta_loc != 0) or key == Keys.HOLD:
                    # Remove any active spins on HOLD or if movement
                    # is caused by anything other than a kick
                    self._spin = Spins.NO_SPIN

        if key == Keys.HARD_DROP:

            perfect_clear = np.all(board == 0)

            if clears:
                if self._spin != Spins.NO_SPIN or clears == 4 or perfect_clear:
                    self._b2b += 1
                else:
                    self._b2b = 0

                # TODO
                # Combo table for tetrio
                self._combo += 1
            else:
                self._combo = 0

            if perfect_clear:
                attack += self._pf_attack[clears]

            elif self._spin == Spins.T_SPIN:
                attack += self._ts_attack[clears]

            elif self._spin == Spins.T_SPIN_MINI:
                attack += self._tsm_attack[clears]

            else:
                attack += self._clear_attack[clears]

            self._spin = Spins.NO_SPIN

        step_reward, hole_penalty, death_penalty = self._supp_reward(board, ended)

        return attack, step_reward, hole_penalty, death_penalty
