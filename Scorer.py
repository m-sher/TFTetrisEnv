from enum import Enum
from Pieces import Piece, PieceType
from Moves import Keys
import numpy as np

class Spins(Enum):
    # TODO
    # More spins
    NO_SPIN = 0
    T_SPIN_MINI = 1
    T_SPIN = 2

class Scorer():
    
    def __init__(self):

        self.reset()
        
    def reset(self):
        self._b2b = 0
        self._combo = 0 
        
        self._spin = Spins.NO_SPIN
        
    def judge(self, piece: Piece, board: np.ndarray, key, clears) -> float:
        
        # TODO
        # all-mini+ immobile spin detection
        
        reward = 0
        
        if piece.piece_type == PieceType.T:
            if piece.delta_r != 0:
                # Check for corners clockwise from top-left
                # Corners are filled if cell is nonzero or out of board
                corner_cells = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=np.int32)
                corner_inds = piece.loc + corner_cells
                corners = (np.any(corner_inds >= board.shape) or 
                           np.any(corner_inds < 0) or
                           board[corner_inds[:, 0], corner_inds[:, 1]] != 0)
                
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
                reward += [0, 5, 6, 7, 9][clears]
            
            elif self._spin == Spins.T_SPIN:
                reward += [0, 2, 4, 6, 0][clears]
                
            elif self._spin == Spins.T_SPIN_MINI:
                reward += [0, 0, 1, 2, 0][clears]
            
            else:
                reward += [0, 0, 1, 2, 4][clears]                      
                    
            self._spin = Spins.NO_SPIN
                
        return reward