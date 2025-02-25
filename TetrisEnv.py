from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from RotationSystem import RotationSystem
from Pieces import Piece, PieceType
from Moves import Moves, Keys
import numpy as np
import random

class TetrisPyEnv(py_environment.PyEnvironment):
    
    def __init__(self, queue_size, seed):
        
        self._random = random.Random(seed)
        
        self._queue_size = queue_size
        
        self._board = np.zeros((24, 10), dtype=np.float32)
        self._hold_piece = PieceType.N
        self._can_hold = True
        self._rotation_system = RotationSystem(board=self._board)
        
        self._valid_pieces = [piece for piece in PieceType if piece is not PieceType.N]
        self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._fill_queue(True)
        
        self._episode_ended = False
        
        self._observation_spec = {
            'board': array_spec.BoundedArraySpec(
                shape=(24, 10), dtype=np.float32, minimum=0.0, maximum=1.0, name='board'),
            'pieces': array_spec.BoundedArraySpec(
                shape=(2 + queue_size,), dtype=np.int32, minimum=0, maximum=8, name='board')
        }
        
        self._action_spec = {
            'hold': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=1, name='hold'),
            'standard': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=36, name='standard'),
            'spin': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=7, name='spin')
        }
        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        # Empty all cells
        self._board[:] = 0
        self._hold_piece = PieceType.N
        self._can_hold = True
        
        self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._fill_queue(True)
        
        self._episode_ended = False
    
    def _convert_to_keys(self, action: dict[str, int]):
        key_sequence = Moves.to_keys[(action['hold'],
                                      action['standard'],
                                      action['spin'])]
        
        return key_sequence
    
    def _move(self, delta: np.ndarray):
        # Vertical movement
        if delta[0]:
            # -1 = up (NEVER)
            # +1 = down
            direction = np.sign(delta[0])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_y in range(0, delta[0] + direction, direction):
                new_delta_loc = np.array([delta_y, 0], dtype=np.int32)
                if self._rotation_system.overlaps(cells=self._active_piece.cells, loc=self._active_piece.loc + new_delta_loc):
                    break
                else:
                    delta_loc = new_delta_loc
                    
            self._active_piece.loc += delta_loc
        
        # Horizontal movement
        elif delta[1]:
            # -1 = left
            # +1 = right
            direction = np.sign(delta[1])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_x in range(0, delta[1] + direction, direction):
                new_delta_loc = np.array([0, delta_x], dtype=np.int32)
                if self._rotation_system.overlaps(cells=self._active_piece.cells, loc=self._active_piece.loc + new_delta_loc):
                    break
                else:
                    delta_loc = new_delta_loc
                    
            self._active_piece.loc += delta_loc
            
        else:
            # Delta would have to be [0, 0] to get here
            raise ValueError(f"Should never reach this! Delta: {delta}")
            
    
    def _try_key_vector(self, key_vector: np.ndarray):
        # Key vector is delta [row, column, rotation]
        
        delta = key_vector[:-1]
        rotation = key_vector[-1]
        
        if rotation:
            self._rotation_system.rotate(self._active_piece, rotation)
        else:
            self._move(delta)
    
    def _clear_lines(self):
        for i, row in enumerate(self._board):
            if all(row != 0):
                self._board[0] = 0
                self._board[1:i+1] = self._board[:i]
    
    def _lock_piece(self) -> bool:
        # DOES NOT MOVE PIECE TO THE BOTTOM
        # This method assumes the piece is already in a settled position.
        # Returns True if piece locked successfully, False if died
        cell_coords = self._active_piece.cells + self._active_piece.loc
        rows, cols = cell_coords.T
        self._board[rows, cols] = 1
        
        self._clear_lines()
        
        if np.any(self._board[:4] != 0):
            return False
        else:
            self._active_piece = self._spawn_piece(self._queue.pop(0))
            return True
    
    def _try_hold(self):
        # Using `_can_hold` is unnecessary for this implementation
        # since only valid action sequences exist and none include pressing
        # the hold key twice. This is included in case of future changes.
        if self._can_hold:
            if self._hold_piece == PieceType.N:
                # No piece held, so this is the first one
                self._hold_piece = self._active_piece.piece_type
                self._active_piece = self._spawn_piece(self._queue.pop(0))
            else:
                self._active_piece, self._hold_piece = (self._spawn_piece(self._hold_piece),
                                                        self._active_piece.piece_type)
            self._can_hold = False
    
    def _fill_queue(self, init: bool=False) -> list[PieceType]:
        if init:
            self._queue = []
        while len(self._queue) < self._queue_size:
            self._queue.append(self._next_bag.pop(0))
            if len(self._next_bag) == 0:
                self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
    
    def _spawn_piece(self, piece_type: PieceType) -> Piece:
        # All pieces spawn 3 cells from the left on a default board
        # For the O piece, this is actually 4 cells including the padding
        spawn_loc = np.array([0, 3], np.int32)
        cells = self._rotation_system.orientations[piece_type][0]
        
        return Piece(piece_type=piece_type,
                     loc=spawn_loc,
                     r=0,
                     cells=cells)
    
    def _step(self, action: dict[str, int]):
        # `_lock_piece` does not move piece to the bottom, and only tries
        # locking at the current location. Action sequences all already end in
        # hard drop.
        key_sequence = self._convert_to_keys(action)
        
        for key in key_sequence:
            if key in Keys.key_vectors.keys():
                key_vector = Keys.key_vectors[key]
                self._try_key_vector(key_vector)
            elif key == Keys.hold:
                self._try_hold()
        
        if self._lock_piece():
            # Locked successfully, cycle queue and allow next hold
            self._fill_queue()
            self._can_hold = True
        else:
            self._reset()
            
            