from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from .RotationSystem import RotationSystem
from .Scorer import Scorer
from .Pieces import Piece, PieceType
from .Moves import Moves, Keys
import numpy as np
import random
import copy
from typing import List, Dict, Tuple

class PyTetrisEnv(py_environment.PyEnvironment):

    def __init__(self, queue_size, max_holes, seed, idx):

        self._hole_penalty = -0.5
        self._height_penalty = -0.3
        self._bumpy_penalty = -0.2
        self._death_penalty = -10.0

        self._max_holes = max_holes

        self._seed = seed

        self._random = random.Random(seed)

        self._board = np.zeros((24, 10), dtype=np.int32)

        self._rotation_system = RotationSystem()

        self._scorer = Scorer()

        self._last_heights = 0
        self._last_holes = 0
        self._last_bumpy = 0

        self._hold_piece = PieceType.N

        self._queue_size = queue_size
        self._valid_pieces = [piece for piece in PieceType if piece is not PieceType.N]
        self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._queue = self._fill_queue([])
        
        self._episode_ended = False

        self._observation_spec = {
            'board': array_spec.BoundedArraySpec(
                shape=(24, 10), dtype=np.int32, minimum=0, maximum=1, name='board'),
            'pieces': array_spec.BoundedArraySpec(
                shape=(2 + queue_size,), dtype=np.int32, minimum=0, maximum=7, name='pieces')
        }

        self._action_spec = {
            'hold': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=len(Moves._holds) - 1, name='hold'),
            'standard': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=len(Moves._standards) - 1, name='standard'),
            'spin': array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=len(Moves._spins) - 1, name='spin')
        }

        self._reward_spec = {
            'attack': array_spec.ArraySpec(
                shape=(), dtype=np.float32, name='attack'),
            'height_penalty': array_spec.ArraySpec(
                shape=(), dtype=np.float32, name='height_penalty'),
            'hole_penalty': array_spec.ArraySpec(
                shape=(), dtype=np.float32, name='hole_penalty'),
            'bumpy_penalty': array_spec.ArraySpec(
                shape=(), dtype=np.float32, name='bumpy_penalty'),
            'death_penalty': array_spec.ArraySpec(
                shape=(), dtype=np.float32, name='death_penalty')
        }

        print(f"Initialized Env {idx}", flush=True)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec


    def _reset(self):

        self._seed += 1

        self._random = random.Random(self._seed)

        self._board[:] = 0

        self._scorer.reset()

        self._last_heights = 0
        self._last_holes = 0
        self._last_bumpy = 0

        self._hold_piece = PieceType.N

        self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
        self._active_piece = self._spawn_piece(self._next_bag.pop(0))
        self._queue = self._fill_queue([])

        self._episode_ended = False

        observation = self._create_observation()

        return ts.restart(observation=observation,
                          reward_spec=self._reward_spec)
    
    def _step(self, action: Dict[str, int]):
        """
        `_lock_piece` does not move piece to the bottom, and only tries
        locking at the current location. Action sequences all already end in
        hard drop."
        """
        if self._episode_ended:
            return self.reset()

        (top_out, attack, board, active_piece,
         hold_piece, queue) = self._execute_action(self._board, self._active_piece,
                                                   self._hold_piece, self._queue, action)
        
        # Get board stats and compute supplementary rewards
        sum_heights, sum_holes, sum_bumpy = self._board_stats(board)
        height_penalty = self._height_penalty * (sum_heights - self._last_heights)
        hole_penalty = self._hole_penalty * (sum_holes - self._last_holes)
        bumpy_penalty = self._bumpy_penalty * (sum_bumpy - self._last_bumpy)

        exceeded_holes = sum_holes > self._max_holes if self._max_holes is not None else False
        self._episode_ended = top_out or exceeded_holes

        queue = self._fill_queue(queue)

        # Update state
        self._board = board
        self._active_piece = active_piece
        self._hold_piece = hold_piece
        self._queue = queue        

        self._last_heights = sum_heights
        self._last_holes = sum_holes
        self._last_bumpy = sum_bumpy

        observation = self._create_observation()

        reward = {
            'attack': np.array(attack),
            'height_penalty': np.array(height_penalty),
            'hole_penalty': np.array(hole_penalty),
            'bumpy_penalty': np.array(bumpy_penalty),
            'death_penalty': np.array(self._death_penalty) if self._episode_ended else np.array(0.0, dtype=np.float32)
        }

        if self._episode_ended:
            return ts.termination(observation=observation,
                                  reward=reward)
        else:
            return ts.transition(observation=observation,
                                 reward=reward)

    def _execute_action(self, board: np.ndarray, active_piece: Piece, hold_piece: PieceType,
                        queue: List[PieceType], action: Dict[str, int]) -> Tuple[bool, float, Piece, np.ndarray, List[PieceType]]:
        
        # Avoid modifying original state
        board = copy.deepcopy(board)
        active_piece = copy.deepcopy(active_piece)
        hold_piece = copy.deepcopy(hold_piece)
        queue = copy.deepcopy(queue)

        # Convert action to key sequence        
        key_sequence = self._convert_to_keys(action)
        clears = 0
        can_hold = True
        for key in key_sequence:
            if key == Keys.HOLD:
                can_hold, active_piece, hold_piece, queue = self._try_hold(can_hold, active_piece, hold_piece, queue)
            elif key in Keys.key_vectors.keys():
                key_vector = Keys.key_vectors[key]
                active_piece = self._try_key_vector(key_vector, active_piece, board)

                if key == Keys.HARD_DROP:
                    clears, top_out, active_piece, board, queue = self._lock_piece(active_piece, board, queue)

            attack = self._scorer.judge(active_piece, board, key, clears)

        return top_out, attack, board, active_piece, hold_piece, queue

    def _spawn_piece(self, piece_type: PieceType) -> Piece:
        # All pieces spawn 3 cells from the left on a default board
        # For the O piece, this is actually 4 cells including the padding
        spawn_loc = np.array([0, 3], np.int32)
        cells = self._rotation_system.orientations[piece_type][0]

        return Piece(piece_type=piece_type,
                     loc=spawn_loc,
                     r=0,
                     cells=cells)

    def _create_observation(self):
        pieces = [self._active_piece.piece_type, self._hold_piece] + self._queue
        pieces = np.array([piece.value for piece in pieces], dtype=np.int32)

        observation = {
            'board': self._board,
            'pieces': pieces
        }

        return observation

    def _convert_to_keys(self, action: Dict[str, int]) -> List[int]:
        hold = Moves._holds[action['hold']]
        standard = Moves._standards[action['standard']]
        spin = Moves._spins[action['spin']]

        key_sequence = hold + standard + spin + [Keys.HARD_DROP]

        return key_sequence

    def _try_key_vector(self, key_vector: np.ndarray, active_piece: Piece, board: np.ndarray) -> Piece:
        # Key vector is delta [row, column, rotation]

        delta_loc = key_vector[:-1]
        delta_r = key_vector[-1]

        if delta_r:
            active_piece = self._rotate(delta_r, active_piece=active_piece, board=board)
        else:
            active_piece = self._move(delta_loc, active_piece=active_piece, board=board)

        return active_piece

    def _rotate(self, try_delta_r: int, active_piece: Piece, board: np.ndarray) -> Piece:
        new_r = (active_piece.r + try_delta_r) % 4
        cells = self._rotation_system.orientations[active_piece.piece_type][new_r]

        if not self._rotation_system.overlaps(cells=cells, loc=active_piece.loc, board=board):
            # Applying rotation doesn't overlap, so do it
            active_piece.r = new_r
            active_piece.delta_r = try_delta_r

            active_piece.delta_loc = np.zeros((2,), dtype=np.int32)
            active_piece.cells = cells

        else:
            # Overlaps, so try kicks
            kick_table = (self._rotation_system.i_kicks
                          if active_piece.piece_type == PieceType.I
                          else self._rotation_system.kicks)
            self._rotation_system.kick_piece(kick_table, active_piece, cells, new_r, try_delta_r, board)

        return active_piece

    def _move(self, try_delta_loc: np.ndarray, active_piece: Piece, board: np.ndarray) -> Piece:
        # Vertical movement
        if try_delta_loc[0]:
            # -1 = up (NEVER)
            # +1 = down
            direction = np.sign(try_delta_loc[0])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_y in range(0, try_delta_loc[0] + direction, direction):
                new_delta_loc = np.array([delta_y, 0], dtype=np.int32)
                if self._rotation_system.overlaps(cells=active_piece.cells, loc=active_piece.loc + new_delta_loc, board=board):
                    break
                else:
                    delta_loc = new_delta_loc

        # Horizontal movement
        elif try_delta_loc[1]:
            # -1 = left
            # +1 = right
            direction = np.sign(try_delta_loc[1])
            delta_loc = np.array([0, 0], dtype=np.int32)
            # Try moving the full amount, stop at first collision
            for delta_x in range(0, try_delta_loc[1] + direction, direction):
                new_delta_loc = np.array([0, delta_x], dtype=np.int32)
                if self._rotation_system.overlaps(cells=active_piece.cells, loc=active_piece.loc + new_delta_loc, board=board):
                    break
                else:
                    delta_loc = new_delta_loc

        else:
            # Delta would have to be [0, 0] to get here
            raise ValueError(f"Should never reach this! Delta: {try_delta_loc}")

        active_piece.loc += delta_loc
        active_piece.delta_loc = delta_loc
        active_piece.delta_r = 0

        return active_piece

    def _try_hold(self, can_hold: bool, active_piece: Piece,
                  hold_piece: PieceType, queue: List[PieceType]) -> Tuple[bool, Piece, PieceType, List[PieceType]]:
        # Using `_can_hold` is unnecessary for this implementation
        # since only valid action sequences exist and none include pressing
        # the hold key twice. This is included in case of future changes.
        if can_hold:
            if hold_piece == PieceType.N:
                # No piece held, so this is the first time holding
                hold_piece = active_piece.piece_type
                active_piece = self._spawn_piece(queue.pop(0))
            else:
                active_piece, hold_piece = (self._spawn_piece(hold_piece),
                                            active_piece.piece_type)
            can_hold = False

        return can_hold, active_piece, hold_piece, queue

    def _lock_piece(self, active_piece: Piece,
                    board: np.ndarray, queue: List[PieceType]) -> Tuple[bool, Piece, np.ndarray, List[PieceType]]:
        # DOES NOT MOVE PIECE TO THE BOTTOM
        # This method assumes the piece is already in a settled position.
        cell_coords = active_piece.cells + active_piece.loc

        board[cell_coords[:, 0], cell_coords[:, 1]] = 1

        board, clears = self._clear_lines(board)

        active_piece = self._spawn_piece(queue.pop(0))

        top_out = np.any(board[:4] != 0)

        return clears, top_out, active_piece, board, queue

    def _fill_queue(self, queue: List[PieceType]) -> List[PieceType]:
        while len(queue) < self._queue_size:
            queue.append(self._next_bag.pop(0))
            if len(self._next_bag) == 0:
                self._next_bag = self._random.sample(self._valid_pieces, len(self._valid_pieces))
        return queue

    def _clear_lines(self, board: np.ndarray) -> Tuple[np.ndarray, int]:
        clears = 0
        for i, row in enumerate(board):
            if np.all(row != 0):
                clears += 1
                board[0] = 0
                board[1:i+1] = board[:i]
        return board, clears

    def _get_heights(self, board: np.ndarray) -> np.ndarray:
        # Get heights of each column in the board
        height_matrix = np.arange(board.shape[0], 0, -1)[..., None]
        heights = np.max(board * height_matrix, axis=0)

        return heights

    def _get_holes(self, board: np.ndarray, heights: np.ndarray) -> np.ndarray:
        # Count holes in the board
        holes = heights - np.sum(board, axis=0)
        return holes

    def _get_bumpy(self, heights: np.ndarray) -> np.ndarray:
        # Get bumpiness of the board
        bumpy = np.abs(heights[:-1] - heights[1:])
        return bumpy

    def _board_stats(self, board: np.ndarray) -> Tuple[int, int, int]:
        # Get total heights of the board
        heights = self._get_heights(board)
        sum_heights = np.sum(heights)

        # Get number of holes in the board
        holes = self._get_holes(board, heights)
        sum_holes = np.sum(holes)
        
        # Get bumpiness of the board
        bumpy = self._get_bumpy(heights)
        sum_bumpy = np.sum(bumpy)

        # Return heights, holes, and bumpy
        return sum_heights, sum_holes, sum_bumpy