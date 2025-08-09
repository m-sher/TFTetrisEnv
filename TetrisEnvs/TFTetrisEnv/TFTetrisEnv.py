import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from .TFRotationSystem import TFRotationSystem
from .TFScorer import TFScorer
from .TFPieces import TFPieceType
from .TFMoves import TFKeys
from typing import Tuple


class TFTetrisEnv(py_environment.PyEnvironment):
    def __init__(self, queue_size, max_holes, max_len=9, seed=[0, 0]):
        self._hole_penalty = tf.constant(-0.5, dtype=tf.float32)
        self._height_penalty = tf.constant(-0.1, dtype=tf.float32)
        self._skyline_penalty = tf.constant(-0.1, dtype=tf.float32)
        self._bumpy_penalty = tf.constant(-0.2, dtype=tf.float32)
        self._death_penalty = tf.constant(-10.0, dtype=tf.float32)

        self._queue_size = tf.constant(queue_size, tf.int64)
        self._max_holes = tf.constant(max_holes, tf.float32)
        self._max_len = tf.constant(max_len, tf.int64)
        self._seed = tf.constant(seed, tf.int32)

        self._spawn_loc = tf.constant([0, 3], tf.int64)

        self._rotation_system = TFRotationSystem()

        self._scorer = TFScorer()

        self._board = tf.Variable(
            name="board",
            initial_value=tf.zeros((24, 10), dtype=tf.float32),
            dtype=tf.float32,
        )

        self._last_heights = tf.Variable(
            name="last_heights",
            initial_value=tf.zeros((), dtype=tf.float32),
            dtype=tf.float32,
        )
        self._last_holes = tf.Variable(
            name="last_holes",
            initial_value=tf.zeros((), dtype=tf.float32),
            dtype=tf.float32,
        )
        self._last_skyline = tf.Variable(
            name="last_skyline",
            initial_value=tf.zeros((), dtype=tf.float32),
            dtype=tf.float32,
        )
        self._last_bumpy = tf.Variable(
            name="last_bumpy",
            initial_value=tf.zeros((), dtype=tf.float32),
            dtype=tf.float32,
        )

        self._hold_piece = tf.Variable(
            name="hold_piece",
            initial_value=tf.fill(dims=(), value=TFPieceType.N),
            dtype=tf.int64,
        )

        all_pieces = tf.range(1, 8, dtype=tf.int64)
        self._next_bag = tf.Variable(
            name="next_bag",
            initial_value=self._stateless_shuffle(all_pieces),
            dtype=tf.int64,
        )
        self._bag_ind = tf.Variable(
            name="bag_ind", initial_value=tf.zeros((), dtype=tf.int64), dtype=tf.int64
        )

        self._active_piece_type = tf.Variable(
            name="active_piece_type",
            initial_value=tf.fill(dims=(), value=TFPieceType.N),
            dtype=tf.int64,
        )
        self._active_loc = tf.Variable(
            name="active", initial_value=tf.zeros((2,), dtype=tf.int64), dtype=tf.int64
        )
        self._active_delta_loc = tf.Variable(
            name="active", initial_value=tf.zeros((2,), dtype=tf.int64), dtype=tf.int64
        )
        self._active_r = tf.Variable(
            name="active", initial_value=tf.zeros((), dtype=tf.int64), dtype=tf.int64
        )
        self._active_delta_r = tf.Variable(
            name="active", initial_value=tf.zeros((), dtype=tf.int64), dtype=tf.int64
        )
        self._active_cells = tf.Variable(
            name="active",
            initial_value=tf.zeros((4, 2), dtype=tf.int64),
            dtype=tf.int64,
        )

        to_spawn = tf.gather(self._next_bag, self._bag_ind)
        self._spawn_piece(to_spawn)
        self._bag_ind.assign_add(tf.ones_like(self._bag_ind))

        self._queue = tf.Variable(
            name="queue",
            initial_value=tf.fill(dims=(self._queue_size,), value=TFPieceType.N),
            dtype=tf.int64,
        )
        self._fill_queue()

        self._episode_ended = tf.Variable(
            name="episode_ended", initial_value=False, dtype=tf.bool
        )

        self._observation_spec = {
            "board": tensor_spec.BoundedTensorSpec(
                shape=(24, 10, 1),
                dtype=tf.float32,
                minimum=0.0,
                maximum=1.0,
                name="board",
            ),
            "pieces": tensor_spec.BoundedTensorSpec(
                shape=(2 + self._queue_size,),
                dtype=tf.int64,
                minimum=0,
                maximum=7,
                name="pieces",
            ),
        }

        self._action_spec = tensor_spec.BoundedTensorSpec(
            shape=(9,), dtype=tf.int64, minimum=0, maximum=12, name="key_sequence"
        )

        self._reward_spec = {
            "attack": tensor_spec.TensorSpec(shape=(), dtype=tf.float32, name="attack"),
            "height_penalty": tensor_spec.TensorSpec(
                shape=(), dtype=tf.float32, name="height_penalty"
            ),
            "hole_penalty": tensor_spec.TensorSpec(
                shape=(), dtype=tf.float32, name="hole_penalty"
            ),
            "skyline_penalty": tensor_spec.TensorSpec(
                shape=(), dtype=tf.float32, name="skyline_penalty"
            ),
            "bumpy_penalty": tensor_spec.TensorSpec(
                shape=(), dtype=tf.float32, name="bumpy_penalty"
            ),
            "death_penalty": tensor_spec.TensorSpec(
                shape=(), dtype=tf.float32, name="death_penalty"
            ),
        }

        super(TFTetrisEnv, self).__init__()

        self._recent_time_step = self.reset()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def reward_spec(self):
        return self._reward_spec

    def _current_time_step(self):
        return self._recent_time_step

    @tf.function(jit_compile=False)
    def _reset(self):
        with tf.name_scope("Reset"):
            self._board.assign(tf.zeros((24, 10), dtype=tf.float32))

            self._last_heights.assign(tf.zeros((), dtype=tf.float32))
            self._last_holes.assign(tf.zeros((), dtype=tf.float32))
            self._last_skyline.assign(tf.zeros((), dtype=tf.float32))
            self._last_bumpy.assign(tf.zeros((), dtype=tf.float32))

            self._hold_piece.assign(TFPieceType.N)

            self._next_bag.assign(
                self._stateless_shuffle(tf.range(1, 8, dtype=tf.int64))
            )
            self._bag_ind.assign(tf.zeros((), dtype=tf.int64))

            self._spawn_piece(self._next_bag[self._bag_ind])
            self._bag_ind.assign_add(1)

            self._queue.assign(tf.fill((self._queue_size,), value=TFPieceType.N))
            self._fill_queue()

            self._episode_ended.assign(False)

            observation = self._create_observation()

            reward = {
                name: tf.zeros(spec.shape, dtype=spec.dtype)
                for name, spec in self._reward_spec.items()
            }
            self._recent_time_step = ts.TimeStep(
                step_type=ts.StepType.FIRST,
                reward=reward,
                discount=1.0,
                observation=observation,
            )

            return self._recent_time_step

    @tf.function(
        jit_compile=False,
        input_signature=[
            tf.TensorSpec(shape=(9,), dtype=tf.int64, name="key_sequence")
        ],
    )
    def _step(self, key_sequence):
        """
        `_lock_piece` does not move piece to the bottom, and only tries
        locking at the current location. Action sequences all already end in
        hard drop."
        """
        with tf.name_scope("Step"):

            def do_step():
                top_out, attack = self._execute_action(key_sequence)

                # Get board stats and compute supplementary rewards
                heights_val, holes_val, skyline_val, bumpy_val = self._board_stats()
                height_penalty = self._height_penalty * (
                    heights_val - self._last_heights
                )
                hole_penalty = self._hole_penalty * (holes_val - self._last_holes)
                skyline_penalty = self._skyline_penalty * (
                    skyline_val - self._last_skyline
                )
                bumpy_penalty = self._bumpy_penalty * (bumpy_val - self._last_bumpy)

                exceeded_holes = holes_val > self._max_holes

                self._episode_ended.assign(tf.logical_or(top_out, exceeded_holes))

                self._last_heights.assign(heights_val)
                self._last_holes.assign(holes_val)
                self._last_skyline.assign(skyline_val)
                self._last_bumpy.assign(bumpy_val)

                observation = self._create_observation()

                death_penalty = tf.cond(
                    self._episode_ended,
                    lambda: self._death_penalty,
                    lambda: tf.constant(0.0, dtype=tf.float32),
                )

                reward = {
                    "attack": tf.identity(attack),
                    "height_penalty": tf.identity(height_penalty),
                    "hole_penalty": tf.identity(hole_penalty),
                    "skyline_penalty": tf.identity(skyline_penalty),
                    "bumpy_penalty": tf.identity(bumpy_penalty),
                    "death_penalty": tf.identity(death_penalty),
                }

                time_step = tf.cond(
                    self._episode_ended,
                    lambda: ts.termination(observation=observation, reward=reward),
                    lambda: ts.transition(observation=observation, reward=reward),
                )
                return time_step

            self._recent_time_step = tf.cond(self._episode_ended, self.reset, do_step)

            return self._recent_time_step

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=(9,), dtype=tf.int64, name="key_sequence")
        ],
    )
    def _execute_action(self, key_sequence) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.name_scope("ExecuteAction"):

            def single_key(sequence_ind, top_out, attack):
                key = key_sequence[sequence_ind]

                top_out, attack = self._press_key(key, top_out, attack)

                return sequence_ind + 1, top_out, attack

            sequence_ind = tf.zeros((), dtype=tf.int64)
            top_out = tf.constant(False, dtype=tf.bool)
            attack = tf.zeros((), dtype=tf.float32)

            sequence_ind, top_out, attack = tf.while_loop(
                lambda sequence_ind, top_out, attack: sequence_ind < self._max_len,
                single_key,
                [sequence_ind, top_out, attack],
                parallel_iterations=1,
            )

            return top_out, attack

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=(), dtype=tf.int64, name="key"),
            tf.TensorSpec(shape=(), dtype=tf.bool, name="top_out"),
            tf.TensorSpec(shape=(), dtype=tf.float32, name="attack"),
        ],
    )
    def _press_key(self, key, top_out, attack):
        with tf.name_scope("PressKey"):
            key_vector = tf.gather(TFKeys.key_vectors, key)

            tf.cond(
                key == TFKeys.HOLD,
                self._try_hold,
                lambda: self._try_key_vector(key_vector),
            )

            top_out, attack = tf.cond(
                key == TFKeys.HARD_DROP, self._lock_and_judge, lambda: (top_out, attack)
            )

            return top_out, attack

    @tf.function(jit_compile=True)
    def _lock_and_judge(self):
        clears, top_out = self._lock_piece()
        attack = self._scorer.judge(
            piece_type=self._active_piece_type,
            piece_loc=self._active_loc,
            piece_delta_loc=self._active_delta_loc,
            piece_delta_r=self._active_delta_r,
            piece_cells=self._active_cells,
            board=self._board,
            clears=clears,
        )
        return top_out, attack

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.int64, name="piece_type")],
    )
    def _spawn_piece(self, piece_type):
        # All pieces spawn 3 cells from the left on a default board
        # For the O piece, this is actually 4 cells including the padding
        with tf.name_scope("SpawnPiece"):
            self._active_piece_type.assign(piece_type)
            self._active_loc.assign(self._spawn_loc)
            self._active_delta_loc.assign(tf.zeros((2,), dtype=tf.int64))
            self._active_r.assign(tf.zeros((), dtype=tf.int64))
            self._active_delta_r.assign(tf.zeros((), dtype=tf.int64))
            self._active_cells.assign(
                tf.gather_nd(
                    self._rotation_system.orientations,
                    tf.stack([piece_type, tf.zeros_like(piece_type)], axis=-1),
                )
            )

    @tf.function(jit_compile=True)
    def _create_observation(self) -> dict[str, tf.Tensor]:
        with tf.name_scope("CreateObservation"):
            active_piece_type = tf.identity(self._active_piece_type)[None, ...]
            hold_piece = tf.identity(self._hold_piece)[None, ...]
            queue = tf.identity(self._queue)
            board = tf.identity(self._board)[..., None]

            pieces = tf.cast(
                tf.concat([active_piece_type, hold_piece, queue], axis=0), tf.int64
            )

            observation = {"board": board, "pieces": pieces}

            return observation

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.int64, name="key_vector")],
    )
    def _try_key_vector(self, key_vector):
        # Key vector is delta [row, column, rotation]
        with tf.name_scope("TryKeyVector"):
            try_delta_loc = key_vector[:-1]
            try_delta_r = key_vector[-1]

            tf.cond(
                try_delta_r != 0,
                lambda: self._rotate(try_delta_r),
                lambda: self._move(try_delta_loc),
            )

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(), dtype=tf.int64, name="try_delta_r")],
    )
    def _rotate(self, try_delta_r):
        with tf.name_scope("Rotate"):
            loc = tf.identity(self._active_loc)
            r = tf.identity(self._active_r)
            cells = tf.identity(self._active_cells)
            new_r = (self._active_r + try_delta_r) % 4
            new_cells = self._rotation_system.orientations[self._active_piece_type][
                new_r
            ]

            new_loc, new_delta_loc, new_r, new_delta_r, new_cells = tf.cond(
                self._rotation_system.overlaps(
                    cells=new_cells, loc=self._active_loc, board=self._board
                ),
                lambda: self._rotation_system.kick_piece(
                    piece_type=self._active_piece_type,
                    piece_loc=loc,
                    piece_delta_loc=self._active_delta_loc,
                    piece_r=r,
                    piece_cells=cells,
                    new_r=new_r,
                    new_delta_r=try_delta_r,
                    new_cells=new_cells,
                    board=self._board,
                ),
                lambda: (
                    loc,
                    tf.zeros((2,), dtype=tf.int64),
                    new_r,
                    try_delta_r,
                    new_cells,
                ),
            )
            self._active_loc.assign(new_loc)
            self._active_delta_loc.assign(new_delta_loc)
            self._active_r.assign(new_r)
            self._active_delta_r.assign(new_delta_r)
            self._active_cells.assign(new_cells)

    @tf.function(
        jit_compile=True,
        input_signature=[
            tf.TensorSpec(shape=(2,), dtype=tf.int64, name="try_delta_loc")
        ],
    )
    def _move(self, try_delta_loc):
        # Vertical movement
        with tf.name_scope("Move"):
            direction = tf.sign(try_delta_loc)
            delta_loc = tf.zeros((2,), dtype=tf.int64)

            def can_move(delta_loc):
                should_try = tf.logical_or(
                    tf.abs(delta_loc[0]) < tf.abs(try_delta_loc[0]),
                    tf.abs(delta_loc[1]) < tf.abs(try_delta_loc[1]),
                )
                new_loc = self._active_loc + delta_loc + direction
                would_overlap = self._rotation_system.overlaps(
                    cells=self._active_cells, loc=new_loc, board=self._board
                )

                return tf.logical_and(should_try, tf.logical_not(would_overlap))

            delta_loc = tf.while_loop(
                can_move,
                lambda delta_loc: delta_loc + direction,
                [delta_loc],
                parallel_iterations=1,
            )[0]
            # If I eventually switch to free key sequences, I need to only set delta_r to 0
            # when the piece moved, not every time a non-rotation key is pressed
            self._active_loc.assign_add(delta_loc)
            self._active_delta_loc.assign(delta_loc)
            delta_r = tf.identity(self._active_delta_r)
            new_delta_r = tf.cond(
                tf.reduce_any(delta_loc != 0),
                lambda: tf.zeros((), dtype=tf.int64),
                lambda: delta_r,
            )
            self._active_delta_r.assign(new_delta_r)

    @tf.function(jit_compile=True)
    def _try_hold(self):
        with tf.name_scope("TryHold"):
            # Not checking if holding is locked since all actions are valid
            to_hold = tf.identity(self._active_piece_type)
            popped, new_queue, new_bag_ind, new_next_bag = self._queue_pop(self._queue)

            to_spawn, new_queue, bag_ind, next_bag = tf.cond(
                self._hold_piece == TFPieceType.N,
                lambda: (popped, new_queue, new_bag_ind, new_next_bag),
                lambda: (
                    tf.identity(self._hold_piece),
                    tf.identity(self._queue),
                    tf.identity(self._bag_ind),
                    tf.identity(self._next_bag),
                ),
            )

            self._hold_piece.assign(to_hold)
            self._spawn_piece(to_spawn)
            self._queue.assign(new_queue)
            self._bag_ind.assign(bag_ind)
            self._next_bag.assign(next_bag)

    @tf.function(jit_compile=True)
    def _lock_piece(self) -> Tuple[tf.Tensor, tf.Tensor]:
        # DOES NOT MOVE PIECE TO THE BOTTOM
        # This method assumes the piece is already in a settled position.
        with tf.name_scope("LockPiece"):
            cell_coords = self._active_cells + self._active_loc

            # indices -> (4, 2)
            # updates -> (4,)
            new_board = tf.tensor_scatter_nd_update(
                self._board,
                cell_coords,
                tf.ones((tf.shape(cell_coords)[0],), dtype=tf.float32),
            )
            self._board.assign(new_board)

            clears = self._clear_lines()

            to_spawn, new_queue, bag_ind, next_bag = self._queue_pop(self._queue)
            self._spawn_piece(to_spawn)
            self._queue.assign(new_queue)
            self._bag_ind.assign(bag_ind)
            self._next_bag.assign(next_bag)

            top_out = tf.reduce_any(self._board[:4] != 0.0)

            return clears, top_out

    @tf.function(jit_compile=True)
    def _fill_queue(self):
        # Should only be called on an empty queue
        with tf.name_scope("FillQueue"):

            def hard_pop(queue):
                popped, new_queue, bag_ind, next_bag = self._queue_pop(queue)
                self._bag_ind.assign(bag_ind)
                self._next_bag.assign(next_bag)
                return new_queue

            queue = tf.identity(self._queue)
            new_queue = tf.while_loop(
                lambda queue: tf.reduce_sum(tf.cast(queue == TFPieceType.N, tf.float32))
                > 0,
                hard_pop,
                [queue],
                parallel_iterations=1,
            )[0]
            self._queue.assign(new_queue)

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(5,), dtype=tf.int64, name="queue")],
    )
    def _queue_pop(self, queue) -> tf.Tensor:
        # Generate the next bag if the current one is empty
        with tf.name_scope("QueuePop"):
            # Pop the first element of the queue
            popped = tf.identity(queue[0])
            bag_piece, bag_ind, next_bag = self._get_from_bag()
            new_queue = tf.concat([queue[1:], [bag_piece]], axis=0)

            return popped, new_queue, bag_ind, next_bag

    @tf.function(jit_compile=True)
    def _get_from_bag(self):
        # Get the next piece from the bag
        with tf.name_scope("GetFromBag"):
            bag_ind, next_bag = tf.cond(
                self._bag_ind >= tf.shape(self._next_bag, out_type=tf.int64)[0],
                lambda: (
                    tf.zeros((), dtype=tf.int64),
                    self._stateless_shuffle(self._next_bag),
                ),
                lambda: (self._bag_ind, self._next_bag),
            )
            bag_piece = tf.identity(next_bag[bag_ind])
            bag_ind = bag_ind + 1

            return bag_piece, bag_ind, next_bag

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(7,), dtype=tf.int64, name="tensor")],
    )
    def _stateless_shuffle(self, tensor):
        # Shuffle the tensor using a stateless random seed
        with tf.name_scope("StatelessShuffle"):
            size = tf.shape(tensor, out_type=tf.int32)[0]
            logits = tf.random.stateless_uniform(
                shape=(size,),
                seed=self._seed + tf.cast(tensor[:2], tf.int32),
                dtype=tf.float32,
            )
            indices = tf.argsort(logits, axis=0)
            shuffled_tensor = tf.gather(tensor, indices)
            return shuffled_tensor

    @tf.function(jit_compile=True)
    def _clear_lines(self) -> tf.Tensor:
        # Clear lines from the board
        with tf.name_scope("ClearLines"):
            board = tf.identity(self._board)
            empty_board = tf.zeros_like(board)

            rows_to_clear = tf.reduce_all(self._board != 0.0, axis=1)
            rows_to_keep = tf.logical_not(rows_to_clear)
            clears = tf.reduce_sum(tf.cast(rows_to_clear, tf.int64))

            # Clear filled rows
            rows_cleared_board = tf.where(rows_to_clear[:, None], empty_board, board)
            # Drop down the rows above using stable argsort
            row_inds = tf.argsort(
                tf.cast(rows_to_keep, tf.int64),
                axis=0,
                direction="ASCENDING",
                stable=True,
            )
            new_board = tf.gather(rows_cleared_board, row_inds)

            self._board.assign(new_board)

            return clears

    @tf.function(jit_compile=True)
    def _get_heights(self) -> tf.Tensor:
        # Get heights of each column in the board
        with tf.name_scope("GetHeights"):
            height_matrix = tf.range(self._board.shape[0], 0, -1, dtype=tf.float32)[
                ..., None
            ]
            heights = tf.reduce_max(self._board * height_matrix, axis=0)

            return heights

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(10,), dtype=tf.float32, name="heights")],
    )
    def _get_holes(self, heights) -> tf.Tensor:
        # Count holes in the board
        with tf.name_scope("GetHoles"):
            holes = heights - tf.reduce_sum(self._board, axis=0)
            return holes

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(10,), dtype=tf.float32, name="heights")],
    )
    def _get_skyline(self, heights) -> tf.Tensor:
        # Computes the difference between the hightest columns and lowest columns
        # For a board with width 10, the skyline is the difference between the
        # sum of the 5 highest columns and the sum of the 5 lowest columns
        with tf.name_scope("GetSkyline"):
            sorted_heights = tf.sort(heights)
            num_cols = tf.shape(sorted_heights)[0]
            skyline = tf.reduce_sum(sorted_heights[-num_cols // 2 :]) - tf.reduce_sum(
                sorted_heights[: num_cols // 2]
            )
            return skyline

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(10,), dtype=tf.float32, name="heights")],
    )
    def _get_bumpy(self, heights) -> tf.Tensor:
        # Get bumpiness of the board
        with tf.name_scope("GetBumpy"):
            bumpy = tf.reduce_sum(tf.abs(heights[:-1] - heights[1:]))
            return bumpy

    @tf.function(jit_compile=True)
    def _board_stats(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Get total heights of the board
        with tf.name_scope("BoardStats"):
            heights = self._get_heights()
            heights_val = tf.reduce_sum(heights)

            # Get number of holes in the board
            holes = self._get_holes(heights)
            holes_val = tf.reduce_sum(holes)

            # Get skyline of the board
            skyline_val = self._get_skyline(heights)

            # Get bumpiness of the board
            bumpy_val = self._get_bumpy(heights)

            # Return heights, holes, and bumpy
            return heights_val, holes_val, skyline_val, bumpy_val
