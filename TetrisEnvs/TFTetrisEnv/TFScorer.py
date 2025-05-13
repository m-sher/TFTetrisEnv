import tensorflow as tf
from tf_agents.utils.common import create_variable
from .TFPieces import TFPieceType

class TFSpins:
    # TODO
    # More spins
    NO_SPIN = tf.constant(0, tf.int64)
    T_SPIN_MINI = tf.constant(1, tf.int64)
    T_SPIN = tf.constant(2, tf.int64)

class TFScorer():

    def __init__(self):
        self._corner_cells = tf.constant([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=tf.int64)
        self._t_fronts = tf.constant([[0, 1], [1, 2], [2, 1], [1, 0]], tf.int64)

        self._pc_attack = tf.constant([0, 5, 6, 7, 9], tf.float32)
        self._tspin_attack = tf.constant([0, 2, 4, 6, 0], tf.float32)
        self._tspin_mini_attack = tf.constant([0, 0, 1, 0, 0], tf.float32) # no tspin mini triple
        self._clear_attack = tf.constant([0, 0, 1, 2, 4], tf.float32)

        self._b2b = create_variable(name='b2b', initial_value=0, dtype=tf.int64)
        self._combo = create_variable(name='combo', initial_value=0, dtype=tf.int64)

    @tf.function(jit_compile=True)
    def reset(self):
        with tf.name_scope('Reset'):
            self._b2b.assign(0)
            self._combo.assign(0)

    @tf.function(jit_compile=True, input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(2,), dtype=tf.int64),
        tf.TensorSpec(shape=(2,), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(4, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(24, 10), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    ])
    def judge(self, piece_type, piece_loc, piece_delta_loc, piece_delta_r, piece_cells, board, clears):

        # TODO
        # all-mini+ immobile spin detection
        with tf.name_scope('Judge'):

            def process_t_piece():

                def check_t_spin():
                    # Check out of bounds corners
                    corner_coords = piece_loc + self._corner_cells # 4, 2
                    corners = tf.reduce_any([tf.reduce_any(corner_coords >= tf.shape(board, out_type=tf.int64), axis=-1),
                                             tf.reduce_any(corner_coords < 0, axis=-1)], axis=0)

                    # Check in bounds corners
                    corner_coords = tf.maximum(tf.cast(0, tf.int64), tf.minimum(corner_coords, [tf.shape(board, out_type=tf.int64)[0] - 1,
                                                                                                tf.shape(board, out_type=tf.int64)[1] - 1]))
                    corners = tf.cast(tf.reduce_any([corners, tf.gather_nd(board, corner_coords) != 0], axis=0), tf.float32)

                    # Find back cell in same order as corners. Corners[back] is
                    # is the corner anticlockwise relative to the cell
                    filled_fronts = tf.reduce_any(tf.reduce_all(self._t_fronts[:, None, :] == piece_cells[None, :, :], axis=-1), axis=-1)
                    back = tf.where(condition=tf.logical_not(filled_fronts))[0, 0]

                    front_inds = tf.stack([(back + 2) % 4, (back + 3) % 4], axis=0)
                    front_corners = tf.reduce_sum(tf.gather(corners, front_inds))

                    back_inds = tf.stack([(back + 0) % 4, (back + 1) % 4], axis=0)
                    back_corners = tf.reduce_sum(tf.gather(corners, back_inds))

                    return tf.cond(tf.logical_and(front_corners == 2, back_corners >= 1),
                                   lambda: TFSpins.T_SPIN,
                                   lambda: tf.cond(tf.logical_and(front_corners == 1, back_corners == 2),
                                                   lambda: tf.cond(tf.reduce_sum(tf.abs(piece_delta_loc)) > 2,
                                                                   lambda: TFSpins.T_SPIN,
                                                                   lambda: TFSpins.T_SPIN_MINI),
                                                   lambda: TFSpins.NO_SPIN))

                return tf.cond(piece_delta_r != 0,
                               check_t_spin,
                               lambda: TFSpins.NO_SPIN)

            spin = tf.cond(piece_type == TFPieceType.T,
                           process_t_piece,
                           lambda: TFSpins.NO_SPIN)

            perfect_clear = tf.reduce_all(board == 0)

            b2b, combo = tf.cond(clears > 0,
                                 lambda: tf.cond((tf.logical_or(tf.logical_or(spin != TFSpins.NO_SPIN, clears == 4), perfect_clear)),
                                                 lambda: (self._b2b + 1, self._combo + 1),
                                                 lambda: (tf.zeros((), dtype=tf.int64), self._combo + 1)),
                                 lambda: (self._b2b, tf.zeros((), dtype=tf.int64)))

            attack = tf.cond(perfect_clear,
                             lambda: self._pc_attack[clears],
                             lambda: tf.cond(spin == TFSpins.T_SPIN,
                                             lambda: self._tspin_attack[clears],
                                             lambda: tf.cond(spin == TFSpins.T_SPIN_MINI,
                                                             lambda: self._tspin_mini_attack[clears],
                                                             lambda: self._clear_attack[clears])))

            self._b2b.assign(b2b)
            self._combo.assign(combo)

            return attack