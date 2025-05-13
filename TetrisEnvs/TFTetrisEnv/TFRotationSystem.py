import tensorflow as tf
from .TFPieces import TFPieceType

class TFRotationSystem():
    
    def __init__(self):
        
        self.orientations = tf.constant([
            [ # N - Never used
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0]]
            ],
            [ # I
                [[1, 0], [1, 1], [1, 2], [1, 3]],
                [[0, 2], [1, 2], [2, 2], [3, 2]],
                [[2, 0], [2, 1], [2, 2], [2, 3]],
                [[0, 1], [1, 1], [2, 1], [3, 1]]
            ],
            [ # J
                [[0, 0], [1, 0], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [2, 1]],
                [[1, 0], [1, 1], [1, 2], [2, 2]],
                [[0, 1], [1, 1], [2, 0], [2, 1]]
            ],
            [ # L
                [[0, 2], [1, 0], [1, 1], [1, 2]],
                [[0, 1], [1, 1], [2, 1], [2, 2]],
                [[1, 0], [1, 1], [1, 2], [2, 0]],
                [[0, 0], [0, 1], [1, 1], [2, 1]]
            ],
            [ # O
                [[0, 1], [0, 2], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [1, 2]],
                [[0, 1], [0, 2], [1, 1], [1, 2]]
            ],
            [ # S
                [[0, 1], [0, 2], [1, 0], [1, 1]],
                [[0, 1], [1, 1], [1, 2], [2, 2]],
                [[1, 1], [1, 2], [2, 0], [2, 1]],
                [[0, 0], [1, 0], [1, 1], [2, 1]]
            ],
            [ # T
                [[0, 1], [1, 0], [1, 1], [1, 2]],
                [[0, 1], [1, 1], [1, 2], [2, 1]],
                [[1, 0], [1, 1], [1, 2], [2, 1]],
                [[0, 1], [1, 0], [1, 1], [2, 1]]
            ],
            [ # Z
                [[0, 0], [0, 1], [1, 1], [1, 2]],
                [[0, 2], [1, 1], [1, 2], [2, 1]],
                [[1, 0], [1, 1], [2, 1], [2, 2]],
                [[0, 1], [1, 0], [1, 1], [2, 0]]
            ],
        ], dtype=tf.int64)
        
        self.kicks = tf.constant([
            [
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]], # Never used
                [[+0, -1], [-1, -1], [+2, +0], [+2, -1], [+0, +0]], # 0 -> 1
                [[-1, +0], [-1, +1], [-1, -1], [+0, +1], [+0, -1]], # 0 -> 2
                [[+0, +1], [-1, +1], [+2, +0], [+2, +1], [+0, +0]]  # 0 -> 3
            ],
            [
                [[+0, +1], [+1, +1], [-2, +0], [-2, +1], [+0, +0]], # 1 -> 0
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]], # Never used
                [[+0, +1], [+1, +1], [-2, +0], [-2, +1], [+0, +0]], # 1 -> 2
                [[+0, +1], [-2, +1], [-1, +1], [-2, +0], [-1, +0]]  # 1 -> 3
            ],
            [
                [[+1, +0], [+1, -1], [+1, +1], [+0, -1], [+0, +1]], # 2 -> 0
                [[+0, -1], [-1, -1], [+2, +0], [+2, -1], [+0, +0]], # 2 -> 1
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]], # Never used
                [[+0, +1], [-1, +1], [+2, +0], [+2, +1], [+0, +0]]  # 2 -> 3
            ],
            [
                [[+0, -1], [+1, -1], [-2, +0], [-2, -1], [+0, +0]], # 3 -> 0
                [[+0, -1], [-2, -1], [-1, -1], [-2, +0], [-1, +0]], # 3 -> 1
                [[+0, -1], [+1, -1], [-2, +0], [-2, -1], [+0, +0]], # 3 -> 2
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]]  # Never used
            ]
        ], dtype=tf.int64)

        self.i_kicks = tf.constant([
            [
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]], # Never used
                [[+0, +1], [+0, -2], [+1, -2], [-2, +1], [+0, +0]], # 0 -> 1
                [[-1, +0], [-1, +1], [-1, -1], [+0, +1], [+0, -1]], # 0 -> 2
                [[+0, -1], [+0, +2], [+1, +2], [-2, -1], [+0, +0]]  # 0 -> 3
            ],
            [
                [[+0, -1], [+0, +2], [+2, -1], [-1, +2], [+0, +0]], # 1 -> 0
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]], # Never used
                [[+0, -1], [+0, +2], [-2, -1], [+1, +2], [+0, +0]], # 1 -> 2
                [[+0, +1], [-2, +1], [-1, +1], [-2, +0], [-1, +0]]  # 1 -> 3
            ],
            [
                [[+1, +0], [+1, -1], [+1, +1], [+0, -1], [+0, +1]], # 2 -> 0
                [[+0, -2], [+0, +1], [-1, -2], [+2, +1], [+0, +0]], # 2 -> 1
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]], # Never used
                [[+0, +2], [+0, -1], [-1, +2], [+2, -1], [+0, +0]]  # 2 -> 3
            ],
            [
                [[+0, +1], [+0, -2], [+2, +1], [-1, -2], [+0, +0]], # 3 -> 0
                [[+0, -1], [-2, -1], [-1, -1], [-2, +0], [-1, +0]], # 3 -> 1
                [[+0, +1], [+0, -2], [-2, +1], [+1, -2], [+0, +0]], # 3 -> 2
                [[+0, +0], [+0, +0], [+0, +0], [+0, +0], [+0, +0]]  # Never used
            ]
        ], dtype=tf.int64)

    @tf.function(jit_compile=True, input_signature=[
        tf.TensorSpec(shape=(4, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(2,), dtype=tf.int64),
        tf.TensorSpec(shape=(24, 10), dtype=tf.float32)
    ])
    def overlaps(self, cells, loc, board) -> bool:
        with tf.name_scope('Overlaps'):          
            cell_coords = cells + loc
    
            # Outside board vertically
            outside_v = tf.logical_or(tf.reduce_any(cell_coords[:, 0] < 0),  tf.reduce_any(cell_coords[:, 0] > board.shape[0] - 1))

            # Outside board horizontally
            outside_h = tf.logical_or(tf.reduce_any(cell_coords[:, 1] < 0), tf.reduce_any(cell_coords[:, 1] > board.shape[1] - 1))
        
            # Overlaps occupied cell
            occupied = tf.reduce_any(tf.gather_nd(board, cell_coords) != 0)
            
            return tf.logical_or(tf.logical_or(outside_v, outside_h), occupied)
    
    @tf.function(jit_compile=True, input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(2,), dtype=tf.int64),
        tf.TensorSpec(shape=(2,), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(4, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(4, 2), dtype=tf.int64),
        tf.TensorSpec(shape=(24, 10), dtype=tf.float32)
    ])
    def kick_piece(self, piece_type, piece_loc, piece_delta_loc, piece_r, piece_cells, new_r, new_delta_r, new_cells, board):
        with tf.name_scope('Kick'):

            kick_table = tf.cond(piece_type == TFPieceType.I,
                                 lambda: self.i_kicks,
                                 lambda: self.kicks)

            rot_inds = tf.stack([piece_r, new_r], axis=0)
            delta_loc_candidates = tf.gather_nd(kick_table, rot_inds)

            def check_kick_body(kicked, delta_loc, i):
                kicked, delta_loc, i = tf.cond(tf.reduce_all(delta_loc_candidates[i] == tf.zeros((2,), dtype=tf.int64)),
                                               lambda: (tf.constant(False), delta_loc, tf.shape(delta_loc_candidates, out_type=tf.int64)[0]),
                                               lambda: tf.cond(self.overlaps(cells=new_cells, loc=piece_loc + delta_loc_candidates[i], board=board),
                                                               lambda: (tf.constant(False), delta_loc, i + 1),
                                                               lambda: (tf.constant(True), delta_loc_candidates[i], tf.shape(delta_loc_candidates, out_type=tf.int64)[0])))
                return kicked, delta_loc, i
            
            delta_loc = tf.identity(piece_delta_loc)
            delta_loc_ind = tf.constant(0, dtype=tf.int64)
            kicked = tf.constant(False)

            kicked, delta_loc, delta_loc_ind = tf.while_loop(lambda k, d, i: tf.logical_and(tf.logical_not(k), i < tf.shape(delta_loc_candidates, out_type=tf.int64)[0]),
                                                             check_kick_body,
                                                             [kicked, delta_loc, delta_loc_ind])

            return tf.cond(kicked,
                           lambda: (tf.identity(piece_loc + delta_loc),
                                    tf.identity(delta_loc),
                                    tf.identity(new_r),
                                    tf.identity(new_delta_r),
                                    tf.identity(new_cells)),
                           lambda: (tf.identity(piece_loc),
                                    tf.identity(delta_loc),
                                    tf.identity(piece_r),
                                    tf.zeros((), dtype=tf.int64),
                                    tf.identity(piece_cells)))