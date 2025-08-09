import tensorflow as tf


class TFKeys:
    START = tf.constant(0, tf.int64)
    HOLD = tf.constant(1, tf.int64)
    TAP_LEFT = tf.constant(2, tf.int64)
    TAP_RIGHT = tf.constant(3, tf.int64)
    DAS_LEFT = tf.constant(4, tf.int64)
    DAS_RIGHT = tf.constant(5, tf.int64)
    CLOCKWISE = tf.constant(6, tf.int64)
    ANTICLOCKWISE = tf.constant(7, tf.int64)
    ROTATE_180 = tf.constant(8, tf.int64)
    SOFT_DROP = tf.constant(9, tf.int64)
    HARD_DROP = tf.constant(10, tf.int64)
    PAD = tf.constant(11, tf.int64)

    key_vectors = tf.constant(
        [
            [+0, +0, +0],  # STARTnot used in current version
            [+0, +0, +0],  # HOLD
            [+0, -1, +0],  # TAP_LEFT
            [+0, +1, +0],  # TAP_RIGHT
            [+0, -100, +0],  # DAS_LEFT
            [+0, +100, +0],  # DAS_RIGHT
            [+0, +0, +1],  # CLOCKWISE
            [+0, +0, -1],  # ANTICLOCKWISE
            [+0, +0, +2],  # ROTATE_180
            [+100, +0, +0],  # SOFT_DROP
            [+100, +0, +0],  # HARD_DROP
            [+0, +0, +0],  # PAD
        ],
        dtype=tf.int64,
    )


class TFMoves:
    _holds = [[], [TFKeys.HOLD]]

    _standards = [
        [TFKeys.DAS_LEFT],
        [TFKeys.DAS_LEFT, TFKeys.TAP_RIGHT],
        [TFKeys.TAP_LEFT, TFKeys.TAP_LEFT],
        [TFKeys.TAP_LEFT],
        [],
        [TFKeys.TAP_RIGHT],
        [TFKeys.TAP_RIGHT, TFKeys.TAP_RIGHT],
        [TFKeys.DAS_RIGHT, TFKeys.TAP_LEFT],
        [TFKeys.DAS_RIGHT],  # No rotations
        [TFKeys.DAS_LEFT, TFKeys.CLOCKWISE],
        [TFKeys.TAP_LEFT, TFKeys.TAP_LEFT, TFKeys.CLOCKWISE],
        [TFKeys.TAP_LEFT, TFKeys.CLOCKWISE],
        [TFKeys.CLOCKWISE],
        [TFKeys.TAP_RIGHT, TFKeys.CLOCKWISE],
        [TFKeys.TAP_RIGHT, TFKeys.TAP_RIGHT, TFKeys.CLOCKWISE],
        [TFKeys.DAS_RIGHT, TFKeys.TAP_LEFT, TFKeys.CLOCKWISE],
        [TFKeys.DAS_RIGHT, TFKeys.CLOCKWISE],  # CLOCKWISE rotations
        [TFKeys.DAS_LEFT, TFKeys.ANTICLOCKWISE],
        [TFKeys.TAP_LEFT, TFKeys.TAP_LEFT, TFKeys.ANTICLOCKWISE],
        [TFKeys.TAP_LEFT, TFKeys.ANTICLOCKWISE],
        [TFKeys.ANTICLOCKWISE],
        [TFKeys.TAP_RIGHT, TFKeys.ANTICLOCKWISE],
        [TFKeys.TAP_RIGHT, TFKeys.TAP_RIGHT, TFKeys.ANTICLOCKWISE],
        [TFKeys.DAS_RIGHT, TFKeys.TAP_LEFT, TFKeys.ANTICLOCKWISE],
        [TFKeys.DAS_RIGHT, TFKeys.ANTICLOCKWISE],  # ANTICLOCKWISE rotations
        [TFKeys.DAS_LEFT, TFKeys.ROTATE_180],
        [TFKeys.TAP_LEFT, TFKeys.TAP_LEFT, TFKeys.ROTATE_180],
        [TFKeys.TAP_LEFT, TFKeys.ROTATE_180],
        [TFKeys.ROTATE_180],
        [TFKeys.TAP_RIGHT, TFKeys.ROTATE_180],
        [TFKeys.TAP_RIGHT, TFKeys.TAP_RIGHT, TFKeys.ROTATE_180],
        [TFKeys.DAS_RIGHT, TFKeys.TAP_LEFT, TFKeys.ROTATE_180],
        [TFKeys.DAS_RIGHT, TFKeys.ROTATE_180],  # 180 rotations
        [TFKeys.CLOCKWISE, TFKeys.DAS_LEFT],
        [TFKeys.ANTICLOCKWISE, TFKeys.DAS_RIGHT],  # Spin first DAS
    ]

    _spins = [
        [],
        [TFKeys.SOFT_DROP, TFKeys.CLOCKWISE],
        [TFKeys.SOFT_DROP, TFKeys.ANTICLOCKWISE],
        [
            TFKeys.SOFT_DROP,
            TFKeys.ROTATE_180,
        ],
        [TFKeys.SOFT_DROP, TFKeys.CLOCKWISE, TFKeys.CLOCKWISE],
        [TFKeys.SOFT_DROP, TFKeys.ANTICLOCKWISE, TFKeys.ANTICLOCKWISE],
        [TFKeys.SOFT_DROP, TFKeys.CLOCKWISE, TFKeys.ROTATE_180],
        [TFKeys.SOFT_DROP, TFKeys.ANTICLOCKWISE, TFKeys.ROTATE_180],
    ]


def pad_sequence(sequence):
    padded_sequence = tf.concat([sequence, [TFKeys.PAD] * (9 - len(sequence))], axis=0)
    return padded_sequence


class TFConvert:
    to_sequence = tf.cast(
        tf.stack(
            [
                pad_sequence(
                    [TFKeys.START] + hold + standard + spin + [TFKeys.HARD_DROP]
                )
                for hold in TFMoves._holds
                for standard in TFMoves._standards
                for spin in TFMoves._spins
            ],
            axis=0,
        ),
        tf.int64,
    )
