import tensorflow as tf


# ruff: noqa: E741
class TFPieceType:
    N = tf.constant(0, tf.int64)  # NULL - empty and only used for initial hold
    I = tf.constant(1, tf.int64)
    J = tf.constant(2, tf.int64)
    L = tf.constant(3, tf.int64)
    O = tf.constant(4, tf.int64)
    S = tf.constant(5, tf.int64)
    T = tf.constant(6, tf.int64)
    Z = tf.constant(7, tf.int64)
