import numpy as np

class Keys:
    START = 0
    hold = 1
    tap_left = 2
    tap_right = 3
    DAS_left = 4
    DAS_right = 5
    clockwise = 6
    anticlockwise = 7
    rotate_180 = 8
    soft_drop = 9
    hard_drop = 10
    PAD = 11
    
    key_vectors = {
        # vertical, horizontal, rotation
        tap_left: np.array([+0, -1, +0], dtype=np.int32),
        tap_right: np.array([+0, +1, +0], dtype=np.int32),
        DAS_left: np.array([+0, -100, +0], dtype=np.int32),
        DAS_right: np.array([+0, +100, +0], dtype=np.int32),
        clockwise: np.array([+0, +0, +1], dtype=np.int32),
        anticlockwise: np.array([+0, +0, -1], dtype=np.int32),
        rotate_180: np.array([+0, +0, +2], dtype=np.int32),
        soft_drop: np.array([+100, +0, +0], dtype=np.int32), # soft and hard drop
        hard_drop: np.array([+100, +0, +0], dtype=np.int32), # are the same deltas
    }
    
class Moves:
    _holds = [
        [],
        [Keys.hold]
    ]
    _standards = [
        [Keys.DAS_left],
        [Keys.DAS_left, Keys.tap_right],
        [Keys.tap_left, Keys.tap_left],
        [Keys.tap_left],
        [],
        [Keys.tap_right],
        [Keys.tap_right, Keys.tap_right],
        [Keys.DAS_right, Keys.tap_left],
        [Keys.DAS_right],                                       # No rotations
        
        [Keys.DAS_left, Keys.clockwise],
        [Keys.tap_left, Keys.tap_left, Keys.clockwise],
        [Keys.tap_left, Keys.clockwise],
        [Keys.clockwise],
        [Keys.tap_right, Keys.clockwise],
        [Keys.tap_right, Keys.tap_right, Keys.clockwise],
        [Keys.DAS_right, Keys.tap_left, Keys.clockwise],
        [Keys.DAS_right, Keys.clockwise],                       # clockwise rotations
        
        [Keys.DAS_left, Keys.anticlockwise],
        [Keys.tap_left, Keys.tap_left, Keys.anticlockwise],
        [Keys.tap_left, Keys.anticlockwise],
        [Keys.anticlockwise],
        [Keys.tap_right, Keys.anticlockwise],
        [Keys.tap_right, Keys.tap_right, Keys.anticlockwise],
        [Keys.DAS_right, Keys.tap_left, Keys.anticlockwise],
        [Keys.DAS_right, Keys.anticlockwise],                   # anticlockwise rotations
        
        [Keys.DAS_left, Keys.rotate_180],
        [Keys.tap_left, Keys.tap_left, Keys.rotate_180],
        [Keys.tap_left, Keys.rotate_180],
        [Keys.rotate_180],
        [Keys.tap_right, Keys.rotate_180],
        [Keys.tap_right, Keys.tap_right, Keys.rotate_180],
        [Keys.DAS_right, Keys.tap_left, Keys.rotate_180],
        [Keys.DAS_right, Keys.rotate_180],                      # 180 rotations
        
        [Keys.clockwise, Keys.DAS_left],
        [Keys.anticlockwise, Keys.DAS_right],                   # Spin first DAS
    ]
    _spins = [
        [],
        [Keys.soft_drop, Keys.clockwise],
        [Keys.soft_drop, Keys.anticlockwise],
        [Keys.soft_drop, Keys.rotate_180],
        [Keys.soft_drop, Keys.clockwise, Keys.clockwise],
        [Keys.soft_drop, Keys.anticlockwise, Keys.anticlockwise],
        [Keys.soft_drop, Keys.clockwise, Keys.rotate_180],
        [Keys.soft_drop, Keys.anticlockwise, Keys.rotate_180]
    ]