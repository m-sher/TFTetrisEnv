"""
TFTetrisEnv

TensorFlow environment for RL with Tetris SRS+.
"""

from .TFPieces import TFPieceType
from .TFMoves import TFKeys, TFMoves
from .TFRotationSystem import TFRotationSystem
from .TFScorer import TFScorer
from .TFTetrisEnv import TFTetrisEnv
from .TFTetrisRunner import TFTetrisRunner

__version__ = "0.4.0"

__all__ = [
    "TFTetrisEnv",
    "TFPieceType",
    "TFMoves",
    "TFKeys",
    "TFConvert",
    "TFRotationSystem",
    "TFScorer",
    "TFTetrisRunner",
]


def info():
    """Return basic package information."""
    return f"TFTetrisEnv version {__version__}: TensorFlow environment for RL with Tetris SRS+."
