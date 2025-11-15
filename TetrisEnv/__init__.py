"""
TFTetrisEnv

TensorFlow environment for RL with Tetris SRS+.
"""

from .Pieces import PieceType, Piece
from .Moves import Keys, Moves
from .RotationSystem import RotationSystem
from .Scorer import Scorer
from .PyTetrisEnv import PyTetrisEnv
from .PyTetrisRunner import PyTetrisRunner
from .TetrioRandom import TetrioRNG
from .KeySequencesBitboard import BitboardKeySequenceFinder

__version__ = "0.7.0"

__all__ = [
    "PieceType",
    "Piece",
    "Keys",
    "Moves",
    "RotationSystem",
    "Scorer",
    "PyTetrisEnv",
    "PyTetrisRunner",
    "TetrioRNG",
    "BitboardKeySequenceFinder",
]


def info():
    """Return basic package information."""
    return f"TFTetrisEnv version {__version__}: TensorFlow environment for RL with Tetris SRS+."
