"""
TetrisEnvs

Tetris environments for RL with Tetris SRS+.
"""

__version__ = "0.4.0"

# Import and expose PyTetrisEnv and related components
from .PyTetrisEnv import PyTetrisEnv
from .PyTetrisEnv.Pieces import PieceType, Piece
from .PyTetrisEnv.Moves import Moves, Keys
from .PyTetrisEnv.RotationSystem import RotationSystem
from .PyTetrisEnv.Scorer import Scorer
from .PyTetrisEnv.PyTetrisRunner import PyTetrisRunner

from .TFTetrisEnv import TFTetrisEnv
from .TFTetrisEnv.TFPieces import TFPieceType
from .TFTetrisEnv.TFMoves import TFMoves, TFKeys
from .TFTetrisEnv.TFRotationSystem import TFRotationSystem
from .TFTetrisEnv.TFScorer import TFScorer
from .TFTetrisEnv.TFTetrisRunner import TFTetrisRunner

__all__ = [
    'PyTetrisEnv',
    'PieceType', 
    'Piece',
    'Moves', 
    'Keys',
    'RotationSystem',
    'Scorer',
    'PyTetrisRunner',
    'TFTetrisEnv',
    'TFPieceType',
    'TFMoves',
    'TFKeys',
    'TFConvert',
    'TFRotationSystem',
    'TFScorer',
    'TFTetrisRunner'
]

def info():
    """Return basic package information."""
    return f"TetrisEnvs version {__version__}: Tetris environments for RL with Tetris SRS+."