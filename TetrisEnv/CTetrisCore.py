import ctypes
import glob
import os
from typing import List, Optional

import numpy as np


BOARD_ROWS = 40
MAX_PLACEMENTS = 512


class TetEvent(ctypes.Structure):
    _fields_ = [
        ("clears", ctypes.c_int),
        ("attack", ctypes.c_float),
        ("new_b2b", ctypes.c_int),
        ("new_combo", ctypes.c_int),
        ("spin_type", ctypes.c_int),
        ("perfect_clear", ctypes.c_int),
        ("terminal", ctypes.c_int),
        ("garbage_pushed", ctypes.c_int),
    ]


def _load_lib() -> Optional[ctypes.CDLL]:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = glob.glob(os.path.join(curr_dir, "b2b_search*.so"))
    lib_path = candidates[0] if candidates else os.path.join(curr_dir, "b2b_search.so")
    try:
        return ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"Warning: Could not load b2b_search library at {lib_path}: {e}")
        return None


_LIB: Optional[ctypes.CDLL] = None


def _bind() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Could not load b2b_search shared library")

    lib.tet_state_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.tet_state_new.restype = ctypes.c_void_p

    lib.tet_state_free.argtypes = [ctypes.c_void_p]
    lib.tet_state_free.restype = None

    lib.tet_state_clone.argtypes = [ctypes.c_void_p]
    lib.tet_state_clone.restype = ctypes.c_void_p

    lib.tet_state_serialized_size.argtypes = []
    lib.tet_state_serialized_size.restype = ctypes.c_int

    lib.tet_state_serialize.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.tet_state_serialize.restype = ctypes.c_int

    lib.tet_state_deserialize.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    lib.tet_state_deserialize.restype = ctypes.c_void_p

    lib.tet_state_get_board.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.uint16, ndim=1, flags="C_CONTIGUOUS"),
    ]
    lib.tet_state_get_board.restype = None

    lib.tet_state_get_board_height.argtypes = [ctypes.c_void_p]
    lib.tet_state_get_board_height.restype = ctypes.c_int

    lib.tet_state_get_active_piece.argtypes = [ctypes.c_void_p]
    lib.tet_state_get_active_piece.restype = ctypes.c_int

    lib.tet_state_get_hold_piece.argtypes = [ctypes.c_void_p]
    lib.tet_state_get_hold_piece.restype = ctypes.c_int

    lib.tet_state_get_queue.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    lib.tet_state_get_queue.restype = ctypes.c_int

    lib.tet_state_get_b2b.argtypes = [ctypes.c_void_p]
    lib.tet_state_get_b2b.restype = ctypes.c_int

    lib.tet_state_get_combo.argtypes = [ctypes.c_void_p]
    lib.tet_state_get_combo.restype = ctypes.c_int

    lib.tet_state_get_total_garbage.argtypes = [ctypes.c_void_p]
    lib.tet_state_get_total_garbage.restype = ctypes.c_int

    lib.tet_enumerate_placements.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.tet_enumerate_placements.restype = ctypes.c_int

    lib.tet_apply_placement.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(TetEvent),
    ]
    lib.tet_apply_placement.restype = ctypes.c_int

    lib.tet_get_last_search_placement.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.tet_get_last_search_placement.restype = None

    lib.tet_decompose.argtypes = [
        ctypes.c_void_p,
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
        ctypes.c_int,
    ]
    lib.tet_decompose.restype = ctypes.c_int

    lib.tet_get_piece_min_col.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.tet_get_piece_min_col.restype = ctypes.c_int

    lib.tet_inject_random_garbage.argtypes = [
        ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int,
    ]
    lib.tet_inject_random_garbage.restype = ctypes.c_int

    _LIB = lib
    return lib


def get_piece_min_col(piece_type: int, rotation: int) -> int:
    """Bounding-box col + this offset = 0..9 board col for (piece_type, rotation)."""
    return int(_bind().tet_get_piece_min_col(int(piece_type), int(rotation)))


def get_last_search_placement():
    """Return (rot, col, landing_row, spin_type) of the most recent b2b_search_c call."""
    lib = _bind()
    rot = ctypes.c_int(0)
    col = ctypes.c_int(0)
    lr = ctypes.c_int(0)
    sp = ctypes.c_int(0)
    lib.tet_get_last_search_placement(
        ctypes.byref(rot), ctypes.byref(col), ctypes.byref(lr), ctypes.byref(sp)
    )
    return rot.value, col.value, lr.value, sp.value


class TetrisCore:
    """Opaque handle around the C-side TetState. Owns its own memory; frees on __del__."""

    __slots__ = ("_lib", "_handle")

    def __init__(
        self,
        seed: int = 0,
        board_height: int = 24,
        queue_size: int = 5,
        garbage_push_delay: int = 1,
        _adopt_handle: Optional[int] = None,
    ):
        self._lib = _bind()
        if _adopt_handle is not None:
            self._handle = _adopt_handle
        else:
            self._handle = self._lib.tet_state_new(
                int(seed), int(board_height), int(queue_size), int(garbage_push_delay)
            )
            if not self._handle:
                raise MemoryError("tet_state_new returned NULL")

    def clone(self) -> "TetrisCore":
        h = self._lib.tet_state_clone(self._handle)
        if not h:
            raise MemoryError("tet_state_clone returned NULL")
        return TetrisCore(_adopt_handle=h)

    def serialize(self) -> np.ndarray:
        n = self._lib.tet_state_serialized_size()
        buf = np.empty(n, dtype=np.uint8)
        self._lib.tet_state_serialize(self._handle, buf)
        return buf

    @classmethod
    def deserialize(cls, buf: np.ndarray) -> "TetrisCore":
        b = np.ascontiguousarray(buf, dtype=np.uint8)
        lib = _bind()
        h = lib.tet_state_deserialize(b, b.size)
        if not h:
            raise ValueError("tet_state_deserialize returned NULL")
        return cls(_adopt_handle=h)

    @property
    def board(self) -> np.ndarray:
        out = np.empty(BOARD_ROWS, dtype=np.uint16)
        self._lib.tet_state_get_board(self._handle, out)
        return out

    @property
    def board_height(self) -> int:
        return int(self._lib.tet_state_get_board_height(self._handle))

    @property
    def active_piece(self) -> int:
        return int(self._lib.tet_state_get_active_piece(self._handle))

    @property
    def hold_piece(self) -> int:
        return int(self._lib.tet_state_get_hold_piece(self._handle))

    @property
    def queue(self) -> List[int]:
        buf = np.empty(16, dtype=np.int32)
        n = self._lib.tet_state_get_queue(self._handle, buf, buf.size)
        return buf[:n].tolist()

    @property
    def b2b(self) -> int:
        return int(self._lib.tet_state_get_b2b(self._handle))

    @property
    def combo(self) -> int:
        return int(self._lib.tet_state_get_combo(self._handle))

    @property
    def total_garbage(self) -> int:
        return int(self._lib.tet_state_get_total_garbage(self._handle))

    def enumerate_placements(self, include_hold: bool = True,
                             max_placements: int = MAX_PLACEMENTS * 2) -> np.ndarray:
        """Returns an (N, 5) int32 array. Columns: [is_hold, rot, col, landing_row, spin_type]."""
        out = np.empty(max_placements * 5, dtype=np.int32)
        n = self._lib.tet_enumerate_placements(
            self._handle, out, max_placements, 1 if include_hold else 0
        )
        return out[: n * 5].reshape(n, 5)

    def apply_placement(self, is_hold: int, rot: int, col: int,
                        landing_row: int, spin_type: int) -> TetEvent:
        ev = TetEvent()
        rc = self._lib.tet_apply_placement(
            self._handle,
            int(is_hold), int(rot), int(col),
            int(landing_row), int(spin_type),
            ctypes.byref(ev),
        )
        if rc != 0:
            raise RuntimeError(f"tet_apply_placement failed (rc={rc})")
        return ev

    def decompose(self, max_placements: int = MAX_PLACEMENTS) -> np.ndarray:
        """Returns an (N, 21) float32 array — one row per active-piece placement,
        each column a hand-tuned heuristic component (matches b2b_decompose_c)."""
        buf = np.zeros(max_placements * 21, dtype=np.float32)
        n = self._lib.tet_decompose(self._handle, buf, max_placements)
        return buf[: n * 21].reshape(n, 21)

    def inject_random_garbage(self, chance: float, min_rows: int, max_rows: int) -> bool:
        """Stochastically queue ambient garbage. Uses the state's deterministic RNG."""
        return bool(self._lib.tet_inject_random_garbage(
            self._handle, float(chance), int(min_rows), int(max_rows)
        ))

    def __del__(self):
        h = getattr(self, "_handle", None)
        if h:
            try:
                self._lib.tet_state_free(h)
            except Exception:
                pass
            self._handle = None
