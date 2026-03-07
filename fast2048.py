from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    import fast2048_sim as _sim
except ImportError:  # pragma: no cover
    _sim = None


def is_available() -> bool:
    return _sim is not None


def require_sim() -> None:
    if _sim is None:
        raise RuntimeError(
            "fast2048_sim is not installed. Build with: cd rust_sim && maturin develop --release"
        )


def onehot_to_exponents(obs) -> List[int]:
    board = obs.argmax(axis=-1)
    return [int(board[r][c]) for r in range(4) for c in range(4)]


def exponents_to_onehot(board: Sequence[int]):
    import numpy as np

    if len(board) != 16:
        raise ValueError("board must have 16 elements")
    out = np.zeros((4, 4, 16), dtype=np.uint8)
    for i, value in enumerate(board):
        r, c = divmod(i, 4)
        out[r, c, int(value)] = 1
    return out


def apply_action(board: Sequence[int], action: int) -> Tuple[List[int], int, bool]:
    require_sim()
    next_board, reward, moved = _sim.apply_action(list(board), int(action))
    return list(next_board), int(reward), bool(moved)


def legal_actions(board: Sequence[int]) -> List[int]:
    require_sim()
    return [int(a) for a in _sim.legal_actions(list(board))]


def empty_positions(board: Sequence[int]) -> List[int]:
    require_sim()
    return [int(p) for p in _sim.empty_positions(list(board))]


def spawn_tile(board: Sequence[int], position: int, value_exp: int) -> List[int]:
    require_sim()
    return [int(v) for v in _sim.spawn_tile(list(board), int(position), int(value_exp))]