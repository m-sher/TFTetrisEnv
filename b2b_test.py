#!/usr/bin/env python3
"""
Headless benchmark for b2b_search weight tuning.

Runs N games per weight configuration and reports:
  - Avg/max height over the game
  - Total attack, total clears, lines/piece
  - Survival (steps before death)
"""

import ctypes
import numpy as np
import os
import glob
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch


# ── Load the setter function ────────────────────────────────
def _find_lib():
    d = os.path.dirname(os.path.abspath(__file__))
    tetris_dir = os.path.join(d, "TetrisEnv")
    candidates = (
        glob.glob(os.path.join(tetris_dir, "b2b_search*.so"))
        + glob.glob(os.path.join(d, "b2b_search*.so"))
    )
    return candidates[0] if candidates else None


_lib = ctypes.CDLL(_find_lib())
_lib.b2b_set_weights.argtypes = [ctypes.c_float] * 20
_lib.b2b_set_weights.restype = None

# Default weights (baseline)
DEFAULTS = dict(
    height=10.0, avg_height=5.0, bumpiness=0,
    holes=0, hole_col=0, deep_holes=0.0,
    clearable=0, b2b=10.0, combo=0,
    b2b_break=1.0, spike=0.0, spin_setup=1.0,
    tslot=2.0, immobile_clear=2.0, hole_ceiling=0.0,
    wasted_hole=0.0, attack=10.0, app_bonus=0.0,
    garb_cancel=0.0,
    streak=0.0,
)


def set_weights(**overrides):
    w = {**DEFAULTS, **overrides}
    _lib.b2b_set_weights(
        w["height"], w["avg_height"], w["bumpiness"],
        w["holes"], w["hole_col"], w["deep_holes"],
        w["clearable"], w["b2b"], w["combo"],
        w["b2b_break"], w["spike"], w["spin_setup"],
        w["tslot"], w["immobile_clear"], w["hole_ceiling"],
        w["wasted_hole"], w["attack"], w["app_bonus"],
        w["garb_cancel"], w["streak"],
    )


def run_game(seed, num_steps=200, search_depth=4, beam_width=64,
             garbage_chance=0.15, garbage_min=1, garbage_max=4, queue_size=5):
    """Run a single headless game and return stats dict."""
    env = PyTetrisEnv(
        queue_size=queue_size, max_holes=None, max_height=20,
        max_steps=None, max_len=15, pathfinding=False,
        seed=seed, idx=0,
        garbage_chance=garbage_chance, garbage_min=garbage_min,
        garbage_max=garbage_max, auto_push_garbage=True, auto_fill_queue=True,
    )
    env.reset()
    searcher = CB2BSearch()

    total_attack = 0.0
    total_clears = 0
    heights = []
    step = 0
    game_over = False

    for step in range(1, num_steps + 1):
        if game_over:
            break

        board = env._board
        active = env._active_piece.piece_type.value
        hold = env._hold_piece.value
        queue_types = np.array([p.value for p in env._queue], dtype=np.int32)
        b2b = env._scorer._b2b
        combo = env._scorer._combo
        total_garb = env._get_total_garbage()

        action_idx, sequence = searcher.search(
            board=board, active_piece=active, hold_piece=hold,
            queue=queue_types, b2b=b2b, combo=combo,
            total_garbage=total_garb,
            garbage_push_delay=env._garbage_push_delay,
            search_depth=search_depth,
            beam_width=beam_width, max_len=15,
        )

        if action_idx < 0:
            game_over = True
            break

        ts = env._step(sequence)

        atk = float(ts.reward["attack"])
        clr = float(ts.reward["clear"])
        total_attack += atk
        total_clears += int(clr)

        h = 0
        for r in range(24):
            if np.any(env._board[r] != 0):
                h = 24 - r
                break
        heights.append(h)

        if ts.is_last():
            game_over = True

    return {
        "steps": step,
        "total_attack": total_attack,
        "total_clears": total_clears,
        "lines_per_piece": total_clears / max(step, 1),
        "avg_height": np.mean(heights) if heights else 0,
        "max_height": max(heights) if heights else 0,
        "survived": not game_over,
        "attack_per_piece": total_attack / max(step, 1),
    }


def benchmark(label, num_games=5, num_steps=100, **weight_overrides):
    """Run num_games with given weights, print summary."""
    set_weights(**weight_overrides)

    results = []
    for seed in range(num_games):
        r = run_game(seed=seed + 100, num_steps=num_steps)
        results.append(r)

    avg = lambda key: np.mean([r[key] for r in results])
    survived = sum(1 for r in results if r["survived"])

    print(f"\n{'='*60}")
    print(f"  {label}")
    changes = {k: v for k, v in weight_overrides.items() if v != DEFAULTS.get(k)}
    if changes:
        print(f"  Changes: {changes}")
    print(f"{'='*60}")
    print(f"  Games: {num_games} | Steps/game: {num_steps}")
    print(f"  Survived:       {survived}/{num_games}")
    print(f"  Avg steps:      {avg('steps'):.1f}")
    print(f"  Avg height:     {avg('avg_height'):.1f}")
    print(f"  Max height:     {avg('max_height'):.1f}")
    print(f"  Total attack:   {avg('total_attack'):.1f}")
    print(f"  Total clears:   {avg('total_clears'):.1f}")
    print(f"  Lines/piece:    {avg('lines_per_piece'):.3f}")
    print(f"  Attack/piece:   {avg('attack_per_piece'):.3f}")
    return results


CONFIGS = {
    "A": ("A: BASELINE (current defaults)", {}),
}


def run_no_garbage_game(seed, num_steps=200, search_depth=4, beam_width=64,
                         queue_size=5):
    """Run a game with NO garbage — tests pure B2B maintenance."""
    return run_game(seed=seed, num_steps=num_steps, search_depth=search_depth,
                    beam_width=beam_width, garbage_chance=0.0,
                    garbage_min=0, garbage_max=0, queue_size=queue_size)


def no_garbage_benchmark(label, num_games=3, num_steps=200):
    """Run games with no garbage, report B2B maintenance."""
    set_weights()  # Use defaults

    results = []
    for seed in range(num_games):
        r = run_no_garbage_game(seed=seed + 500, num_steps=num_steps)
        results.append(r)

    avg = lambda key: np.mean([r[key] for r in results])
    survived = sum(1 for r in results if r["survived"])

    print(f"\n{'='*60}")
    print(f"  NO GARBAGE: {label}")
    print(f"{'='*60}")
    print(f"  Games: {num_games} | Steps/game: {num_steps}")
    print(f"  Survived:       {survived}/{num_games}")
    print(f"  Avg steps:      {avg('steps'):.1f}")
    print(f"  Avg height:     {avg('avg_height'):.1f}")
    print(f"  Max height:     {avg('max_height'):.1f}")
    print(f"  Total attack:   {avg('total_attack'):.1f}")
    print(f"  Attack/piece:   {avg('attack_per_piece'):.3f}")
    return results


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]

    if args and args[0].lower() == "nogarb":
        no_garbage_benchmark("Current defaults", num_games=3, num_steps=200)
        sys.exit(0)

    keys = args if args else list(CONFIGS.keys())
    for key in keys:
        key = key.upper()
        if key not in CONFIGS:
            print(f"Unknown config: {key}. Available: {', '.join(CONFIGS.keys())}")
            continue
        label, overrides = CONFIGS[key]
        benchmark(label, **overrides)
