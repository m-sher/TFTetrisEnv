#!/usr/bin/env python3
"""
CMA-ES weight optimizer for the b2b_search Tetris heuristic.

Algorithm: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
  - Maintains a multivariate Gaussian over log-space weights
  - Ranks candidates by fitness, updates mean toward the best
  - Learns weight correlations via covariance matrix adaptation
  - Adapts step-size (sigma) automatically

Fitness priorities (lexicographic):
  1. Survival rate  (×10 000)
  2. Max B2B streak (×   100)
  3. Attack/piece   (×    10)
  Minus L2 regularisation on log-deviation from baseline.

Features:
  - Log-space optimisation (weights stay positive automatically)
  - L2 regularisation against baseline prevents explosion
  - Shared seeds per generation for fair candidate comparison
  - Baseline injected as candidate on generation 0
  - W&B or TensorBoard telemetry + JSON checkpointing

Usage:
  uv run python b2b_optimize.py
  uv run python b2b_optimize.py --generations 50 --pop-size 20
  uv run python b2b_optimize.py --resume runs/b2b_opt_20240101_120000
  uv run python b2b_optimize.py --logger tensorboard   # fallback to TB

Logging:
  Default: Weights & Biases (wandb) — dashboards at wandb.ai
  Alt:     tensorboard --logdir runs/
"""

import numpy as np
import sys
import os
import io
import json
import time
import argparse
import contextlib
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

from b2b_test import set_weights, DEFAULTS
from TetrisEnv.PyTetrisEnv import PyTetrisEnv
from TetrisEnv.CB2BSearch import CB2BSearch


# ── Weight configuration ─────────────────────────────────────

WEIGHT_NAMES = list(DEFAULTS.keys())
N = len(WEIGHT_NAMES)
BASELINE = np.array([DEFAULTS[k] for k in WEIGHT_NAMES], dtype=np.float64)

# Hard bounds in real space
W_LO = 0.05
W_HI = 60.0


def vec_to_dict(v):
    """Weight vector → dict for set_weights()."""
    return {k: float(v[i]) for i, k in enumerate(WEIGHT_NAMES)}


# ── Stdout suppression (hides PyTetrisEnv "Initialized Env" spam) ─

@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


# ── Game runner with B2B tracking ─────────────────────────────

def run_eval_game(seed, num_steps=100, search_depth=4, beam_width=64,
                  garbage_chance=0.15, garbage_min=1, garbage_max=4):
    """Run one game, return stats dict including max_b2b."""
    with suppress_stdout():
        env = PyTetrisEnv(
            queue_size=5, max_holes=None, max_height=20,
            max_steps=None, max_len=15, pathfinding=False,
            seed=seed, idx=0,
            garbage_chance=garbage_chance, garbage_min=garbage_min,
            garbage_max=garbage_max, auto_push_garbage=True,
            auto_fill_queue=True,
        )
        env.reset()
    searcher = CB2BSearch()

    total_attack = 0.0
    heights = []
    max_b2b = 0
    step = 0
    game_over = False

    for step in range(1, num_steps + 1):
        if game_over:
            break

        b2b = env._scorer._b2b
        if b2b > max_b2b:
            max_b2b = b2b

        board = env._board
        active = env._active_piece.piece_type.value
        hold = env._hold_piece.value
        queue_arr = np.array([p.value for p in env._queue], dtype=np.int32)
        combo = env._scorer._combo
        total_garb = env._get_total_garbage()

        action_idx, seq = searcher.search(
            board=board, active_piece=active, hold_piece=hold,
            queue=queue_arr, b2b=b2b, combo=combo,
            total_garbage=total_garb, search_depth=search_depth,
            beam_width=beam_width, max_len=15,
        )

        if action_idx < 0:
            game_over = True
            break

        ts_result = env._step(seq)
        total_attack += float(ts_result.reward["attack"])

        h = 0
        for r in range(24):
            if np.any(env._board[r] != 0):
                h = 24 - r
                break
        heights.append(h)

        if ts_result.is_last():
            game_over = True

    end_height = heights[-1] if heights else 0
    if game_over and not (step >= num_steps):
        end_height = 20  # died → treat as max height

    return {
        "steps": step,
        "survived": not game_over,
        "total_attack": total_attack,
        "attack_per_piece": total_attack / max(step, 1),
        "avg_height": float(np.mean(heights)) if heights else 0.0,
        "max_height": max(heights) if heights else 0,
        "end_height": end_height,
        "max_b2b": max_b2b,
    }


# ── Candidate evaluation ─────────────────────────────────────

def evaluate(weights_vec, garb_seeds, no_garb_seeds, num_steps=100, depth=4):
    """Full evaluation of a weight vector on shared seeds."""
    set_weights(**vec_to_dict(weights_vec))
    results = []

    for s in garb_seeds:
        results.append(run_eval_game(
            s, num_steps, depth,
            garbage_chance=0.15, garbage_min=1, garbage_max=4))

    for s in no_garb_seeds:
        results.append(run_eval_game(
            s, num_steps, depth,
            garbage_chance=0.0, garbage_min=0, garbage_max=0))

    return {
        'survival_rate':  float(np.mean([r['survived'] for r in results])),
        'avg_max_b2b':    float(np.mean([r['max_b2b'] for r in results])),
        'avg_app':        float(np.mean([r['attack_per_piece'] for r in results])),
        'avg_height':     float(np.mean([r['avg_height'] for r in results])),
        'avg_max_height': float(np.mean([r['max_height'] for r in results])),
        'avg_end_height': float(np.mean([r['end_height'] for r in results])),
    }


# ── Fitness function ──────────────────────────────────────────

def compute_fitness(metrics, weights_vec, reg_lambda):
    """
    Scalar fitness with lexicographic priorities + regularisation.

    survive >> b2b >> app >> low end-height.  The 100× gaps between
    tiers make them effectively lexicographic.  End-height penalty
    sits within the APP tier as a tiebreaker that discourages
    upstacking — a candidate with clean low boards beats one that
    achieves the same APP at dangerous heights.
    """
    raw = (10000.0 * metrics['survival_rate']
         +   100.0 * metrics['avg_max_b2b']
         +    10.0 * metrics['avg_app']
         -     0.5 * metrics['avg_end_height'])

    # L2 regularisation in log-space against baseline
    log_dev = np.log(weights_vec + 1e-8) - np.log(BASELINE + 1e-8)
    reg = reg_lambda * float(np.sum(log_dev ** 2))

    return raw - reg


# ── Parallel worker ──────────────────────────────────────────

def _eval_worker(args):
    """
    Process-pool worker: evaluate one candidate.

    Each forked process has its own copy of the C extension's static
    globals, so set_weights() calls are process-safe with no locking.
    """
    weights_vec, garb_seeds, no_garb_seeds, num_steps, depth, reg_lambda = args
    m = evaluate(weights_vec, garb_seeds, no_garb_seeds, num_steps, depth)
    f = compute_fitness(m, weights_vec, reg_lambda)
    return m, f


# ── CMA-ES Optimiser ─────────────────────────────────────────

class CMAES:
    """
    Covariance Matrix Adaptation Evolution Strategy.

    Optimises in log-space so all weights stay positive.
    Follows Hansen (2016) "The CMA Evolution Strategy: A Tutorial".
    """

    def __init__(self, n, pop_size=None, sigma0=0.3, rng_seed=42):
        self.n = n
        self.lam = pop_size or (4 + int(3 * np.log(n)))  # ~13 for n=19
        self.mu = self.lam // 2
        self.rng = np.random.RandomState(rng_seed)

        # Mean in log-space (start at baseline)
        self.mean = np.log(BASELINE + 1e-8)

        # ── Recombination weights ─────────────────────────────
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = raw_w / raw_w.sum()
        self.mu_eff = 1.0 / np.sum(self.w ** 2)

        # ── Step-size control parameters ──────────────────────
        self.sigma = sigma0
        self.c_sig = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sig = (1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1)
                      + self.c_sig)
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        # ── Covariance adaptation parameters ──────────────────
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1 - self.c_1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff)
            / ((n + 2) ** 2 + self.mu_eff),
        )

        # ── Mutable state ─────────────────────────────────────
        self.p_sig = np.zeros(n)          # evolution path for sigma
        self.p_c = np.zeros(n)            # evolution path for C
        self.C = np.eye(n)                # covariance matrix
        self.B = np.eye(n)                # eigenvectors of C
        self.D = np.ones(n)               # sqrt(eigenvalues) of C
        self.inv_sqrt_C = np.eye(n)       # C^{-1/2}
        self.gen = 0

        # ── Best-ever tracking ────────────────────────────────
        self.best_fit = -np.inf
        self.best_vec = BASELINE.copy()
        self.best_met = {}

    # ── Eigen decomposition ───────────────────────────────────

    def _eigen_update(self):
        """Recompute eigenvectors/values from C (symmetrise first)."""
        self.C = (self.C + self.C.T) / 2
        eigvals, self.B = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)
        self.D = np.sqrt(eigvals)
        self.inv_sqrt_C = self.B @ np.diag(1.0 / self.D) @ self.B.T

    # ── Conversions ───────────────────────────────────────────

    def to_real(self, log_vec):
        """Log-space vector → clamped real-space weights."""
        return np.exp(np.clip(log_vec, np.log(W_LO), np.log(W_HI)))

    # ── Sampling ──────────────────────────────────────────────

    def ask(self):
        """Sample λ candidates. Returns list of log-space vectors."""
        # Refresh eigenvectors periodically
        if self.gen == 0 or self.gen % max(1, self.n // 5) == 0:
            self._eigen_update()

        log_cands = []

        # Inject baseline as first candidate on gen 0
        if self.gen == 0:
            log_cands.append(np.log(BASELINE + 1e-8).copy())

        while len(log_cands) < self.lam:
            z = self.rng.randn(self.n)
            x = self.mean + self.sigma * (self.B @ (self.D * z))
            x = np.clip(x, np.log(W_LO), np.log(W_HI))
            log_cands.append(x)

        return log_cands[:self.lam]

    # ── Update ────────────────────────────────────────────────

    def tell(self, log_cands, fits):
        """Update distribution from ranked fitness values."""
        order = np.argsort(fits)[::-1]  # best-first
        sel = [log_cands[order[i]] for i in range(self.mu)]

        old_mean = self.mean.copy()
        self.mean = sum(self.w[i] * sel[i] for i in range(self.mu))

        diff = (self.mean - old_mean) / self.sigma

        # ── Step-size evolution path ──────────────────────────
        self.p_sig = (
            (1 - self.c_sig) * self.p_sig
            + np.sqrt(self.c_sig * (2 - self.c_sig) * self.mu_eff)
            * (self.inv_sqrt_C @ diff)
        )

        # ── Stall indicator h_sigma ───────────────────────────
        ps_norm = np.linalg.norm(self.p_sig)
        threshold = (
            (1.4 + 2 / (self.n + 1)) * self.chi_n
            * np.sqrt(1 - (1 - self.c_sig) ** (2 * (self.gen + 1)))
        )
        h_sig = 1.0 if ps_norm < threshold else 0.0

        # ── Covariance evolution path ─────────────────────────
        self.p_c = (
            (1 - self.c_c) * self.p_c
            + h_sig * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
            * diff
        )

        # ── Rank-μ update vectors ─────────────────────────────
        y = np.array([
            (sel[i] - old_mean) / self.sigma for i in range(self.mu)
        ])

        # ── Covariance matrix update ─────────────────────────
        self.C = (
            (1 - self.c_1 - self.c_mu
             + (1 - h_sig) * self.c_1 * self.c_c * (2 - self.c_c))
            * self.C
            + self.c_1 * np.outer(self.p_c, self.p_c)
            + self.c_mu * sum(
                self.w[i] * np.outer(y[i], y[i]) for i in range(self.mu))
        )

        # ── Sigma adaptation ─────────────────────────────────
        self.sigma *= np.exp(
            (self.c_sig / self.d_sig) * (ps_norm / self.chi_n - 1)
        )
        self.sigma = np.clip(self.sigma, 0.005, 3.0)

        self.gen += 1

    # ── Serialisation ─────────────────────────────────────────

    def state_dict(self):
        d = {}
        for k in ('mean', 'C', 'p_sig', 'p_c', 'B', 'D', 'best_vec'):
            d[k] = getattr(self, k).tolist()
        d['sigma'] = float(self.sigma)
        d['gen'] = self.gen
        d['best_fit'] = float(self.best_fit)
        d['best_met'] = self.best_met
        return d

    def load_state(self, d):
        for k in ('mean', 'C', 'p_sig', 'p_c', 'B', 'D', 'best_vec'):
            setattr(self, k, np.array(d[k]))
        self.sigma = float(d['sigma'])
        self.gen = int(d['gen'])
        self.best_fit = float(d['best_fit'])
        self.best_met = d['best_met']
        self._eigen_update()


# ── Main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='CMA-ES weight optimiser for b2b_search heuristic')
    ap.add_argument('--generations', type=int, default=100,
                    help='Number of generations to run')
    ap.add_argument('--pop-size', type=int, default=None,
                    help='Population size (default: 4+3·ln(n) ≈ 13)')
    ap.add_argument('--sigma', type=float, default=0.3,
                    help='Initial step size in log-space (default: 0.3 ≈ ±35%%)')
    ap.add_argument('--reg', type=float, default=5.0,
                    help='L2 regularisation strength (default: 5.0)')
    ap.add_argument('--steps', type=int, default=64,
                    help='Steps per game')
    ap.add_argument('--garb-games', type=int, default=3,
                    help='Garbage games per candidate evaluation')
    ap.add_argument('--no-garb-games', type=int, default=2,
                    help='No-garbage games per candidate evaluation')
    ap.add_argument('--depth', type=int, default=4,
                    help='Beam search depth')
    ap.add_argument('--seed', type=int, default=42,
                    help='RNG seed for reproducibility')
    ap.add_argument('--resume', type=str, default=None,
                    help='Path to run directory to resume from')
    ap.add_argument('--run-dir', type=str, default=None,
                    help='Explicit output directory (default: auto-timestamped)')
    ap.add_argument('--workers', type=int, default=None,
                    help='Parallel workers (default: cpu_count)')
    ap.add_argument('--logger', choices=['wandb', 'tensorboard'],
                    default='wandb',
                    help='Logging backend (default: wandb)')
    args = ap.parse_args()

    # ── Run directory ─────────────────────────────────────────
    if args.resume:
        run_dir = Path(args.resume)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(f'runs/b2b_opt_{stamp}')
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(run_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Logger setup ─────────────────────────────────────────
    if args.logger == 'wandb':
        import wandb
        wandb.init(
            project='b2b-optimize',
            dir=str(run_dir),
            config=vars(args),
            name=run_dir.name,
            resume='allow' if args.resume else None,
        )
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        writer = tf.summary.create_file_writer(str(run_dir))

    # Worker pool
    n_workers = args.workers or mp.cpu_count() or 1

    # ── Optimiser ─────────────────────────────────────────────
    opt = CMAES(N, pop_size=args.pop_size, sigma0=args.sigma, rng_seed=args.seed)
    start_gen = 0

    if args.resume and (run_dir / 'checkpoint.json').exists():
        with open(run_dir / 'checkpoint.json') as f:
            opt.load_state(json.load(f))
        start_gen = opt.gen
        print(f'Resumed from gen {start_gen}, best fitness {opt.best_fit:.1f}')

    # Separate RNG for game seeds (deterministic, independent of CMA-ES RNG)
    seed_rng = np.random.RandomState(args.seed + 7777)

    total_games = args.garb_games + args.no_garb_games
    print(f'CMA-ES  n={N}  λ={opt.lam}  μ={opt.mu}  σ₀={opt.sigma:.2f}'
          f'  workers={n_workers}')
    print(f'Games/candidate: {total_games}  ({args.garb_games}g + {args.no_garb_games}ng)'
          f'  steps: {args.steps}  depth: {args.depth}')
    print(f'Regularisation: {args.reg}')
    print(f'Baseline weights: {dict(zip(WEIGHT_NAMES, BASELINE))}')
    print(f'Run dir: {run_dir}')
    print(flush=True)

    # ── Generation loop ───────────────────────────────────────
    pool = mp.Pool(n_workers) if n_workers > 1 else None

    try:
      for gen in range(start_gen, args.generations):
        t0 = time.time()

        # Shared seeds for this generation (fair comparison)
        g_seeds = seed_rng.randint(0, 100_000, args.garb_games).tolist()
        n_seeds = seed_rng.randint(0, 100_000, args.no_garb_games).tolist()

        log_cands = opt.ask()
        real_cands = [opt.to_real(lc) for lc in log_cands]

        # Build work items: (weights_vec, garb_seeds, no_garb_seeds, steps, depth, reg)
        work = [
            (rv, g_seeds, n_seeds, args.steps, args.depth, args.reg)
            for rv in real_cands
        ]

        fits = np.empty(opt.lam)
        mets = []

        if pool is not None:
            # Parallel: imap preserves order and lets us show progress
            for i, (m, f) in enumerate(pool.imap(_eval_worker, work)):
                fits[i] = f
                mets.append(m)
                sys.stderr.write(f'\r  gen {gen}  [{i + 1}/{opt.lam}]')
                sys.stderr.flush()
        else:
            # Sequential fallback (--workers 1)
            for i, item in enumerate(work):
                m, f = _eval_worker(item)
                fits[i] = f
                mets.append(m)
                sys.stderr.write(f'\r  gen {gen}  [{i + 1}/{opt.lam}]')
                sys.stderr.flush()

        sys.stderr.write('\r' + ' ' * 40 + '\r')

        # Update distribution
        opt.tell(log_cands, fits)

        # Track best-ever
        bi = int(np.argmax(fits))
        new_best = fits[bi] > opt.best_fit
        if new_best:
            opt.best_fit = fits[bi]
            opt.best_vec = real_cands[bi].copy()
            opt.best_met = mets[bi]

        bm = mets[bi]  # best this generation
        mm = {k: float(np.mean([m[k] for m in mets])) for k in bm}

        # ── Logging ────────────────────────────────────────────
        cond = float(np.max(opt.D) ** 2 / max(np.min(opt.D) ** 2, 1e-30))

        log_data = {
            # Fitness
            'fitness/gen_best':  fits[bi],
            'fitness/gen_mean':  float(np.mean(fits)),
            'fitness/gen_worst': float(np.min(fits)),
            'fitness/best_ever': opt.best_fit,
            # Best-in-gen metrics
            'best/survival':   bm['survival_rate'],
            'best/max_b2b':    bm['avg_max_b2b'],
            'best/app':        bm['avg_app'],
            'best/avg_height': bm['avg_height'],
            'best/max_height': bm['avg_max_height'],
            'best/end_height': bm['avg_end_height'],
            # Population-mean metrics
            'mean/survival':   mm['survival_rate'],
            'mean/max_b2b':    mm['avg_max_b2b'],
            'mean/app':        mm['avg_app'],
            'mean/avg_height': mm['avg_height'],
            'mean/end_height': mm['avg_end_height'],
            # Optimiser internals
            'opt/sigma':       opt.sigma,
            'opt/cond_number': cond,
        }
        # Per-weight values (best in gen + best ever)
        for j, name in enumerate(WEIGHT_NAMES):
            log_data[f'w_gen/{name}']  = float(real_cands[bi][j])
            log_data[f'w_best/{name}'] = float(opt.best_vec[j])

        if args.logger == 'wandb':
            wandb.log(log_data, step=gen)
        else:
            with writer.as_default():
                for key, val in log_data.items():
                    tf.summary.scalar(key, val, step=gen)
            writer.flush()

        dt = time.time() - t0
        marker = ' ★' if new_best else ''
        print(
            f'gen {gen:3d} │ fit {fits[bi]:8.1f} (μ {np.mean(fits):8.1f}) │ '
            f'surv {bm["survival_rate"]:.0%}  b2b {bm["avg_max_b2b"]:4.1f}  '
            f'app {bm["avg_app"]:.3f}  endH {bm["avg_end_height"]:4.1f} │ '
            f'σ {opt.sigma:.3f}  κ {cond:.0f} │ {dt:.0f}s{marker}'
        )

        # ── Checkpoint ────────────────────────────────────────
        # Save best_weights.json immediately on new best
        if new_best:
            with open(run_dir / 'best_weights.json', 'w') as f:
                json.dump(vec_to_dict(opt.best_vec), f, indent=2)

        # Full checkpoint every 5 generations
        if (gen + 1) % 5 == 0 or gen == args.generations - 1:
            with open(run_dir / 'checkpoint.json', 'w') as f:
                json.dump(opt.state_dict(), f)
            with open(run_dir / 'best_weights.json', 'w') as f:
                json.dump(vec_to_dict(opt.best_vec), f, indent=2)

    finally:
      if pool is not None:
          pool.close()
          pool.join()

    # ── Final report ──────────────────────────────────────────
    print()
    print('=' * 70)
    print('OPTIMISATION COMPLETE')
    print('=' * 70)
    bm = opt.best_met
    print(f'Best fitness : {opt.best_fit:.1f}')
    print(f'Survival     : {bm.get("survival_rate", 0):.0%}')
    print(f'Max B2B      : {bm.get("avg_max_b2b", 0):.1f}')
    print(f'Attack/piece : {bm.get("avg_app", 0):.3f}')
    print(f'End height   : {bm.get("avg_end_height", 0):.1f}')
    print()
    print('Best weights (paste into b2b_test.py DEFAULTS):')
    print('DEFAULTS = dict(')
    for i, name in enumerate(WEIGHT_NAMES):
        v = opt.best_vec[i]
        d = BASELINE[i]
        pct = (v / d - 1) * 100 if d > 0 else float('inf')
        print(f'    {name}={v:.2f},  # baseline {d:.1f} ({pct:+.0f}%)')
    print(')')
    if args.logger == 'wandb':
        print(f'\nW&B dashboard: {wandb.run.url}')
        wandb.finish()
    else:
        print(f'\nTensorBoard:  tensorboard --logdir {run_dir.parent}/')
    print(f'Checkpoint:   {run_dir / "checkpoint.json"}')
    print(f'Best weights: {run_dir / "best_weights.json"}')


if __name__ == '__main__':
    main()
