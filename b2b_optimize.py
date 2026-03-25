#!/usr/bin/env python3
"""
CMA-ES weight optimizer for the b2b_search Tetris heuristic.

Algorithm: Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
  - Maintains a multivariate Gaussian over log-space weights
  - Ranks candidates by fitness, updates mean toward the best
  - Learns weight correlations via covariance matrix adaptation
  - Adapts step-size (sigma) automatically

Fitness priorities (lexicographic):
  1. Survival rate          (×10 000)
  2. Worst-case max B2B     (×   100)
  3. Attack/piece           (×    10)
  4. −Worst-case peak height (×  0.5)
  Minus L2 regularisation on log-deviation from baseline.

Features:
  - Full game loop in C (zero Python overhead per step)
  - Log-space optimisation (weights stay positive automatically)
  - L2 regularisation against baseline prevents explosion
  - Shared seeds per generation for fair candidate comparison
  - Baseline injected as candidate on generation 0
  - Auto-resume from latest run (use --new to start fresh)
  - W&B or TensorBoard telemetry + JSON checkpointing

Usage:
  uv run python b2b_optimize.py                        # auto-resumes latest
  uv run python b2b_optimize.py --new                   # force new run
  uv run python b2b_optimize.py --generations 50 --pop-size 20
  uv run python b2b_optimize.py --resume runs/b2b_opt_20240101_120000
  uv run python b2b_optimize.py --logger tensorboard    # fallback to TB

Logging:
  Default: Weights & Biases (wandb) — dashboards at wandb.ai
  Alt:     tensorboard --logdir runs/
"""

import numpy as np
import sys
import os
import json
import time
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

from b2b_test import set_weights, DEFAULTS
from TetrisEnv.CB2BSearch import CB2BSearch, GameConfig, GameResult


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


# ── Fitness helpers ───────────────────────────────────────────

def _aggregate_results(game_dicts):
    """Aggregate per-game result dicts into candidate-level metrics."""
    return {
        'survival_rate':    float(np.mean([r['survived'] for r in game_dicts])),
        'min_max_b2b':      float(np.min([r['max_b2b'] for r in game_dicts])),
        'avg_max_b2b':      float(np.mean([r['max_b2b'] for r in game_dicts])),
        'avg_app':          float(np.mean([
            r['total_attack'] / max(r['steps_completed'], 1)
            for r in game_dicts])),
        'avg_height':       float(np.mean([r['avg_height'] for r in game_dicts])),
        'worst_max_height': float(np.max([r['max_height'] for r in game_dicts])),
        'avg_end_height':   float(np.mean([r['end_height'] for r in game_dicts])),
    }


def compute_fitness(metrics, weights_vec, reg_lambda):
    """
    Scalar fitness with lexicographic priorities + regularisation.

    survive >> b2b >> app >> low peak-height.  The 100× gaps between
    tiers make them effectively lexicographic.

    Uses worst-case (min) b2b and worst-case (max) peak height
    across games so that consistently good candidates beat those
    that are brilliant on some seeds but terrible on others.
    """
    raw = (10000.0 * metrics['survival_rate']
         +   100.0 * metrics['min_max_b2b']
         +    10.0 * metrics['avg_app']
         -     0.5 * metrics['worst_max_height'])

    # L2 regularisation in log-space against baseline
    log_dev = np.log(weights_vec + 1e-8) - np.log(BASELINE + 1e-8)
    reg = reg_lambda * float(np.sum(log_dev ** 2))

    return raw - reg


# ── Parallel worker (one game per task) ──────────────────────

def _game_worker(args):
    """
    Run a SINGLE game for one candidate.

    Work is flattened to (candidate, game) pairs so that all CPUs
    stay busy instead of only λ cores being used.
    """
    ci, weights_vec, seed, garb_chance, garb_min, garb_max, garb_delay, \
        num_steps, depth, beam_width = args
    set_weights(**vec_to_dict(weights_vec))
    searcher = CB2BSearch()
    cfg = GameConfig(seed=seed, garbage_chance=garb_chance,
                     garbage_min=garb_min, garbage_max=garb_max,
                     garbage_push_delay=garb_delay)
    results = searcher.run_eval_games(
        [cfg], num_steps=num_steps,
        search_depth=depth, beam_width=beam_width)
    r = results[0]
    return ci, {
        'survived': r.survived,
        'max_b2b': r.max_b2b,
        'total_attack': r.total_attack,
        'steps_completed': r.steps_completed,
        'avg_height': r.avg_height,
        'max_height': r.max_height,
        'end_height': r.end_height,
    }


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
    ap.add_argument('--new', action='store_true',
                    help='Force a new run (default: auto-resume latest)')
    ap.add_argument('--workers', type=int, default=None,
                    help='Parallel workers (default: cpu_count)')
    ap.add_argument('--beam-width', type=int, default=64,
                    help='Beam width for search (default: 64)')
    ap.add_argument('--logger', choices=['wandb', 'tensorboard'],
                    default='wandb',
                    help='Logging backend (default: wandb)')
    args = ap.parse_args()

    # ── Run directory (auto-resume by default) ─────────────────
    if args.resume:
        run_dir = Path(args.resume)
    elif args.new or args.run_dir:
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = Path(f'runs/b2b_opt_{stamp}')
    else:
        # Auto-resume from latest run with a checkpoint
        run_dir = None
        runs_dir = Path('runs')
        if runs_dir.exists():
            candidates = sorted(runs_dir.glob('b2b_opt_*'))
            for cand in reversed(candidates):
                if (cand / 'checkpoint.json').exists():
                    run_dir = cand
                    break
        if run_dir is None:
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

    if (run_dir / 'checkpoint.json').exists() and not args.new:
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

        # Build flattened work items: one task per (candidate, game)
        work = []
        for ci, rv in enumerate(real_cands):
            for s in g_seeds:
                work.append((ci, rv, s, 0.15, 1, 4, 1,
                             args.steps, args.depth, args.beam_width))
            for s in n_seeds:
                work.append((ci, rv, s, 0.0, 0, 0, 1,
                             args.steps, args.depth, args.beam_width))

        total_tasks = len(work)
        results_by_cand = [[] for _ in range(opt.lam)]

        if pool is not None:
            for done, (ci, r) in enumerate(
                    pool.imap_unordered(_game_worker, work)):
                results_by_cand[ci].append(r)
                sys.stderr.write(
                    f'\r  gen {gen}  [{done + 1}/{total_tasks} games]')
                sys.stderr.flush()
        else:
            for done, item in enumerate(work):
                ci, r = _game_worker(item)
                results_by_cand[ci].append(r)
                sys.stderr.write(
                    f'\r  gen {gen}  [{done + 1}/{total_tasks} games]')
                sys.stderr.flush()

        sys.stderr.write('\r' + ' ' * 50 + '\r')

        # Aggregate per candidate
        fits = np.empty(opt.lam)
        mets = []
        for ci in range(opt.lam):
            m = _aggregate_results(results_by_cand[ci])
            mets.append(m)
            fits[ci] = compute_fitness(m, real_cands[ci], args.reg)

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
            'best/survival':     bm['survival_rate'],
            'best/min_b2b':      bm['min_max_b2b'],
            'best/avg_b2b':      bm['avg_max_b2b'],
            'best/app':          bm['avg_app'],
            'best/avg_height':   bm['avg_height'],
            'best/worst_peak_h': bm['worst_max_height'],
            'best/end_height':   bm['avg_end_height'],
            # Population-mean metrics
            'mean/survival':     mm['survival_rate'],
            'mean/min_b2b':      mm['min_max_b2b'],
            'mean/avg_b2b':      mm['avg_max_b2b'],
            'mean/app':          mm['avg_app'],
            'mean/avg_height':   mm['avg_height'],
            'mean/worst_peak_h': mm['worst_max_height'],
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
            f'surv {bm["survival_rate"]:.0%}  b2b↓ {bm["min_max_b2b"]:4.1f}  '
            f'app {bm["avg_app"]:.3f}  peakH↑ {bm["worst_max_height"]:4.1f} │ '
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
    print(f'Min B2B      : {bm.get("min_max_b2b", 0):.1f}')
    print(f'Attack/piece : {bm.get("avg_app", 0):.3f}')
    print(f'Worst peak H : {bm.get("worst_max_height", 0):.1f}')
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
