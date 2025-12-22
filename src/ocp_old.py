# ocp.py
from __future__ import annotations

import itertools
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("ocp")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class WindowStats:
    rows_before: int
    rows_after: int
    dropped_rows: int
    tickers_before: int
    tickers_after: int
    coverage_threshold: float


@dataclass(frozen=True)
class OCPResult:
    formation_date: str  # YYYY-MM-DD
    leader: str
    follower: str
    lag_initial: int
    lag_hat: float
    sigma_lag: float
    cost: float
    path_len: int
    band: int
    window_rows_used: int


# ============================================================
# Helpers: pairs + array checks
# ============================================================

def generate_ticker_pairs(tickers: Sequence[str]) -> List[Tuple[str, str]]:
    """All unordered pairs (i<j)."""
    return list(itertools.combinations(sorted(tickers), 2))


def _ensure_1d_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {a.shape}")
    return a.astype(np.float64, copy=False)


# ============================================================
# Daily boundaries
# ============================================================

def build_trading_days(panel: pl.DataFrame) -> pl.DataFrame:
    """
    Return a DataFrame with one row per date:
      columns: date (Date), start_ts (Datetime), end_ts (Datetime), n_rows (int)

    Assumes panel has a 'timestamp' column.
    """
    if "timestamp" not in panel.columns:
        raise ValueError("panel must contain a 'timestamp' column")

    days = (
        panel
        .select(pl.col("timestamp"))
        .sort("timestamp")
        .with_columns(pl.col("timestamp").dt.date().alias("date"))
        .group_by("date")
        .agg([
            pl.col("timestamp").min().alias("start_ts"),
            pl.col("timestamp").max().alias("end_ts"),
            pl.len().alias("n_rows"),
        ])
        .sort("date")
    )
    return days


# ============================================================
# Formation window preparation (coverage filter + dense rows)
# ============================================================

def prepare_formation_window(
    panel: pl.DataFrame,
    tickers: List[str],
    start_ts,
    end_ts,
    coverage_threshold: float = 0.99,
    min_tickers: int = 20,
    min_rows: int = 300,
) -> Tuple[pl.DataFrame, List[str], WindowStats]:
    """
    Prepare a dense formation window for OCP:

    - slice panel to [start_ts, end_ts)
    - compute per-ticker coverage over the slice
    - keep tickers with coverage >= coverage_threshold
    - drop any rows with nulls in any kept ticker

    Returns:
      dense_window_panel (timestamp + kept tickers),
      kept_tickers,
      stats

    Raises ValueError if too few tickers or rows remain.
    """
    if "timestamp" not in panel.columns:
        raise ValueError("Panel must contain a 'timestamp' column.")

    missing_cols = [t for t in tickers if t not in panel.columns]
    if missing_cols:
        raise ValueError(
            f"Panel missing ticker columns: {missing_cols[:10]}"
            f"{'...' if len(missing_cols) > 10 else ''}"
        )

    w = (
        panel
        .filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))
        .select(["timestamp"] + tickers)
        .sort("timestamp")
    )

    rows_before = w.height
    if rows_before == 0:
        raise ValueError("Formation window slice is empty.")

    # Compute null counts per ticker (one row DataFrame -> tuple)
    null_counts = w.select([pl.col(t).null_count().alias(t) for t in tickers]).row(0)

    kept: List[str] = []
    for t, nc in zip(tickers, null_counts):
        coverage = 1.0 - float(nc) / float(rows_before)
        if coverage >= coverage_threshold:
            kept.append(t)

    if len(kept) < min_tickers:
        raise ValueError(f"Too few tickers after coverage filtering ({len(kept)} < {min_tickers}).")

    w_dense = w.select(["timestamp"] + kept).drop_nulls(subset=kept)
    rows_after = w_dense.height

    if rows_after < min_rows:
        raise ValueError(f"Too few rows after dropping nulls ({rows_after} < {min_rows}).")

    stats = WindowStats(
        rows_before=rows_before,
        rows_after=rows_after,
        dropped_rows=rows_before - rows_after,
        tickers_before=len(tickers),
        tickers_after=len(kept),
        coverage_threshold=float(coverage_threshold),
    )
    return w_dense, kept, stats


def extract_arrays_from_window(window_panel: pl.DataFrame, tickers: List[str]) -> Dict[str, np.ndarray]:
    """Convert a dense window panel into numpy arrays per ticker."""
    out: Dict[str, np.ndarray] = {}
    for t in tickers:
        out[t] = window_panel.get_column(t).to_numpy().astype(np.float64, copy=False)
    return out


# ============================================================
# Pair prefilter (fast): absolute correlation
# ============================================================

def prefilter_pairs_by_abs_corr(
    window_panel_dense: pl.DataFrame,
    tickers: List[str],
    max_pairs: int = 500,
) -> List[Tuple[str, str]]:
    """
    Reduce compute by keeping only max_pairs pairs with highest |corr|
    computed on the dense formation window.
    """
    if max_pairs <= 0:
        return generate_ticker_pairs(tickers)

    X = window_panel_dense.select(tickers).to_numpy()
    C = np.corrcoef(X, rowvar=False)

    n = C.shape[0]
    scored: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = abs(C[i, j])
            if np.isfinite(s):
                scored.append((s, i, j))

    scored.sort(reverse=True, key=lambda z: z[0])
    scored = scored[: min(max_pairs, len(scored))]

    return [(tickers[i], tickers[j]) for _, i, j in scored]


# ============================================================
# OCP Step A: constant lag
# ============================================================

def ocp_step_a_constant_lag(x: np.ndarray, y: np.ndarray, max_lag: Optional[int] = None) -> int:
    """
    Choose lag l >= 0 minimizing sum_{i=1..M} |x_{i+l} - y_i|, requiring N>=M.
    """
    x = _ensure_1d_float(x)
    y = _ensure_1d_float(y)
    N, M = len(x), len(y)

    if N < M:
        raise ValueError(f"Step A expects N >= M (got N={N}, M={M}).")

    Lmax = N - M
    if max_lag is not None:
        Lmax = min(Lmax, int(max_lag))
    Lmax = max(Lmax, 0)

    best_l = 0
    best_cost = np.inf

    for l in range(Lmax + 1):
        xs = x[l:l + M]
        mask = np.isfinite(xs) & np.isfinite(y)
        if not np.any(mask):
            continue
        cost = float(np.sum(np.abs(xs[mask] - y[mask])))
        if cost < best_cost:
            best_cost = cost
            best_l = l

    return int(best_l)


# ============================================================
# OCP Step B: banded dynamic programming path
# ============================================================

def ocp_step_b_optimal_path(
    x: np.ndarray,
    y: np.ndarray,
    band: int,
    max_cells: int = 25_000_000,
) -> Tuple[np.ndarray, float]:
    """
    Banded DP for optimal causal path with moves:
      (1,0), (0,1), (1,1)

    Only compute cells where |n - m| <= band.

    Returns:
      path (I,2) with indices (n,m),
      total_cost (sum |x[n]-y[m]| over path)
    """
    x = _ensure_1d_float(x)
    y = _ensure_1d_float(y)

    N, M = len(x), len(y)
    if N == 0 or M == 0:
        raise ValueError("Empty series passed to Step B.")
    if band <= 0:
        raise ValueError("band must be a positive integer (e.g. 20..60).")

    approx_cells = (2 * band + 1) * max(N, M)
    if approx_cells > max_cells:
        raise MemoryError(
            f"Refusing DP: approx_cells={approx_cells:,} exceeds max_cells={max_cells:,}. "
            f"Use smaller windows and/or a smaller band."
        )

    dp = np.full((N, M), np.inf, dtype=np.float64)
    prev = np.full((N, M, 2), -1, dtype=np.int32)

    def in_band(n: int, m: int) -> bool:
        return abs(n - m) <= band

    for n in range(N):
        m_start = max(0, n - band)
        m_end = min(M - 1, n + band)

        xn = x[n]
        for m in range(m_start, m_end + 1):
            ym = y[m]
            if not (np.isfinite(xn) and np.isfinite(ym)):
                continue
            c = abs(xn - ym)

            if n == 0 and m == 0:
                dp[n, m] = c
                prev[n, m] = (-1, -1)
                continue

            best_val = np.inf
            best_prev = (-1, -1)

            # from (n-1, m)
            if n - 1 >= 0 and in_band(n - 1, m):
                v = dp[n - 1, m]
                if v < best_val:
                    best_val = v
                    best_prev = (n - 1, m)

            # from (n, m-1)
            if m - 1 >= 0 and in_band(n, m - 1):
                v = dp[n, m - 1]
                if v < best_val:
                    best_val = v
                    best_prev = (n, m - 1)

            # from (n-1, m-1)
            if n - 1 >= 0 and m - 1 >= 0 and in_band(n - 1, m - 1):
                v = dp[n - 1, m - 1]
                if v < best_val:
                    best_val = v
                    best_prev = (n - 1, m - 1)

            if np.isfinite(best_val):
                dp[n, m] = best_val + c
                prev[n, m] = best_prev

    if not np.isfinite(dp[N - 1, M - 1]):
        raise RuntimeError("No valid DP path found (band too tight or too many invalid values).")

    # Backtrack
    path_rev: List[Tuple[int, int]] = []
    n, m = N - 1, M - 1
    while n >= 0 and m >= 0:
        path_rev.append((n, m))
        pn, pm = prev[n, m]
        if pn == -1 and pm == -1:
            break
        n, m = int(pn), int(pm)

    path = np.array(path_rev[::-1], dtype=np.int32)
    total_cost = float(dp[N - 1, M - 1])
    return path, total_cost


# ============================================================
# OCP Step C: lag stats
# ============================================================

def ocp_step_c_lag_stats(path: np.ndarray) -> Tuple[float, float]:
    """
    lag_i = n_i - m_i
    lag_hat = mean(lag_i)
    sigma_lag = sqrt(mean((lag_i - lag_hat)^2))
    """
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"Expected path shape (I,2), got {path.shape}")
    lags = path[:, 0].astype(np.float64) - path[:, 1].astype(np.float64)
    lag_hat = float(np.mean(lags))
    sigma_lag = float(np.sqrt(np.mean((lags - lag_hat) ** 2)))
    return lag_hat, sigma_lag


# ============================================================
# OCP for one pair (directed) and direction selection
# ============================================================

def ocp_run_directed_pair(
    x: np.ndarray,
    y: np.ndarray,
    band: int,
    max_lag_step_a: Optional[int] = None,
) -> Tuple[int, float, float, float, int]:
    """
    Run Step A/B/C for x->y direction.
    Returns: (lag_initial, lag_hat, sigma_lag, cost, path_len)
    """
    x = _ensure_1d_float(x)
    y = _ensure_1d_float(y)

    L = min(len(x), len(y))
    if L == 0:
        raise ValueError("Empty arrays for pair.")
    x = x[:L]
    y = y[:L]

    lag_initial = ocp_step_a_constant_lag(x, y, max_lag=max_lag_step_a)
    path, cost = ocp_step_b_optimal_path(x, y, band=band)
    lag_hat, sigma_lag = ocp_step_c_lag_stats(path)

    return int(lag_initial), float(lag_hat), float(sigma_lag), float(cost), int(path.shape[0])


def choose_direction_by_stability(
    xa: np.ndarray,
    xb: np.ndarray,
    a: str,
    b: str,
    band: int,
    max_lag_step_a: Optional[int] = None,
) -> Tuple[str, str, int, float, float, float, int]:
    """
    Run both a->b and b->a, keep the one with smaller sigma_lag.
    Returns: leader, follower, lag_initial, lag_hat, sigma_lag, cost, path_len
    """
    r_ab = ocp_run_directed_pair(xa, xb, band=band, max_lag_step_a=max_lag_step_a)
    r_ba = ocp_run_directed_pair(xb, xa, band=band, max_lag_step_a=max_lag_step_a)

    if r_ab[2] <= r_ba[2]:
        lag_initial, lag_hat, sigma_lag, cost, path_len = r_ab
        return a, b, lag_initial, lag_hat, sigma_lag, cost, path_len
    else:
        lag_initial, lag_hat, sigma_lag, cost, path_len = r_ba
        return b, a, lag_initial, lag_hat, sigma_lag, cost, path_len


# ============================================================
# Run OCP on many pairs for one formation window
# ============================================================

def run_ocp_on_pairs(
    arrays_by_ticker: Dict[str, np.ndarray],
    pairs: Iterable[Tuple[str, str]],
    formation_date: str,
    band: int,
    top_k: int = 10,
    require_nonzero_lag: bool = True,
    max_lag_step_a: Optional[int] = None,
    window_rows_used: Optional[int] = None,
) -> List[OCPResult]:
    results: List[OCPResult] = []

    for a, b in pairs:
        xa = arrays_by_ticker[a]
        xb = arrays_by_ticker[b]

        leader, follower, lag_initial, lag_hat, sigma_lag, cost, path_len = choose_direction_by_stability(
            xa, xb, a, b, band=band, max_lag_step_a=max_lag_step_a
        )

        if require_nonzero_lag and abs(lag_hat) < 1e-12:
            continue

        results.append(
            OCPResult(
                formation_date=formation_date,
                leader=leader,
                follower=follower,
                lag_initial=lag_initial,
                lag_hat=lag_hat,
                sigma_lag=sigma_lag,
                cost=cost,
                path_len=path_len,
                band=int(band),
                window_rows_used=int(window_rows_used) if window_rows_used is not None else -1,
            )
        )

    results.sort(key=lambda r: r.sigma_lag)
    return results[:top_k]


# ============================================================
# Checkpointing
# ============================================================

def _run_signature(
    coverage_threshold: float,
    min_tickers: int,
    min_rows_per_day: int,
    min_rows_after_clean: int,
    band: int,
    top_k: int,
    max_pairs_prefilter: int,
    require_nonzero_lag: bool,
    max_lag_step_a: Optional[int],
) -> str:
    payload = {
        "coverage_threshold": coverage_threshold,
        "min_tickers": min_tickers,
        "min_rows_per_day": min_rows_per_day,
        "min_rows_after_clean": min_rows_after_clean,
        "band": band,
        "top_k": top_k,
        "max_pairs_prefilter": max_pairs_prefilter,
        "require_nonzero_lag": require_nonzero_lag,
        "max_lag_step_a": max_lag_step_a,
    }
    return json.dumps(payload, sort_keys=True)


def _checkpoint_paths(checkpoint_path: str) -> Tuple[Path, Path]:
    p = Path(checkpoint_path)
    meta = p.with_suffix(p.suffix + ".meta.json")
    return p, meta


def load_checkpoint_done_dates(checkpoint_path: str) -> Tuple[set, Optional[str]]:
    """
    Returns:
      done_dates: set of formation_date strings already saved
      saved_signature: signature string (or None if not present)
    """
    p, meta = _checkpoint_paths(checkpoint_path)

    done_dates = set()
    saved_signature = None

    if p.exists():
        df = pl.read_parquet(str(p), columns=["formation_date"])
        done_dates = set(df.get_column("formation_date").to_list())

    if meta.exists():
        try:
            saved_signature = json.loads(meta.read_text(encoding="utf-8")).get("signature")
        except Exception:
            saved_signature = None

    return done_dates, saved_signature


def write_checkpoint_metadata(checkpoint_path: str, signature: str) -> None:
    _, meta = _checkpoint_paths(checkpoint_path)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"signature": signature}, indent=2), encoding="utf-8")


def append_checkpoint_rows(checkpoint_path: str, rows_df: pl.DataFrame) -> None:
    """
    Append rows to a parquet checkpoint (safe, simple, good for top_k/day).

    Implementation:
      - if file doesn't exist: write it
      - else: read existing + concat + write tmp + atomic replace
    """
    p, _ = _checkpoint_paths(checkpoint_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        rows_df.write_parquet(str(p))
        return

    existing = pl.read_parquet(str(p))
    combined = pl.concat([existing, rows_df], how="vertical")

    tmp = p.with_suffix(p.suffix + ".tmp")
    combined.write_parquet(str(tmp))
    os.replace(str(tmp), str(p))


def load_full_checkpoint(checkpoint_path: str) -> pl.DataFrame:
    """Load the entire checkpoint parquet (if it exists)."""
    p, _ = _checkpoint_paths(checkpoint_path)
    if not p.exists():
        return pl.DataFrame()
    return pl.read_parquet(str(p))


# ============================================================
# Full daily pipeline (with checkpointing)
# ============================================================

def run_daily_ocp_pipeline(
    panel: pl.DataFrame,
    tickers: List[str],
    coverage_threshold: float = 0.99,
    min_tickers: int = 20,
    min_rows_per_day: int = 300,
    min_rows_after_clean: int = 300,
    band: int = 30,
    top_k: int = 10,
    max_pairs_prefilter: int = 500,
    require_nonzero_lag: bool = True,
    max_lag_step_a: Optional[int] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
) -> pl.DataFrame:
    """
    End-to-end daily loop:

    For each formation day d and the next trading day d+1:
      - formation slice is [start_d, start_{d+1})
      - coverage filter tickers
      - drop any null rows (dense formation window)
      - optionally prefilter pairs by |corr|
      - run OCP and keep top_k stable pairs

    If checkpoint_path is set:
      - append top_k results after each day
      - if resume=True, skip formation dates already present in checkpoint
      - verify a run signature to prevent resuming with changed parameters

    Returns:
      DataFrame of results computed in THIS run (not necessarily the full checkpoint).
      Use load_full_checkpoint(checkpoint_path) to get all saved results.
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    sig = _run_signature(
        coverage_threshold=coverage_threshold,
        min_tickers=min_tickers,
        min_rows_per_day=min_rows_per_day,
        min_rows_after_clean=min_rows_after_clean,
        band=band,
        top_k=top_k,
        max_pairs_prefilter=max_pairs_prefilter,
        require_nonzero_lag=require_nonzero_lag,
        max_lag_step_a=max_lag_step_a,
    )

    done_dates = set()
    saved_sig = None
    if checkpoint_path is not None and resume:
        done_dates, saved_sig = load_checkpoint_done_dates(checkpoint_path)
        if saved_sig is not None and saved_sig != sig:
            raise ValueError(
                "Checkpoint exists but parameters changed.\n"
                "Delete the checkpoint or use a new checkpoint_path."
            )
        if saved_sig is None:
            write_checkpoint_metadata(checkpoint_path, sig)
        if done_dates and verbose:
            logger.info(f"Checkpoint resume: {len(done_dates)} formation days already done. Skipping them.")

    days = build_trading_days(panel).filter(pl.col("n_rows") >= min_rows_per_day).sort("date")
    if days.height < 2:
        raise ValueError("Not enough valid trading days (need at least 2).")

    date_list = days.get_column("date").to_list()
    start_list = days.get_column("start_ts").to_list()

    run_results: List[OCPResult] = []

    for i in range(days.height - 1):
        formation_date = str(date_list[i])
        start_ts = start_list[i]
        end_ts = start_list[i + 1]

        if checkpoint_path is not None and resume and formation_date in done_dates:
            continue

        try:
            w_dense, kept_tickers, stats = prepare_formation_window(
                panel=panel,
                tickers=tickers,
                start_ts=start_ts,
                end_ts=end_ts,
                coverage_threshold=coverage_threshold,
                min_tickers=min_tickers,
                min_rows=min_rows_after_clean,
            )
        except Exception as e:
            logger.warning(f"[{formation_date}] skipped (formation window prep failed): {e}")
            continue

        if verbose:
            logger.info(
                f"[{formation_date}] formation window: rows {stats.rows_before}->{stats.rows_after} "
                f"(dropped {stats.dropped_rows}), tickers {stats.tickers_before}->{stats.tickers_after}"
            )

        arrays = extract_arrays_from_window(w_dense, kept_tickers)

        if max_pairs_prefilter and max_pairs_prefilter > 0:
            pairs = prefilter_pairs_by_abs_corr(w_dense, kept_tickers, max_pairs=max_pairs_prefilter)
            if verbose:
                logger.info(f"[{formation_date}] prefilter pairs: keeping {len(pairs)} pairs by |corr|")
        else:
            pairs = generate_ticker_pairs(kept_tickers)
            if verbose:
                logger.info(f"[{formation_date}] using all pairs: {len(pairs)}")

        top = run_ocp_on_pairs(
            arrays_by_ticker=arrays,
            pairs=pairs,
            formation_date=formation_date,
            band=band,
            top_k=top_k,
            require_nonzero_lag=require_nonzero_lag,
            max_lag_step_a=max_lag_step_a,
            window_rows_used=w_dense.height,
        )

        # Day results -> DataFrame for checkpoint append
        day_df = pl.DataFrame(
            {
                "formation_date": [r.formation_date for r in top],
                "leader": [r.leader for r in top],
                "follower": [r.follower for r in top],
                "lag_initial": [r.lag_initial for r in top],
                "lag_hat": [r.lag_hat for r in top],
                "sigma_lag": [r.sigma_lag for r in top],
                "cost": [r.cost for r in top],
                "path_len": [r.path_len for r in top],
                "band": [r.band for r in top],
                "window_rows_used": [r.window_rows_used for r in top],
            }
        )

        if checkpoint_path is not None:
            append_checkpoint_rows(checkpoint_path, day_df)
            if verbose:
                logger.info(f"[{formation_date}] checkpoint saved: {len(top)} rows -> {checkpoint_path}")

        run_results.extend(top)

    # Return DF for this run
    if not run_results:
        return pl.DataFrame(
            schema={
                "formation_date": pl.Utf8,
                "leader": pl.Utf8,
                "follower": pl.Utf8,
                "lag_initial": pl.Int32,
                "lag_hat": pl.Float64,
                "sigma_lag": pl.Float64,
                "cost": pl.Float64,
                "path_len": pl.Int32,
                "band": pl.Int32,
                "window_rows_used": pl.Int32,
            }
        )

    df = pl.DataFrame(
        {
            "formation_date": [r.formation_date for r in run_results],
            "leader": [r.leader for r in run_results],
            "follower": [r.follower for r in run_results],
            "lag_initial": [r.lag_initial for r in run_results],
            "lag_hat": [r.lag_hat for r in run_results],
            "sigma_lag": [r.sigma_lag for r in run_results],
            "cost": [r.cost for r in run_results],
            "path_len": [r.path_len for r in run_results],
            "band": [r.band for r in run_results],
            "window_rows_used": [r.window_rows_used for r in run_results],
        }
    ).sort(["formation_date", "sigma_lag"])

    return df


# ============================================================
# Example (commented)
# ============================================================

# results_df = run_daily_ocp_pipeline(
#     panel=panel,
#     tickers=tickers,
#     band=30,
#     max_pairs_prefilter=500,
#     top_k=10,
#     checkpoint_path="checkpoints/ocp_daily.parquet",
#     resume=True,
# )
# full_df = load_full_checkpoint("checkpoints/ocp_daily.parquet")
