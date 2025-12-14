from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import polars as pl


# =========================
# Data structures
# =========================

@dataclass(frozen=True)
class OCPResult:
    leader: str
    follower: str
    lag_initial: int
    lag_hat: float
    sigma_lag: float
    path_len: int
    cost: float


# =========================
# Utilities
# =========================

def _ensure_1d_float(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {a.shape}")
    return a.astype(np.float64, copy=False)


def _finite_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


# =========================
# Step A: constant lag
# =========================

def ocp_step_a_constant_lag(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: Optional[int] = None,
) -> int:
    """
    Step A (proposal): constant lag estimation by minimizing:
        c_l = sum_{i=1..M} |x_{i+l} - y_i|
    with l >= 0 (x shifted forward).

    Args:
        x: leader series (length N)
        y: follower series (length M)
        max_lag: optional cap for l (for speed). If None, uses N-M.

    Returns:
        lag_initial (int) >= 0
    """
    x = _ensure_1d_float(x)
    y = _ensure_1d_float(y)

    N, M = len(x), len(y)
    if N < M:
        raise ValueError(f"Step A expects N >= M (got N={N}, M={M})")

    Lmax = N - M
    if max_lag is not None:
        Lmax = min(Lmax, int(max_lag))
    if Lmax < 0:
        Lmax = 0

    best_l = 0
    best_cost = np.inf

    # Compute c_l for l=0..Lmax
    for l in range(Lmax + 1):
        xs = x[l:l + M]
        mask = _finite_mask(xs, y)
        if not np.any(mask):
            continue
        cost = np.sum(np.abs(xs[mask] - y[mask]))
        if cost < best_cost:
            best_cost = cost
            best_l = l

    return best_l


# =========================
# Step B: Optimal Causal Path (DTW-like DP)
# =========================

def ocp_step_b_optimal_path(
    x: np.ndarray,
    y: np.ndarray,
    band: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal path p* minimizing sum |x[n] - y[m]| under moves:
      (1,0), (0,1), (1,1)
    with boundary (0,0) to (N-1,M-1) and monotonicity.

    This is classic DTW with L1 cost, but interpreted as "causal path".

    Args:
        x: series length N
        y: series length M
        band: optional Sakoe-Chiba band width around diagonal for speed.
              If None, full DP O(NM). If int, only compute cells where
              |n - m| <= band (plus boundary reachability).

    Returns:
        path: array of shape (I,2) with indices (n_i, m_i)
        total_cost: float (sum of costs along path)
    """
    x = _ensure_1d_float(x)
    y = _ensure_1d_float(y)
    N, M = len(x), len(y)

    # Cost matrix is implicit: |x[n]-y[m]|
    # DP arrays:
    dp = np.full((N, M), np.inf, dtype=np.float64)
    prev = np.full((N, M, 2), -1, dtype=np.int32)  # store predecessor (pn, pm)

    def in_band(n: int, m: int) -> bool:
        if band is None:
            return True
        return abs(n - m) <= band

    for n in range(N):
        # If banded, limit m range
        if band is None:
            m_start, m_end = 0, M - 1
        else:
            m_start = max(0, n - band)
            m_end = min(M - 1, n + band)

        for m in range(m_start, m_end + 1):
            if not in_band(n, m):
                continue

            c = abs(x[n] - y[m]) if (np.isfinite(x[n]) and np.isfinite(y[m])) else np.inf

            if n == 0 and m == 0:
                dp[n, m] = c
                prev[n, m] = (-1, -1)
                continue

            best_val = np.inf
            best_prev = (-1, -1)

            # from (n-1, m) : move (1,0)
            if n - 1 >= 0 and in_band(n - 1, m):
                v = dp[n - 1, m]
                if v < best_val:
                    best_val = v
                    best_prev = (n - 1, m)

            # from (n, m-1) : move (0,1)
            if m - 1 >= 0 and in_band(n, m - 1):
                v = dp[n, m - 1]
                if v < best_val:
                    best_val = v
                    best_prev = (n, m - 1)

            # from (n-1, m-1) : move (1,1)
            if n - 1 >= 0 and m - 1 >= 0 and in_band(n - 1, m - 1):
                v = dp[n - 1, m - 1]
                if v < best_val:
                    best_val = v
                    best_prev = (n - 1, m - 1)

            if np.isfinite(best_val) and np.isfinite(c):
                dp[n, m] = best_val + c
                prev[n, m] = best_prev

    if not np.isfinite(dp[N - 1, M - 1]):
        raise RuntimeError("No valid DP path found (check band too tight or NaNs).")

    # Backtrack path
    path_rev = []
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


# =========================
# Step C: lag and fluctuation
# =========================

def ocp_step_c_lag_stats(path: np.ndarray) -> Tuple[float, float]:
    """
    Given path indices (n_i, m_i), compute:
      lag_hat = mean(n_i - m_i)
      sigma_lag = std(n_i - m_i)

    Args:
        path: (I,2) int array

    Returns:
        (lag_hat, sigma_lag)
    """
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"Expected path shape (I,2), got {path.shape}")

    lags = path[:, 0].astype(np.float64) - path[:, 1].astype(np.float64)
    lag_hat = float(np.mean(lags))
    sigma_lag = float(np.sqrt(np.mean((lags - lag_hat) ** 2)))
    return lag_hat, sigma_lag


# =========================
# Full OCP for one pair (x leads y)
# =========================

def ocp_run_pair(
    x: np.ndarray,
    y: np.ndarray,
    leader: str,
    follower: str,
    max_lag_step_a: Optional[int] = None,
    band: Optional[int] = None,
) -> OCPResult:
    """
    Run OCP Steps A, B, C for a given direction leader->follower.

    Args:
        x: leader returns series
        y: follower returns series
        leader: leader ticker name
        follower: follower ticker name
        max_lag_step_a: cap on lag search in step A
        band: DP band width for step B (None = full)

    Returns:
        OCPResult
    """
    x = _ensure_1d_float(x)
    y = _ensure_1d_float(y)

    # Step A requires N >= M in the formulation. In practice for equal lengths,
    # it’s fine. If lengths differ, we enforce N >= M by truncating.
    N, M = len(x), len(y)
    if N == 0 or M == 0:
        raise ValueError("Empty series passed to ocp_run_pair.")

    if N < M:
        # truncate y to N (or you could swap; but we keep direction)
        y = y[:N]
        M = len(y)
    elif M < N:
        x = x[:M]
        N = len(x)

    # Now N == M
    lag_initial = ocp_step_a_constant_lag(x, y, max_lag=max_lag_step_a)

    # Step B: full causal path (DTW). (Initialization via lag_initial is not
    # explicitly used here because DP already finds global optimum; you can add
    # a diagonal bias later if you want to match the paper's iterative approach.)
    path, cost = ocp_step_b_optimal_path(x, y, band=band)

    # Step C
    lag_hat, sigma_lag = ocp_step_c_lag_stats(path)

    return OCPResult(
        leader=leader,
        follower=follower,
        lag_initial=int(lag_initial),
        lag_hat=float(lag_hat),
        sigma_lag=float(sigma_lag),
        path_len=int(path.shape[0]),
        cost=float(cost),
    )


# =========================
# Running OCP over a panel window
# =========================

def extract_window_arrays(
    panel: pl.DataFrame,
    tickers: List[str],
    start_ts,
    end_ts,
) -> Dict[str, np.ndarray]:
    """
    Extract returns arrays for each ticker in [start_ts, end_ts) from wide panel.

    panel: columns = ['timestamp', ticker1, ticker2, ...]
    start_ts/end_ts: datetime-like compatible with Polars filtering

    Returns:
        dict ticker -> np.ndarray (float64)
    """
    needed = ["timestamp"] + tickers
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    w = (
        panel
        .filter((pl.col("timestamp") >= start_ts) & (pl.col("timestamp") < end_ts))
        .select(tickers)
    )

    out: Dict[str, np.ndarray] = {}
    for t in tickers:
        out[t] = w.get_column(t).to_numpy().astype(np.float64, copy=False)
    return out


def run_ocp_on_pairs(
    arrays_by_ticker: Dict[str, np.ndarray],
    pairs: Iterable[Tuple[str, str]],
    top_k: int = 10,
    require_nonzero_lag: bool = True,
    max_lag_step_a: Optional[int] = None,
    band: Optional[int] = None,
) -> List[OCPResult]:
    """
    Run OCP for each unordered pair (a,b) in BOTH directions and keep the direction
    that yields a positive lag_hat (meaning leader leads follower on average),
    or the lower sigma if you prefer.

    Selection rule used here:
      - compute OCP(a->b) and OCP(b->a)
      - choose the result with smaller sigma_lag
      - optionally require lag_hat != 0

    Finally returns top_k results by smallest sigma_lag.

    Args:
        arrays_by_ticker: ticker->1D returns array (same length per ticker ideally)
        pairs: iterable of (a,b)
        top_k: number of results to keep
        require_nonzero_lag: filter out lag_hat very close to zero
        max_lag_step_a: cap for step A lag search
        band: DP band width

    Returns:
        List[OCPResult] sorted by sigma_lag asc
    """
    results: List[OCPResult] = []

    for a, b in pairs:
        xa = arrays_by_ticker[a]
        xb = arrays_by_ticker[b]

        r_ab = ocp_run_pair(xa, xb, leader=a, follower=b, max_lag_step_a=max_lag_step_a, band=band)
        r_ba = ocp_run_pair(xb, xa, leader=b, follower=a, max_lag_step_a=max_lag_step_a, band=band)

        # Choose the more stable direction (lower sigma)
        r = r_ab if r_ab.sigma_lag <= r_ba.sigma_lag else r_ba

        if require_nonzero_lag:
            if abs(r.lag_hat) < 1e-9:
                continue

        results.append(r)

    results.sort(key=lambda z: z.sigma_lag)
    return results[:top_k]
