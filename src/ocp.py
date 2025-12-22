import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import numba as nb




@dataclass(frozen=True)
class OCPResult:
    leader: str
    follower: str
    l_hat: float
    sigma_l: float
    cost: float
    path_len: int


def build_daily_return_matrices(
    data_dir: Path,
    ticker_list: list[str],
    timestamp_col: str = "timestamp",
    return_col: str = "return"
):
    """
    Returns:
        dict[date -> pd.DataFrame]
        Each DataFrame:
            index   = minute timestamps
            columns = tickers
            values  = minute returns
    """

    # Load all tickers into memory (acceptable at this scale)
    ticker_data = {}
    for ticker in ticker_list:
        df = pd.read_parquet(data_dir / f"{ticker}.parquet")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df["date"] = df[timestamp_col].dt.date
        ticker_data[ticker] = df

    # Group data by day
    daily_frames = defaultdict(list)

    for ticker, df in ticker_data.items():
        for date, day_df in df.groupby("date"):
            day_df = day_df.set_index(timestamp_col)[[return_col]]
            day_df.columns = [ticker]
            daily_frames[date].append(day_df)

    # Build final matrices
    daily_matrices = {}
    for date, frames in daily_frames.items():
        # Inner join → keeps only tickers with full minute coverage
        day_matrix = pd.concat(frames, axis=1, join="inner")

        # Skip days with fewer than 2 tickers
        if day_matrix.shape[1] < 2:
            continue

        daily_matrices[date] = day_matrix.sort_index()

    return daily_matrices


def build_daily_pairs(daily_matrices):
    """
    Returns:
        dict[date -> list[tuple[str, str]]]
    """

    daily_pairs = {}

    for date, matrix in daily_matrices.items():
        tickers = matrix.columns.tolist()
        pairs = list(combinations(tickers, 2))
        daily_pairs[date] = pairs

    return daily_pairs


def prefilter_pairs_by_xcorr(
    day_matrix,            # pd.DataFrame (minutes x tickers)
    max_lag=30,
    keep_top_k=400,        # tune: 200-800
    min_abs_xcorr=0.05     # safety floor
):
    X = day_matrix.to_numpy(dtype=np.float64)  # shape (T, N)
    tickers = day_matrix.columns.to_list()
    T, N = X.shape

    # z-score each ticker (avoid scale issues)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    X = X / std

    scores = []
    for i, j in combinations(range(N), 2):
        xi, xj = X[:, i], X[:, j]
        best = 0.0
        # leader i -> follower j corresponds to corr(xi[t], xj[t+lag])
        for lag in range(1, max_lag + 1):
            a = xi[:-lag]
            b = xj[lag:]
            c1 = float(np.mean(a * b))
            # also check opposite direction in the score (we’ll decide direction later)
            a2 = xj[:-lag]
            b2 = xi[lag:]
            c2 = float(np.mean(a2 * b2))
            best = max(best, abs(c1), abs(c2))
        if best >= min_abs_xcorr:
            scores.append((best, tickers[i], tickers[j]))

    if not scores:
        return []

    scores.sort(reverse=True, key=lambda x: x[0])
    scores = scores[:keep_top_k]
    
    return [(a, b) for _, a, b in scores]



def _step_a_constant_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> int:
    """
    Step A: Find initial lag l in [0, max_lag] minimizing sum_i |x[i+l] - y[i]|.
    We assume x is potentially the leader (shifted forward) and y the follower.
    """
    n = len(x)
    m = len(y)
    L = min(max_lag, max(0, n - m))
    if m == 0 or n == 0:
        return 0
    if L == 0:
        return 0

    best_l = 0
    best_cost = np.inf

    # Compare y[0:m] with x[l:l+m]
    for l in range(L + 1):
        seg = x[l:l + m]
        if len(seg) != m:
            break
        c = np.sum(np.abs(seg - y))
        if c < best_cost:
            best_cost = c
            best_l = l

    return best_l


def step_a_constant_lag_fast(x, y, max_lag):
    n, m = len(x), len(y)
    L = min(max_lag, max(0, n - m))
    if L <= 0:
        return 0

    # Build a (L+1, m) view: x[l:l+m] for l=0..L
    # Using stride trick to avoid copies
    stride = x.strides[0]
    Xwin = np.lib.stride_tricks.as_strided(
        x, shape=(L + 1, m), strides=(stride, stride)
    )
    costs = np.sum(np.abs(Xwin - y[None, :]), axis=1)
    return int(np.argmin(costs))


@nb.njit
def ocp_banded_dp_stats(x, y, l_init, band):
    n = x.shape[0]
    m = y.shape[0]
    INF = 1e30

    # banded indexing: for each i, allowed j in [j0, j1]
    # center: j ~= i - l_init
    width = 2 * band + 1

    # dp rolling rows
    prev = np.full(width, INF)
    curr = np.full(width, INF)

    # store parent moves for backtracking: 0=diag,1=up,2=left
    # store only for valid band cells
    parent = np.full((n, width), 255, dtype=np.uint8)

    # helper to map (i,j) -> band index k
    # j_center = i - l_init, k = j - (j_center - band)
    for i in range(n):
        j_center = i - l_init
        j_min = j_center - band
        j_max = j_center + band

        # reset current row
        for k in range(width):
            curr[k] = INF

        for j in range(max(0, j_min), min(m - 1, j_max) + 1):
            k = j - j_min  # band index
            c = abs(x[i] - y[j])

            if i == 0 and j == 0:
                curr[k] = c
                parent[i, k] = 0
                continue

            best = INF
            move = 255

            # from up: (i-1, j) => prev row, band index depends on i-1
            if i > 0:
                j_min_prev = (i - 1 - l_init) - band
                k_up = j - j_min_prev
                if 0 <= k_up < width:
                    val = prev[k_up]
                    if val < best:
                        best = val
                        move = 1  # up

            # from left: (i, j-1) => same row, k-1
            if j > 0 and k - 1 >= 0:
                val = curr[k - 1]
                if val < best:
                    best = val
                    move = 2  # left

            # from diag: (i-1, j-1)
            if i > 0 and j > 0:
                j_min_prev = (i - 1 - l_init) - band
                k_diag = (j - 1) - j_min_prev
                if 0 <= k_diag < width:
                    val = prev[k_diag]
                    if val < best:
                        best = val
                        move = 0  # diag

            if move != 255:
                curr[k] = best + c
                parent[i, k] = move

        # roll rows
        prev, curr = curr, prev

    # Backtrack from (n-1, m-1) if it is in band of last row
    i = n - 1
    j_center = i - l_init
    j_min = j_center - band
    k = (m - 1) - j_min
    if k < 0 or k >= width:
        return INF, 0.0, 1e30  # unreachable

    total_cost = prev[k]
    if total_cost >= INF / 2:
        return INF, 0.0, 1e30

    # backtrack lag stats
    count = 0
    sum_d = 0.0
    sum_d2 = 0.0

    j = m - 1
    while True:
        d = i - j
        count += 1
        sum_d += d
        sum_d2 += d * d

        move = parent[i, k]
        if i == 0 and j == 0:
            break

        if move == 0:  # diag
            i -= 1
            j -= 1
        elif move == 1:  # up
            i -= 1
        else:  # left
            j -= 1

        j_center = i - l_init
        j_min = j_center - band
        k = j - j_min
        if k < 0 or k >= width:
            return INF, 0.0, 1e30

    l_hat = sum_d / count
    var = (sum_d2 / count) - l_hat * l_hat
    if var < 0.0:
        var = 0.0
    sigma = np.sqrt(var)
    return total_cost, l_hat, sigma


def _step_b_optimal_path_cost_and_backtrack(
    x: np.ndarray,
    y: np.ndarray,
    l_init: int,
    band: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Step B: Dynamic programming for minimal cost path with moves:
      (i-1,j), (i,j-1), (i-1,j-1)
    Cost per cell: |x[i] - y[j]|

    We constrain computation to a Sakoe-Chiba-like band around the diagonal
    shifted by l_init: i ≈ j + l_init  within +/- band.

    Returns:
      total_cost, path as list of (i, j) indices (0-based), from (0,0) to (n-1,m-1)
    """
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.inf, []

    # DP arrays
    INF = 1e30
    dp = np.full((n, m), INF, dtype=np.float64)
    parent = np.full((n, m, 2), -1, dtype=np.int32)  # store predecessor (pi, pj)

    def in_band(i: int, j: int) -> bool:
        # i close to j + l_init
        return abs(i - (j + l_init)) <= band

    # Initialize
    for i in range(n):
        # For each i, compute allowed j range
        j_center = i - l_init
        j_min = max(0, j_center - band)
        j_max = min(m - 1, j_center + band)
        for j in range(j_min, j_max + 1):
            c = abs(x[i] - y[j])

            if i == 0 and j == 0:
                dp[i, j] = c
                parent[i, j] = (-1, -1)
                continue

            best_prev = INF
            best_pi, best_pj = -1, -1

            # (i-1, j)
            if i > 0 and in_band(i - 1, j) and dp[i - 1, j] < best_prev:
                best_prev = dp[i - 1, j]
                best_pi, best_pj = i - 1, j

            # (i, j-1)
            if j > 0 and in_band(i, j - 1) and dp[i, j - 1] < best_prev:
                best_prev = dp[i, j - 1]
                best_pi, best_pj = i, j - 1

            # (i-1, j-1)
            if i > 0 and j > 0 and in_band(i - 1, j - 1) and dp[i - 1, j - 1] < best_prev:
                best_prev = dp[i - 1, j - 1]
                best_pi, best_pj = i - 1, j - 1

            if best_prev < INF:
                dp[i, j] = best_prev + c
                parent[i, j] = (best_pi, best_pj)

    total_cost = dp[n - 1, m - 1]
    if not np.isfinite(total_cost) or total_cost >= INF / 2:
        return np.inf, []

    # Backtrack
    path = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi, pj = parent[i, j]
        if pi == -1 and pj == -1:
            break
        i, j = int(pi), int(pj)

    path.reverse()
    return float(total_cost), path


def _step_c_lag_stats(path: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    Step C: l_hat = mean(i - j), sigma_l = std(i - j) over the path points.
    """
    if not path:
        return 0.0, np.inf
    diffs = np.array([i - j for (i, j) in path], dtype=np.float64)
    l_hat = float(np.mean(diffs))
    sigma_l = float(np.std(diffs, ddof=0))
    return l_hat, sigma_l


def ocp_pair(
    day_matrix: pd.DataFrame,
    pair: Tuple[str, str],
    max_lag: int = 30,
    band: int = 10,
    min_abs_lag: float = 1.0,
    max_sigma: float = 50.0,
):
    """
    Runs OCP (A,B,C) for one pair on a given day_matrix (minutes x tickers returns).

    Convention:
      We try both orientations:
        x leads y (positive lag) and y leads x
      and keep the one with smaller sigma_l among valid results.

    Valid result filters:
      |l_hat| >= min_abs_lag
      sigma_l <= max_sigma
    """
    a, b = pair
    xa = day_matrix[a].to_numpy(dtype=np.float64)
    xb = day_matrix[b].to_numpy(dtype=np.float64)

    def run_orientation(x_name, y_name, x, y):
        l_init = _step_a_constant_lag(x, y, max_lag=max_lag)
        cost, path = _step_b_optimal_path_cost_and_backtrack(x, y, l_init=l_init, band=band)
        if not np.isfinite(cost) or not path:
            return None
        l_hat, sigma_l = _step_c_lag_stats(path)

        # In this orientation, "x leads y" corresponds to l_hat > 0 typically.
        # We enforce leader/follower using sign of l_hat.
        if abs(l_hat) < min_abs_lag or sigma_l > max_sigma:
            return None

        if l_hat > 0:
            leader, follower = x_name, y_name
        else:
            leader, follower = y_name, x_name  # if negative, it implies y leads x under our convention

        return OCPResult(
            leader=leader,
            follower=follower,
            l_hat=l_hat,
            sigma_l=sigma_l,
            cost=cost,
            path_len=len(path),
        )

    r1 = run_orientation(a, b, xa, xb)
    r2 = run_orientation(b, a, xb, xa)

    if r1 is None and r2 is None:
        return None
    if r1 is None:
        return r2
    if r2 is None:
        return r1

    # Choose the more "stable" (lower sigma_l); tie-breaker: lower cost
    if r1.sigma_l < r2.sigma_l:
        return r1
    if r2.sigma_l < r1.sigma_l:
        return r2
    return r1 if r1.cost <= r2.cost else r2


def ocp_pair_fast(day_matrix, pair, max_lag=30, band=10, min_abs_lag=1.0, max_sigma=50.0):
    a, b = pair
    xa = day_matrix[a].to_numpy(np.float64)
    xb = day_matrix[b].to_numpy(np.float64)

    # Try orientation a->b
    l0 = step_a_constant_lag_fast(xa, xb, max_lag=max_lag)
    cost1, lhat1, sig1 = ocp_banded_dp_stats(xa, xb, l0, band)

    best = None
    if np.isfinite(cost1) and abs(lhat1) >= min_abs_lag and sig1 <= max_sigma:
        leader, follower = (a, b) if lhat1 > 0 else (b, a)
        best = (leader, follower, float(lhat1), float(sig1), float(cost1))

    # Try orientation b->a
    l0 = step_a_constant_lag_fast(xb, xa, max_lag=max_lag)
    cost2, lhat2, sig2 = ocp_banded_dp_stats(xb, xa, l0, band)

    cand = None
    if np.isfinite(cost2) and abs(lhat2) >= min_abs_lag and sig2 <= max_sigma:
        leader, follower = (b, a) if lhat2 > 0 else (a, b)
        cand = (leader, follower, float(lhat2), float(sig2), float(cost2))

    if best is None:
        return cand
    if cand is None:
        return best

    # choose lower sigma, then lower cost
    if cand[3] < best[3]:
        return cand
    if cand[3] > best[3]:
        return best
    return cand if cand[4] < best[4] else best


# -------------------------
# Daily runner: top-K pairs
# -------------------------

def ocp_top_pairs_for_day(
    day_matrix: pd.DataFrame,
    day_pairs: List[Tuple[str, str]],
    keep_top_pairs: int = 400,
    top_k: int = 10,
    max_lag: int = 30,
    band: int = 10,
    min_abs_lag: float = 1.0,
    max_sigma: float = 50.0,
) -> pd.DataFrame:
    """
    Runs OCP on all pairs for a single formation day and returns top_k results
    sorted by sigma_l asc (then cost asc).
    """
    
    pairs = prefilter_pairs_by_xcorr(
        day_matrix=day_matrix,
        max_lag=max_lag,
        keep_top_k=keep_top_pairs,
        min_abs_xcorr=0.05
    )
    
    results: List[OCPResult] = []

    for pair in pairs:
        r = ocp_pair(
            day_matrix=day_matrix,
            pair=pair,
            max_lag=max_lag,
            band=band,
            min_abs_lag=min_abs_lag,
            max_sigma=max_sigma,
        )
        if r is not None:
            results.append(r)

    if not results:
        return pd.DataFrame(columns=["leader", "follower", "l_hat", "sigma_l", "cost", "path_len"])

    df = pd.DataFrame([r.__dict__ for r in results])
    df = df.sort_values(["sigma_l", "cost"], ascending=[True, True]).head(top_k).reset_index(drop=True)
    return df

def ocp_top_pairs_for_day_fast(day_matrix, day_pair, top_k=10, max_lag=30, band=10, keep_top_pairs: Optional[int] = None) -> pd.DataFrame:
    
    if keep_top_pairs is None:
        pairs = day_pair
    else:
        pairs = prefilter_pairs_by_xcorr(day_matrix, max_lag=max_lag, keep_top_k=keep_top_pairs)

    rows = []
    for p in pairs:
        r = ocp_pair_fast(day_matrix, p, max_lag=max_lag, band=band)
        if r is not None:
            rows.append(r)

    if not rows:
        return pd.DataFrame(columns=["leader","follower","l_hat","sigma_l","cost"])

    df = pd.DataFrame(rows, columns=["leader","follower","l_hat","sigma_l","cost"])
    return df.sort_values(["sigma_l","cost"]).head(top_k).reset_index(drop=True)




def ocp_run_all_days_fast(
    daily_matrices: Dict,
    daily_pairs: Dict,
    keep_top_pairs: int = 400,
    top_k: int = 10,
    max_lag: int = 30,
    band: int = 10,
    first_n_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Runs OCP top pairs for every day in daily_matrices and returns one long table.

    Args:
        first_n_days: if provided, run only on the first N days (chronologically).
    """
    rows = []

    days = sorted(daily_matrices.keys())
    if first_n_days is not None:
        if first_n_days <= 0:
            return pd.DataFrame(columns=["date", "leader", "follower", "l_hat", "sigma_l", "cost", "path_len"])
        days = days[:first_n_days]

    for day in tqdm(days):
        day_matrix = daily_matrices[day]
        pairs = daily_pairs.get(day, [])
        if len(pairs) == 0:
            continue

        top_df = ocp_top_pairs_for_day_fast(
            day_matrix=day_matrix,
            top_k=top_k,
            day_pair=pairs,
            keep_top_pairs=keep_top_pairs,
            max_lag=max_lag,
            band=band,
        )
        if top_df.empty:
            continue

        top_df.insert(0, "date", day)
        rows.append(top_df)

    if not rows:
        return pd.DataFrame(columns=["date", "leader", "follower", "l_hat", "sigma_l", "cost", "path_len"])

    return pd.concat(rows, ignore_index=True)
