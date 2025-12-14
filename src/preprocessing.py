import os
import shutil
import tarfile
import itertools
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import datetime as dt

import polars as pl
import pandas as pd
import pandas_market_calendars as mcal
from tqdm.notebook import tqdm




# =============================================================================
# Logging configuration
# =============================================================================

LOG_FOLDER = Path("logs")
LOG_FOLDER.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # save everything

# Remove existing handlers to avoid duplicates in notebook
if logger.hasHandlers():
    logger.handlers.clear()

# File handler: save everything
file_handler = logging.FileHandler(LOG_FOLDER / "preprocessing.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(file_handler)

# Stream handler: only show INFO and above
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(stream_handler)


# =============================================================================
# Global constants
# =============================================================================

RAW_FOLDER = Path("data/raw/SP100/bbo/")  # original tar archives
EXTRACTED_FOLDER = Path("data/extracted/SP100/bbo/")  # temporary extraction folder
PREPROCESSED_FOLDER = Path("data/preprocessed/SP100/bbo/")  # final output parquet
SELECTED_FOLDER = Path("data/selected/SP100/bbo/")  # selected data for analysis

EXPECTED_MIDPRICES_PER_DAY = 390  # number of minutes in regular trading hours (6.5h * 60)
EXPECTED_RETURNS_PER_DAY = EXPECTED_MIDPRICES_PER_DAY - 1  # returns are one less than prices
TIMEZONE = "America/New_York"  # timezone for US markets

START_DATE = dt.date(2015, 1, 1)
END_DATE = dt.date(2017, 3, 31)
EXPECTED_TRADING_DAYS = 565  # number of NASDAQ trading days between START_DATE and END_DATE without 27-Nov-2015


# =============================================================================
# Utility functions
# =============================================================================

def get_tickers_list() -> List[str]:
    """
    Scan RAW_FOLDER to identify all unique tickers based on tar file names.

    Returns:
        List[str]: List of tickers found in RAW_FOLDER.
    """
    logger.info("Scanning RAW_FOLDER for tickers...")
    tickers = set()
    for f in RAW_FOLDER.iterdir():
        if f.is_file() and f.suffix == ".tar":
            ticker = f.name.split("_")[0]  # assume format TICKER_*.tar
            tickers.add(ticker)

    logger.info(f"Found {len(tickers)} tickers.")
    return list(tickers)


def convert_xltime_to_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert Excel-style float 'xltime' into proper timestamp.

    Excel xltime counts days since 1899-12-30. This function converts
    it into a UTC datetime column 'timestamp' and removes 'xltime'.

    Args:
        df (pl.DataFrame): DataFrame with 'xltime' column.

    Returns:
        pl.DataFrame: DataFrame with 'timestamp' column.
    """
    logger.debug("Converting Excel xltime to timestamps...")

    if "xltime" not in df.columns:
        msg = "Missing 'xltime' column."
        logger.error(msg)
        raise ValueError(msg)

    excel_epoch = dt.datetime(1899, 12, 30)
    micro_per_day = 86_400_000_000  # microseconds per day

    ts_expr = (
        pl.lit(excel_epoch)
        + (pl.col("xltime") * micro_per_day)
        .cast(pl.Int64)
        .cast(pl.Duration(time_unit="us"))
    )

    return df.with_columns(timestamp=ts_expr).drop("xltime")


def extract_archives_for_ticker(ticker: str) -> Path:
    """
    Extract all tar archives for a given ticker into EXTRACTED_FOLDER.

    Args:
        ticker (str): Ticker symbol to extract.

    Returns:
        Path: Folder where extracted parquet files are stored.
    """
    logger.info(f"[{ticker}] Extracting archives...")
    EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

    archives = [
        f for f in RAW_FOLDER.iterdir()
        if f.is_file() and f.name.startswith(ticker) and f.suffix == ".tar"
    ]

    logger.debug(f"[{ticker}] Found {len(archives)} archives.")

    for archive in archives:
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(EXTRACTED_FOLDER)
            logger.debug(f"[{ticker}] Extracted {archive.name}")
        except Exception as e:
            logger.error(f"[{ticker}] Error extracting {archive.name}: {e}")

    ticker_folder = EXTRACTED_FOLDER / ticker
    ticker_folder.mkdir(exist_ok=True)
    return ticker_folder


def load_parquet_files(folder: Path) -> List[pl.DataFrame]:
    """
    Load all parquet files from a folder and cast columns to correct types.

    Args:
        folder (Path): Folder containing parquet files.

    Returns:
        List[pl.DataFrame]: List of typed DataFrames.
    """
    logger.info(f"Loading parquet files in {folder}...")
    dfs = []
    for f in folder.glob("*.parquet"):
        try:
            df = pl.read_parquet(f)
            logger.debug(f"Loaded {f.name} ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Error reading {f}: {e}")
            continue

        if len(df) > 0:
            df = df.with_columns(
                pl.col("xltime").cast(pl.Float64),
                pl.col("bid-price").cast(pl.Float64),
                pl.col("ask-price").cast(pl.Float64),
                pl.col("bid-volume").cast(pl.Int32),
                pl.col("ask-volume").cast(pl.Int32),
            )
            dfs.append(df)

    logger.info(f"Loaded {len(dfs)} parquet files.")
    return dfs


def fill_missing_minutes(df_day_1m: pl.DataFrame) -> pl.DataFrame:
    """
    Fill missing minutes in regular trading hours with forward fill and backward fill.

    Args:
        df_day_1m (pl.DataFrame): 1-minute resampled data for one day.

    Returns:
        pl.DataFrame: DataFrame with missing minutes filled.
    """
    if df_day_1m.height == 0:
        return df_day_1m

    day = df_day_1m["timestamp"][0].date()
    logger.debug(f"Filling missing minutes for day {day}...")

    expected_ts = pl.datetime_range(
        start=dt.datetime(day.year, day.month, day.day, 9, 31),
        end=dt.datetime(day.year, day.month, day.day, 16, 0),
        interval="1m",
        time_zone=TIMEZONE,
        eager=True
    ).alias("timestamp")

    expected_df = pl.DataFrame({"timestamp": expected_ts})

    merged = expected_df.join(df_day_1m, on="timestamp", how="left")
    merged = merged.with_columns(
        pl.col("mid_price").forward_fill().backward_fill()
    )

    return merged


def clean_and_transform_bbo(df: pl.DataFrame) -> pl.DataFrame:
    """
    Full preprocessing pipeline for a single ticker's raw BBO data.

    Steps:
    1. Convert Excel xltime to timestamp
    2. Convert timezone to TIMEZONE
    3. Add date column
    4. Process data day by day
        - Filter RTH (9:30-16:00)
        - Compute midprice
        - Resample to 1-minute bars
        - Fill missing minutes with forward fill and backward fill
        - Compute returns
    5. Merge daily data

    Args:
        df (pl.DataFrame): Raw BBO DataFrame.

    Returns:
        pl.DataFrame: Preprocessed DataFrame with 'timestamp' and 'mid_price_return'.
    """
    logger.info("Cleaning and transforming raw BBO dataframe...")

    df = convert_xltime_to_timestamp(df)
    df = df.with_columns(pl.col("timestamp").dt.convert_time_zone(TIMEZONE))
    df = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

    daily_dfs = []

    for day, df_day in df.group_by("date"):
        logger.debug(f"Processing day {day}...")

        # Filter regular trading hours
        df_day = df_day.filter(
            (pl.col("timestamp").dt.time() > pl.time(9, 30)) &
            (pl.col("timestamp").dt.time() < pl.time(16, 0))
        )

        if df_day.height == 0:
            logger.debug(f"No RTH data for day {day}, skipping.")
            continue

        # Compute midprice
        df_day = df_day.with_columns(
            ((pl.col("bid-price") + pl.col("ask-price")) / 2).alias("mid_price")
        )

        # Resample to 1-minute bars
        df_day_1m = (
            df_day.group_by_dynamic("timestamp", every="1m")
                  .agg(pl.col("mid_price").last())
                  .sort("timestamp")
        )

        df_day_1m = df_day_1m.with_columns(
            (pl.col("timestamp") + pl.duration(minutes=1)).alias("timestamp")
        )

        if df_day_1m.height != EXPECTED_MIDPRICES_PER_DAY:
            logger.warning(
                f"[DAY {day}] {df_day_1m.height} midprices found, expected {EXPECTED_MIDPRICES_PER_DAY}."
            )
            df_day_1m = fill_missing_minutes(df_day_1m)

        # Compute returns
        df_day_1m = df_day_1m.with_columns(
            pl.col("mid_price").pct_change().alias("mid_price_return")
        )
        df_day_1m = df_day_1m.filter(pl.col("mid_price_return").is_not_null())

        if df_day_1m.height != EXPECTED_RETURNS_PER_DAY:
            logger.warning(
                f"[DAY {day}] {df_day_1m.height} returns found, expected {EXPECTED_RETURNS_PER_DAY}."
            )

        daily_dfs.append(df_day_1m)

    if daily_dfs:
        merged = (
            pl.concat(daily_dfs, how="vertical")
            .sort("timestamp")
            .select(["timestamp", "mid_price_return"])
        )
        logger.info("Daily data merged successfully.")
        return merged

    logger.warning("No daily data produced for this ticker.")
    return pl.DataFrame(schema=["timestamp", "mid_price_return"])


def save_preprocessed_dataframe(ticker: str, df: pl.DataFrame) -> None:
    """
    Save preprocessed ticker data to parquet.

    Args:
        ticker (str): Ticker symbol.
        df (pl.DataFrame): Preprocessed DataFrame.
    """
    PREPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = PREPROCESSED_FOLDER / f"{ticker}.parquet"
    df.write_parquet(output_path)
    logger.info(f"[{ticker}] Saved preprocessed dataframe to {output_path}.")


def preprocess_ticker(ticker: str) -> pl.DataFrame:
    """
    Full preprocessing pipeline for a single ticker.

    Args:
        ticker (str): Ticker symbol.

    Returns:
        pl.DataFrame: Preprocessed DataFrame.
    """
    logger.info(f"=== Preprocessing ticker {ticker} ===")

    ticker_folder = extract_archives_for_ticker(ticker)
    dfs = load_parquet_files(ticker_folder)
    if len(dfs) == 0:
        msg = f"No parquet files found for ticker {ticker}"
        logger.error(msg)
        raise RuntimeError(msg)

    merged = pl.concat(dfs, how="vertical")
    preprocessed = clean_and_transform_bbo(merged)
    save_preprocessed_dataframe(ticker, preprocessed)

    logger.info(f"=== Finished ticker {ticker} ===")
    return preprocessed


def preprocess_all_tickers() -> None:
    """
    Preprocess all tickers found in RAW_FOLDER.

    Skips tickers that already have a preprocessed parquet file in PREPROCESSED_FOLDER.
    Logs progress using tqdm.
    """
    logger.info("Starting preprocessing of all tickers...")

    tickers = get_tickers_list()
    existing = {f.name for f in PREPROCESSED_FOLDER.glob("*.parquet")}
    to_process = [t for t in tickers if f"{t}.parquet" not in existing]

    logger.info(f"{len(to_process)} tickers need preprocessing.")

    successfully = list(set(tickers) - set(to_process))

    for ticker in tqdm(to_process, desc="Preprocessing tickers"):
        try:
            preprocess_ticker(ticker)
            successfully.append(ticker)
        except Exception as e:
            logger.error(f"[{ticker}] Error during preprocessing: {e}")

    logger.info(
        f"{len(successfully)} tickers processed successfully over {len(tickers)} total."
    )


def select_complete_tickers(
    exception_dates: List[pd.Timestamp] = None,
    delete_exception_days: bool = False
) -> None:
    """
    Select and copy parquet files from PREPROCESSED_FOLDER to SELECTED_FOLDER 
    that contain all NASDAQ trading days between START_DATE and END_DATE,
    with the option to delete specific exception dates from the files.

    Args:
        exception_dates (List[pd.Timestamp], optional): List of dates that are allowed
            to be missing from the data. Defaults to None.
        delete_exception_days (bool, optional): If True, remove the exception dates
            from tickers that contain them. Defaults to False.
    """
    logger.info("Starting selection of complete tickers...")

    # Initialize exception dates
    if exception_dates is None:
        exception_dates = []
    exception_dates_set = set([d.date() if isinstance(d, pd.Timestamp) else d for d in exception_dates])

    # Generate NASDAQ trading calendar
    nasdaq = mcal.get_calendar('NASDAQ')
    schedule = nasdaq.schedule(start_date=pd.Timestamp(START_DATE),
                               end_date=pd.Timestamp(END_DATE))
    trading_days = set([d.date() for d in schedule.index])
    logger.debug(f"Total trading days: {len(trading_days)}, exception dates: {exception_dates_set}")

    # Adjust trading days by removing exception dates
    trading_days_to_keep = trading_days - exception_dates_set
    logger.debug(f"Trading days after removing exceptions: {len(trading_days_to_keep)}")

    # Ensure SELECTED_FOLDER exists and is empty
    SELECTED_FOLDER.mkdir(parents=True, exist_ok=True)
    for f in SELECTED_FOLDER.glob("*.parquet"):
        f.unlink()

    # Loop through all preprocessed parquet files
    files = list(PREPROCESSED_FOLDER.glob("*.parquet"))
    copied_files = []

    for f in files:
        df = pl.read_parquet(f)
        df_dates = set(df.select(pl.col("timestamp").dt.date().unique()).to_series().to_list())

        # Delete exception dates if requested
        if delete_exception_days and exception_dates_set & df_dates:
            df = df.filter(pl.col("timestamp").dt.date().is_in(list(trading_days_to_keep)))
            logger.debug(f"[{f.name}] Deleted exception dates: {exception_dates_set & df_dates}")

        # Recalculate present dates after optional deletion
        df_dates_cleaned = set(df.select(pl.col("timestamp").dt.date().unique()).to_series().to_list())

        # Compute missing trading days ignoring exceptions
        missing_dates = trading_days - df_dates_cleaned - exception_dates_set

        if missing_dates:
            logger.warning(f"[{f.name}] Missing {len(missing_dates)} trading days: {sorted(missing_dates)}")
        else:
            # Save cleaned file if we deleted exception rows, otherwise copy original
            output_path = SELECTED_FOLDER / f.name
            if delete_exception_days and exception_dates_set & df_dates:
                df.write_parquet(output_path)
            else:
                shutil.copy(f, output_path)
            copied_files.append(f.name)
            logger.debug(f"[{f.name}] Copied to selected folder")

    logger.info(f"Selection completed: {len(copied_files)}/{len(files)} files copied to {SELECTED_FOLDER}")


def count_tickers_with_expected_rows(
    expected_rows_per_day: int = EXPECTED_TRADING_DAYS, 
    expected_days: int = EXPECTED_RETURNS_PER_DAY
) -> None:
    """
    Count how many tickers in the SELECTED_FOLDER have at least the expected number of rows.

    Args:
        expected_rows_per_day (int): Number of rows expected per trading day (default 389).
        expected_days (int): Number of trading days expected (default 565).
    """
    logger.info("Counting rows for each selected ticker...")

    expected_num_rows = expected_rows_per_day * expected_days
    selected_files = list(SELECTED_FOLDER.glob("*.parquet"))
    num_tickers_with_enough_rows = 0

    for f in selected_files:
        try:
            df = pl.read_parquet(f)
        except Exception as e:
            logger.error(f"Error reading {f.name}: {e}")
            continue

        if df.height >= expected_num_rows:
            num_tickers_with_enough_rows += 1
            logger.debug(f"{f.name}: {df.height} rows (meets expected {expected_num_rows})")
        else:
            logger.warning(f"{f.name}: {df.height} rows (below expected {expected_num_rows})")

    logger.info(f"Number of tickers with the expected number of rows: {num_tickers_with_enough_rows} "
                f"out of {len(selected_files)}")


def get_selected_tickers() -> List[str]:
    """
    List all tickers that have a selected parquet file in SELECTED_FOLDER.

    Returns:
        List[str]: Sorted list of ticker symbols (without .parquet).
    """
    tickers = []
    for f in SELECTED_FOLDER.glob("*.parquet"):
        # e.g. "AAPL.parquet" -> "AAPL"
        tickers.append(f.stem)
    tickers = sorted(tickers)
    logger.info(f"Found {len(tickers)} selected tickers.")
    return tickers


def load_selected_ticker_frames() -> Dict[str, pl.DataFrame]:
    """
    Load all selected tickers into memory as a dict: ticker -> DataFrame(timestamp, mid_price_return).

    Returns:
        Dict[str, pl.DataFrame]: One DataFrame per ticker, aligned on its own timestamp grid.
    """
    frames: Dict[str, pl.DataFrame] = {}

    for f in SELECTED_FOLDER.glob("*.parquet"):
        ticker = f.stem
        try:
            df = pl.read_parquet(f)
        except Exception as e:
            logger.error(f"[{ticker}] Error reading selected parquet: {e}")
            continue

        # Expect at least columns: timestamp, mid_price_return
        if not {"timestamp", "mid_price_return"}.issubset(df.columns):
            logger.warning(f"[{ticker}] Missing expected columns, skipping.")
            continue

        # Enforce dtypes and sort
        df = (
            df
            .with_columns(pl.col("timestamp").cast(pl.Datetime(time_zone=TIMEZONE)))
            .sort("timestamp")
            .select(["timestamp", "mid_price_return"])
        )

        frames[ticker] = df
        logger.debug(f"[{ticker}] Loaded {df.height} rows from selected data.")

    logger.info(f"Loaded {len(frames)} selected ticker frames into memory.")
    return frames

def build_returns_panel(frames: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Build a wide returns panel from per-ticker DataFrames.

    The result has:
        - one row per timestamp
        - one column per ticker containing its mid_price_return

    Assumes that all frames cover (almost) the same timestamp grid; any
    residual misalignment will show up as nulls, which you can handle
    downstream (e.g., drop rows with any nulls).

    Args:
        frames (Dict[str, pl.DataFrame]): Mapping ticker -> DataFrame(timestamp, mid_price_return).

    Returns:
        pl.DataFrame: Wide panel with columns ['timestamp', <ticker1>, <ticker2>, ...].
    """
    if not frames:
        logger.warning("No frames provided to build_returns_panel; returning empty DataFrame.")
        return pl.DataFrame()

    # Stack all tickers vertically, then pivot to wide format
    stacked = []
    for ticker, df in frames.items():
        stacked.append(
            df.with_columns(
                pl.lit(ticker).alias("ticker")
            )
        )

    all_data = pl.concat(stacked, how="vertical")

    panel = (
        all_data
        .pivot(
            values="mid_price_return",
            index="timestamp",
            columns="ticker"
        )
        .sort("timestamp")
    )

    logger.info(
        f"Returns panel built: {panel.height} rows x {len(panel.columns) - 1} tickers "
        f"(+1 timestamp column)."
    )

    # Optional: sanity checks on nulls
    null_counts = panel.null_count()
    total_nulls = int(sum(null_counts.select(pl.all().sum()).row(0)))
    if total_nulls > 0:
        logger.warning(
            f"Returns panel contains {total_nulls} null entries. "
            "Consider dropping or imputing rows with nulls before OCP."
        )

    return panel


def generate_ticker_pairs(tickers: List[str]) -> List[Tuple[str, str]]:
    """
    Generate all unordered ticker pairs from a list of tickers.

    Args:
        tickers (List[str]): List of ticker symbols.

    Returns:
        List[Tuple[str, str]]: List of (ticker_i, ticker_j) with i < j.
    """
    pairs = list(itertools.combinations(sorted(tickers), 2))
    logger.info(f"Generated {len(pairs)} ticker pairs from {len(tickers)} tickers.")
    return pairs
