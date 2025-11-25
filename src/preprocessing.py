import os
import tarfile
from pathlib import Path
from typing import List
import polars as pl
import datetime
from tqdm.notebook import tqdm
import datetime as dt
import logging


# =============================================================================
# Logging configuration
# =============================================================================

LOG_FOLDER = Path("logs")
LOG_FOLDER.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FOLDER / "preprocessing.log", mode="a"),  # log in file
        logging.StreamHandler()  # also log in console
    ]
)

logger = logging.getLogger(__name__)


# =============================================================================
# Global constants
# =============================================================================

RAW_FOLDER = Path("data/raw/SP100/bbo/")  # original tar archives
EXTRACTED_FOLDER = Path("data/extracted/SP100/bbo/")  # temporary extraction folder
PREPROCESSED_FOLDER = Path("data/preprocessed/SP100/bbo/")  # final output parquet

EXPECTED_MIDPRICES_PER_DAY = 390  # number of minutes in regular trading hours (6.5h * 60)
EXPECTED_RETURNS_PER_DAY = EXPECTED_MIDPRICES_PER_DAY - 1  # returns are one less than prices
TIMEZONE = "America/New_York"  # timezone for US markets


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

    excel_epoch = datetime.datetime(1899, 12, 30)
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
