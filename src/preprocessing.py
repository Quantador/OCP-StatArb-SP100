import os
import tarfile
from pathlib import Path
from typing import List
import polars as pl
import datetime
from tqdm.notebook import tqdm


RAW_FOLDER = Path("data/raw/SP100/bbo/")
EXTRACTED_FOLDER = Path("data/extracted/SP100/bbo/")
PREPROCESSED_FOLDER = Path("data/preprocessed/SP100/bbo/")


def get_tickers_list() -> List[str]:
    """
    Extract all unique tickers from the raw .tar files in RAW_FOLDER.
    Assumes filenames are of the form TICKER.EXCHANGE_YYYYM_bbo.tar
    """
    tickers = set()
    for f in RAW_FOLDER.iterdir():
        if f.is_file() and f.suffix == ".tar":
            ticker = f.name.split("_")[0]
            tickers.add(ticker)
    return list(tickers)


def convert_xltime_to_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convert the float Excel xltime column into a proper UTC timestamp.
    The Excel epoch starts on 1899-12-30.

    Returns a DataFrame with a 'timestamp' column (datetime[us]) and
    without the original 'xltime' column.
    """
    if "xltime" not in df.columns:
        raise ValueError("Missing 'xltime' column in DataFrame.")

    excel_epoch = datetime.datetime(1899, 12, 30)
    microseconds_per_day = 86_400_000_000  # 24h in microseconds

    # Build timestamp expression
    ts_expr = (
        pl.lit(excel_epoch)
        + (pl.col("xltime") * microseconds_per_day)
        .cast(pl.Int64)
        .cast(pl.Duration(time_unit="us"))
    )

    return df.with_columns(
        timestamp=ts_expr
    ).drop("xltime")


def extract_archives_for_ticker(ticker: str) -> Path:
    """
    Extract all .tar archives belonging to a ticker.

    Returns the path to the extracted folder (even if empty).
    """
    EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

    # Select all .tar files beginning with the ticker prefix
    archives = [
        f for f in RAW_FOLDER.iterdir()
        if f.is_file() and f.name.startswith(ticker) and f.suffix == ".tar"
    ]

    # Extract archives one by one
    for archive in archives:
        with tarfile.open(archive) as tar:
            tar.extractall(EXTRACTED_FOLDER)

    # Return folder where extracted parquet files should be
    ticker_folder = EXTRACTED_FOLDER / ticker
    ticker_folder.mkdir(exist_ok=True)

    return ticker_folder


def load_parquet_files(folder: Path) -> List[pl.DataFrame]:
    """
    Read all .parquet files in the given folder
    Cast columns to appropriate types.
    Returns a list of Polars DataFrames.
    """
    dfs = []
    for f in folder.glob("*.parquet"):
        try:
            df = pl.read_parquet(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
        if len(df) > 0:
            # make sure columns have correct types
            df = df.with_columns(
                pl.col("xltime").cast(pl.Float64),
                pl.col("bid-price").cast(pl.Float64),
                pl.col("ask-price").cast(pl.Float64),
                pl.col("bid-volume").cast(pl.Int32),
                pl.col("ask-volume").cast(pl.Int32),
            )
            dfs.append(df)
    return dfs


def clean_and_transform_bbo(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and transform a BBO DataFrame:
    - Convert Excel time to timestamp
    - Convert timezone to America/New_York
    - Filter to regular trading hours (9:30am - 4:00pm)
    - Compute midprice from bid-price and ask-price
    - Resample to 1-minute bars (last midprice)
    - Compute midprice returns
    """
    # Convert Excel time
    df = convert_xltime_to_timestamp(df)

    # Convert timezone
    df = df.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York")
    )

    # Extract date column for grouping
    df = df.with_columns(
        pl.col("timestamp").dt.date().alias("date")
    )

    daily_dfs = []

    # Preprocess day by day
    for day, df_day in df.group_by("date"):
        # Filter RTH for this day
        df_day = df_day.filter(
            (pl.col("timestamp").dt.time() >= pl.time(9, 30)) &
            (pl.col("timestamp").dt.time() <= pl.time(16, 0))
        )

        if len(df_day) == 0:
            continue  # skip empty days

        # Compute midprice
        df_day = df_day.with_columns(
            ((pl.col("bid-price") + pl.col("ask-price")) / 2).alias("mid_price")
        )

        # Resample 1-minute bars (last midprice)
        df_day_1m = (
            df_day.group_by_dynamic("timestamp", every="1m")
                  .agg(pl.col("mid_price").last())
                  .sort("timestamp")
        )

        # Compute returns
        df_day_1m = df_day_1m.with_columns(
            pl.col("mid_price").pct_change().alias("mid_price_return")
        )

        # Drop first row with NaN return
        df_day_1m = df_day_1m.filter(pl.col("mid_price_return").is_not_null())

        daily_dfs.append(df_day_1m)

    # Concatenate all days
    if daily_dfs:
        return pl.concat(daily_dfs, how="vertical").sort("timestamp")
    else:
        return pl.DataFrame(schema=["timestamp", "mid_price", "mid_price_return"])
    

def save_preprocessed_dataframe(ticker: str, df: pl.DataFrame) -> None:
    """
    Save the preprocessed DataFrame to PREPROCESSED_FOLDER as a parquet file.
    """
    PREPROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = PREPROCESSED_FOLDER / f"{ticker}.parquet"
    df.write_parquet(output_path)


def preprocess_ticker(ticker: str) -> pl.DataFrame:
    """
    Full pipeline for a single ticker:
    1. extract archives
    2. load parquet files
    3. merge data
    4. clean and transform

    Returns a Polars DataFrame with timestamp + mid_price.
    """
    # Extract archives
    ticker_folder = extract_archives_for_ticker(ticker)

    # Load parquet files
    dfs = load_parquet_files(ticker_folder)
    if len(dfs) == 0:
        raise RuntimeError(f"No parquet files found for {ticker}")

    # Merge all daily files
    merged = pl.concat(dfs, how="vertical")

    # Clean and compute midprice
    preprocessed_df = clean_and_transform_bbo(merged)

    # Save preprocessed DataFrame
    save_preprocessed_dataframe(ticker, preprocessed_df)

    return preprocessed_df


def preprocess_all_tickers() -> None:
    """
    Preprocess all tickers found in RAW_FOLDER.
    
    This function checks which tickers have already been preprocessed,
    processes only the missing ones, and keeps track of successfully
    processed tickers. Any errors during preprocessing are caught
    without stopping the entire pipeline.
    """
    # Get the list of all tickers to process
    tickers = get_tickers_list()

    # Get the set of already preprocessed parquet files
    existing_files = {f.name for f in PREPROCESSED_FOLDER.glob("*.parquet")}

    # Determine which tickers still need to be processed
    tickers_to_process = [t for t in tickers if f"{t}.parquet" not in existing_files]

    # Keep track of successfully processed tickers
    successfully_processed_tickers = list(set(tickers) - set(tickers_to_process))

    # Loop through tickers to process with a progress bar
    for ticker in tqdm(tickers_to_process, desc="Preprocessing tickers"):
        try:
            preprocess_ticker(ticker)
            successfully_processed_tickers.append(ticker)
        except Exception as e:
            # Print the error but continue with the next ticker
            print(f"Error preprocessing {ticker}: {e}")

    print(
        f"{len(successfully_processed_tickers)} tickers preprocessed successfully "
        f"over {len(tickers)} attempted."
    )