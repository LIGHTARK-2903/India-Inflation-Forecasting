"""
data_processing.py
-------------------
High-reliability data ingestion and cleaning module for CPI & WPI time-series.
Designed for research-grade forecasting pipelines (RBI/ISI/IIT style).

Author: LIGHTARK (Naman)
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


# -----------------------------
# Utility Helpers
# -----------------------------
def ensure_path(path: str) -> Path:
    """Ensure that a given path exists. Create it if missing."""
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {p}")
    return p


# -----------------------------
# CPI Loading and Cleaning
# -----------------------------
def load_cpi_excel(path: str, sheet_name: str = "Annex-VI") -> pd.DataFrame:
    """
    Load raw CPI data from MOSPI Excel file.

    Parameters
    ----------
    path : str
        File path to the raw CPI Excel sheet.
    sheet_name : str
        Sheet name where CPI is located.

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
        logging.info(f"CPI raw file loaded: {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load CPI file: {path}")
        raise e


def clean_cpi(df: pd.DataFrame) -> pd.Series:
    """
    Clean CPI time-series:
    - Extract Year/Month/value rows
    - Handle merged formatting
    - Build datetime index
    - Return pd.Series(time-indexed)

    Returns
    -------
    pd.Series with datetime index
    """
    try:
        # Keep only valid numeric CPI entries
        df = df.rename(columns={df.columns[0]: "Year", df.columns[1]: "Month", df.columns[2]: "CPI"})
        df = df[df["Year"].apply(lambda x: str(x).isdigit())]

        df["Year"] = df["Year"].astype(int)

        # Parse date
        df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"].astype(str), format="%Y-%b", errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")

        cpi_series = df["CPI"].astype(float).sort_index()

        logging.info(f"CPI cleaned: {len(cpi_series)} monthly observations")
        return cpi_series

    except Exception as e:
        logging.error("CPI cleaning failed.")
        raise e


# -----------------------------
# WPI Loading and Cleaning
# -----------------------------
def load_wpi_excel(path: str) -> pd.DataFrame:
    """
    Load raw WPI data from DPIIT Excel weekly/monthly release.

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_excel(path, header=None)
        logging.info(f"WPI raw file loaded: {path}")
        return df
    except Exception as e:
        logging.error(f"Failed to load WPI file: {path}")
        raise e


def clean_wpi(df: pd.DataFrame, start_row: int = 6) -> pd.Series:
    """
    Clean WPI table:
    - Skip header noise
    - Remove completely empty columns
    - Extract YearRange column + 12 months
    - Convert financial year to monthly dates
    - Produce continuous time-series

    Returns
    -------
    pd.Series
    """
    try:
        df = df.iloc[start_row:].reset_index(drop=True)
        df = df.dropna(how="all", axis=1)

        # First usable row is header e.g. "Year/Month, APR, MAY..."
        df.columns = ["YearRange"] + [f"M{i}" for i in range(1, 13)]

        wpi_records = []

        for _, row in df.iterrows():
            year_text = str(row["YearRange"])
            if "-" not in year_text:
                continue

            start_year = int(year_text.split("-")[0])
            months = row[1:].values.astype(float)

            for i, val in enumerate(months):
                month_num = i + 4  # April = 4
                year = start_year if month_num >= 4 else start_year + 1
                date = pd.to_datetime(f"{year}-{month_num:02d}-01")
                wpi_records.append((date, val))

        wpi_series = (
            pd.DataFrame(wpi_records, columns=["Date", "WPI"])
            .dropna()
            .set_index("Date")
            .sort_index()
        )

        logging.info(f"WPI cleaned: {len(wpi_series)} monthly observations")
        return wpi_series["WPI"]

    except Exception as e:
        logging.error("WPI cleaning failed.")
        raise e


# -----------------------------
# Save Utility
# -----------------------------
def save_series(series: pd.Series, path: str) -> None:
    """Save cleaned time-series safely."""
    try:
        ensure_path(Path(path).parent)
        series.to_csv(path)
        logging.info(f"Saved cleaned series → {path}")
    except Exception as e:
        logging.error(f"Failed to save series: {path}")
        raise e