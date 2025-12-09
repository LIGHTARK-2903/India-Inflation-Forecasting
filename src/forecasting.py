"""
forecasting.py
----------------
Forecasting utilities: produce point forecasts, confidence intervals, YoY conversions,
plots, and simple scenario analysis. Research-grade, designed to integrate with modeling.py
and data_processing.py.

Author: LIGHTARK (Naman)
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Forecast utilities
# -----------------------------

def forecast_sarima(res, steps: int = 12, alpha: float = 0.05) -> pd.DataFrame:
    """Produce point forecast and confidence intervals from a fitted SARIMAX result.

    Parameters
    ----------
    res : SARIMAXResults
        Fitted SARIMAX results object (statsmodels)
    steps : int
        Forecast horizon in months
    alpha : float
        Significance level for the confidence interval (0.05 -> 95% CI)

    Returns
    -------
    pd.DataFrame with columns ['CPI_forecast','CPI_lower','CPI_upper'] indexed by forecast dates
    """
    try:
        fc = res.get_forecast(steps=steps)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=alpha)

        df = pd.DataFrame({
            'CPI_forecast': mean,
            'CPI_lower': ci.iloc[:, 0],
            'CPI_upper': ci.iloc[:, 1]
        })
        df.index.name = 'Date'
        logging.info(f"Produced {steps}-step forecast")
        return df
    except Exception as e:
        logging.error("Forecasting failed")
        raise e


# -----------------------------
# YoY conversion
# -----------------------------

def compute_yoy_from_index(series: pd.Series) -> pd.Series:
    """Compute Year-on-Year percent change from index-level series.

    series must be time-indexed monthly series.
    """
    return series.pct_change(12) * 100


# -----------------------------
# Plot forecast
# -----------------------------

def plot_forecast(historical: pd.Series, forecast_df: pd.DataFrame, savepath: Path = None,
                  title: str = 'Forecast', ylabel: str = 'Index') -> None:
    """Plot historical series with forecast and confidence intervals.

    forecast_df expected columns: ['CPI_forecast','CPI_lower','CPI_upper']
    """
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(historical.index, historical.values, label='Observed', linewidth=1.6)
        plt.plot(forecast_df.index, forecast_df['CPI_forecast'], label='Forecast', linewidth=2, color='C1')
        plt.fill_between(forecast_df.index,
                         forecast_df['CPI_lower'],
                         forecast_df['CPI_upper'],
                         color='C1', alpha=0.18)
        plt.axvline(historical.index.max(), linestyle='--', color='k', alpha=0.6)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        if savepath is not None:
            ensure_dir(savepath)
            plt.savefig(savepath, dpi=160)
            logging.info(f"Saved forecast plot to {savepath}")
        plt.show()
    except Exception as e:
        logging.error("Failed to plot forecast")
        raise e


# -----------------------------
# Save forecast outputs
# -----------------------------

def save_forecast_table(forecast_df: pd.DataFrame, path: str) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(p)
        logging.info(f"Saved forecast table to {path}")
    except Exception as e:
        logging.error("Failed to save forecast table")
        raise e


# -----------------------------
# Simple WPI shock scenario (naive transfer via lagged regression)
# -----------------------------

def wpi_shock_scenario(cpi_series: pd.Series, wpi_yoy: pd.Series, forecast_index_df: pd.DataFrame,
                       lags: int = 10, shock_pp: float = 3.0, shock_h: int = 6) -> pd.DataFrame:
    """Construct a simple scenario where WPI YoY is increased by shock_pp for shock_h months
    and estimate naive CPI YoY response via OLS on lagged WPI.

    This is an illustrative, not structural, scenario used for policy sensitivity checks.
    Returns a DataFrame with scenario CPI YoY for the shock horizon.
    """
    try:
        # prepare lagged WPI matrix
        wpi_yoy = wpi_yoy.dropna()
        lagged = pd.concat([wpi_yoy.shift(l) for l in range(1, lags + 1)], axis=1)
        lagged.columns = [f'wpi_lag_{l}' for l in range(1, lags + 1)]

        df = pd.concat([cpi_series.pct_change(12) * 100, lagged], axis=1).dropna()
        df.rename(columns={df.columns[0]: 'cpi_yoy'}, inplace=True)

        import statsmodels.api as sm
        X = sm.add_constant(df[[f'wpi_lag_{l}' for l in range(1, lags + 1)]])
        y = df['cpi_yoy']
        ols = sm.OLS(y, X).fit()

        logging.info("Estimated lagged WPI->CPI OLS for scenario analysis")

        # Build scenario WPI series (starting at forecast start)
        start = forecast_index_df.index[0]
        scenario_index = pd.date_range(start=start, periods=shock_h, freq='MS')
        last_wpi = wpi_yoy.iloc[-1]
        wpi_scenario = pd.Series(last_wpi, index=scenario_index) + shock_pp

        # construct lagged features for scenario using last observed lags (approx)
        # For simplicity use last observed lag vector repeated shifting
        last_lags = wpi_yoy.iloc[-lags:].values[::-1]  # most recent lags
        scenario_rows = []
        for t in range(shock_h):
            # build lag vector: for illustrative purpose we use last_lags shifted by t
            vec = np.roll(last_lags, t)[:lags]
            row = [1.0] + list(vec)
            scenario_rows.append(row)

        X_scn = pd.DataFrame(scenario_rows, columns=X.columns, index=scenario_index)
        cpi_yoy_scn = ols.predict(X_scn)

        result = pd.DataFrame({'CPI_yoy_scenario': cpi_yoy_scn}, index=scenario_index)
        logging.info("Constructed naive WPI shock CPI scenario")
        return result

    except Exception as e:
        logging.error("WPI shock scenario generation failed")
        raise e


# -----------------------------
# Quick summary stats for forecast YoY
# -----------------------------

def forecast_yoy_summary(forecast_index_df: pd.DataFrame, historical_series: pd.Series) -> Dict[str, float]:
    """Given forecast index-level DataFrame and historical series, compute YoY% and return min/mean/max.
    forecast_index_df must have 'CPI_forecast' column.
    """
    combined = pd.concat([historical_series, forecast_index_df['CPI_forecast']]).rename('CPI')
    combined = combined.sort_index()
    yoy = combined.pct_change(12) * 100
    forecast_yoy = yoy.loc[forecast_index_df.index]
    return {'min': float(forecast_yoy.min()), 'mean': float(forecast_yoy.mean()), 'max': float(forecast_yoy.max())}