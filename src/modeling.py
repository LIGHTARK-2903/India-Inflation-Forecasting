"""
modeling.py
-----------
Research-grade modeling utilities for SARIMA/ARIMA forecasting, diagnostics,
and model selection. Designed to integrate with data_processing.py

Author: LIGHTARK (Naman)
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------------
# Helper utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# ACF / PACF plotting
# -----------------------------

def plot_acf_pacf(series: pd.Series, lags: int = 36, savepath: Optional[Path] = None) -> None:
    """Plot ACF and PACF side-by-side and optionally save figure.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series (e.g., CPI YoY or index level)
    lags : int
        Number of lags to display
    savepath : Optional[Path]
        Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=axes[0], zero=False)
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], zero=False, method='ywm')
    axes[0].set_title('ACF')
    axes[1].set_title('PACF')
    plt.tight_layout()
    if savepath is not None:
        ensure_dir(savepath)
        fig.savefig(savepath, dpi=150)
        logging.info(f"Saved ACF/PACF plot to {savepath}")
    plt.show()


# -----------------------------
# Fit SARIMA
# -----------------------------

def fit_sarima(series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int],
               enforce_stationarity: bool = False, enforce_invertibility: bool = False, disp: bool = False) -> SARIMAX:
    """Fit SARIMA model and return the results object.

    Parameters
    ----------
    series : pd.Series
    order : tuple
    seasonal_order : tuple
    """
    logging.info(f"Fitting SARIMA order={order} seasonal_order={seasonal_order}")
    model = SARIMAX(series,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=enforce_stationarity,
                    enforce_invertibility=enforce_invertibility)
    res = model.fit(disp=disp)
    logging.info(f"Fitted SARIMA; AIC={res.aic:.3f} BIC={res.bic:.3f}")
    return res


# -----------------------------
# Auto ARIMA wrapper
# -----------------------------

def fit_auto_arima(series: pd.Series,
                   m: int = 12,
                   start_p: int = 0, max_p: int = 3,
                   start_q: int = 0, max_q: int = 3,
                   start_P: int = 0, max_P: int = 2,
                   start_Q: int = 0, max_Q: int = 2,
                   stepwise: bool = True,
                   information_criterion: str = 'aic',
                   trace: bool = False) -> Any:
    """Run pmdarima.auto_arima and return the fitted AutoARIMA object.

    The returned object has .summary() and .order_/.seasonal_order_ attributes.
    """
    logging.info("Running auto_arima for model selection")
    auto = pm.auto_arima(
        series,
        start_p=start_p, max_p=max_p,
        start_q=start_q, max_q=max_q,
        seasonal=True, m=m,
        start_P=start_P, max_P=max_P,
        start_Q=start_Q, max_Q=max_Q,
        stepwise=stepwise,
        information_criterion=information_criterion,
        trace=trace,
        error_action='ignore',
        suppress_warnings=True
    )
    logging.info(f"Auto ARIMA selected: order={auto.order} seasonal_order={auto.seasonal_order}")
    return auto


# -----------------------------
# Residual diagnostics
# -----------------------------

def residual_diagnostics(res, lags: int = 24, savepath: Optional[Path] = None) -> Dict[str, Any]:
    """Generate diagnostic plots and Ljung-Box test results.

    Returns a dictionary of diagnostic metrics.
    """
    logging.info("Running residual diagnostics")
    res.plot_diagnostics(figsize=(12, 10))
    if savepath is not None:
        ensure_dir(savepath)
        plt.savefig(savepath, dpi=150)
        logging.info(f"Saved residual diagnostics to {savepath}")
    plt.show()

    lb = acorr_ljungbox(res.resid.dropna(), lags=[6, 12, 18, 24], return_df=True)
    diagnostics = {
        'aic': getattr(res, 'aic', np.nan),
        'bic': getattr(res, 'bic', np.nan),
        'ljungbox_pvalues': lb['lb_pvalue'].to_dict(),
        'resid_mean': float(res.resid.dropna().mean()),
        'resid_std': float(res.resid.dropna().std())
    }
    logging.info(f"Diagnostics: AIC={diagnostics['aic']:.3f} LB_pvals={diagnostics['ljungbox_pvalues']}")
    return diagnostics


# -----------------------------
# Forecast evaluation
# -----------------------------

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute RMSE, MAE, MAPE between true and predicted (aligned by index).

    Returns
    -------
    dict
    """
    # align
    y_true, y_pred = y_true.align(y_pred, join='inner')
    mse = ((y_true - y_pred) ** 2).mean()
    rmse = float(np.sqrt(mse))
    mae = float(np.abs(y_true - y_pred).mean())
    mape = float((np.abs((y_true - y_pred) / y_true)).mean() * 100)
    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
    logging.info(f"Eval metrics: {metrics}")
    return metrics


# -----------------------------
# Small grid search (safe, limited)
# -----------------------------

def seasonal_grid_search(series: pd.Series, p_range=range(0,3), q_range=range(0,3),
                         P_range=range(0,2), Q_range=range(0,2), d=1, D=1, m=12) -> Dict[str, Any]:
    """Run a constrained grid search over SARIMA orders and return the best by AIC.

    NOTE: This is intentionally limited to avoid long runtimes.
    """
    import itertools
    best = {'aic': np.inf}
    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    try:
                        mod = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m),
                                      enforce_stationarity=False, enforce_invertibility=False)
                        res = mod.fit(disp=False)
                        if res.aic < best['aic']:
                            best = {'order': (p, d, q), 'seasonal': (P, D, Q, m), 'aic': res.aic}
                    except Exception as e:
                        continue
    logging.info(f"Grid search best: {best}")
    return best


# -----------------------------
# Model persistence
# -----------------------------

def save_model(obj: Any, filepath: str) -> None:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Saved model to {filepath}")


def load_model(filepath: str) -> Any:
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logging.info(f"Loaded model from {filepath}")
    return obj