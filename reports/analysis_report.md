# RBI-Style Analytical Report: India Inflation Forecasting System (CPI + WPI)

## Executive Summary

This report presents a complete macroeconomic inflation analysis using CPI and WPI time-series data, applying RBI-grade empirical techniques. Forecasts are generated using the SARIMA(0,1,1)(1,0,1,12) model, identified through rigorous AIC comparison, residual diagnostics, and historical out-of-sample performance. The model forecasts a **soft inflation trajectory**, with YoY inflation expected to range between **0.76% and 3.69%** over the next year, averaging **2.72%**, indicating continued **disinflationary conditions**.

---

## 1. Data Overview

### Sources

* **CPI:** MOSPI Annex-VI monthly index series
* **WPI:** DPIIT monthly release (Base 2011–12 = 100)
* Time period: **Jan 2013 – Sep 2025**

### Data Notes

* CPI required extraction of Year–Month–Index format from a merged Excel sheet.
* WPI required reconstruction of fiscal-year matrices into continuous monthly series.
* Both series were cleaned, indexed by month-start, and saved as canonical CSVs.

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Trends

* CPI shows a long-term upward trend reflecting nominal price levels.
* WPI shows stronger cyclical movements driven by commodities.

### 2.2 YoY Inflation

* CPI YoY inflation follows a multi-year declining trend, stabilizing recently.
* WPI volatility has historically preceded CPI movements.

### 2.3 Seasonal Decomposition

* Clear **annual seasonality** in CPI due to food and festival cycles.
* Trend component shows structural moderation post-2016.

### 2.4 Rolling Statistics

* 6M and 12M rolling averages smooth out high-frequency noise.
* Variance clusters correspond to commodity cycles.

### 2.5 Correlation Analysis

* CPI–WPI correlation ≈ **0.26** (contemporaneous).
* Peak cross-correlation occurs at **WPI leading CPI by ~10 months**, reinforcing wholesale-to-retail pass-through dynamics.

---

## 3. Modeling Framework

### 3.1 ACF/PACF Findings

* ACF gradually decays → ARIMA differencing needed (d=1)
* PACF spike at lag 1 → MA(1) or AR(1) structure
* Seasonal spikes at lag 12 → SARIMA required

### 3.2 Model Selection

Two models were tested:

* **Manual SARIMA(1,1,1)(1,1,1,12)**
* **Auto-selected SARIMA(0,1,1)(1,0,1,12)**

Auto-SARIMA outperformed manual SARIMA:

* **Lower RMSE, MAE, MAPE** on out-of-sample test
* Cleaner residuals and no remaining autocorrelation at key lags

### 3.3 Final Model

**SARIMA(0,1,1)(1,0,1,12)** chosen for forecasting.

* Captures seasonal AR and MA behavior without over-differencing.
* Strong performance on the validation horizon.

---

## 4. Forecast Results

### 4.1 Baseline CPI Index Forecast (First 3 Months)

* **Oct 2025:** 198.29 (CI: 196.74 – 199.83)
* **Nov 2025:** 198.83 (CI: 196.26 – 201.39)
* **Dec 2025:** 198.16 (CI: 194.87 – 201.45)

### 4.2 YoY Inflation Forecast (First 3 Months)

* **Oct 2025:** 0.76%
* **Nov 2025:** 1.18%
* **Dec 2025:** 1.41%

### 4.3 YoY Range (Full 12 Months)

* **Min:** 0.76%
* **Mean:** 2.72%
* **Max:** 3.69%

Interpretation:
The inflation forecast lies **below RBI’s midpoint target (4%)** across the horizon, indicating **disinflation with limited upward pressure**.

---

## 5. Policy Insights (RBI-Style)

### 5.1 Inflation Conditions

* Baseline inflation remains **benign**, with significant slack in headline momentum.
* Food price seasonality remains a source of short-term volatility but does not alter medium-term trajectory.

### 5.2 Monetary Policy Stance

Given the projection path:

* A **hold / mildly accommodative** stance is appropriate.
* No justification for tightening unless core inflation or exogenous shocks rise.
* Real interest rates remain moderately positive, supporting stability.

### 5.3 Transmission Dynamics

* WPI → CPI pass-through is **gradual** (peak at ~10 months).
* This provides policymakers a **lead window** to anticipate retail inflation.

### 5.4 Risk Scenarios

**Upside Risks:**

* Oil price spikes (imported inflation)
* Weak monsoon affecting food inflation
* Fiscal policy shocks (administered price revisions)
* Currency depreciation

**Downside Risks:**

* Global commodity disinflation persistence
* Weak domestic demand

### 5.5 Policy Watchlist

* Core inflation persistence
* Food supply chain conditions
* FX movements
* Global crude benchmarks
* Credit growth and demand indicators

---

## 6. Scenario Analysis (Optional Module)

A simplified WPI-shock sensitivity test indicates:

* A +3 percentage-point shock to WPI YoY can raise CPI YoY by **0.5 to 1.2 percentage points** over 6–12 months (approx, model-dependent).
* Demonstrates the importance of monitoring wholesale inflation as an **early warning indicator**.

---

## 7. Recommendations for Further Enhancements

* Add SARIMAX with external regressors (crude, FX, core inflation)
* Build a structural VAR for impulse-response analysis
* Implement rolling-origin cross-validation
* Develop automated data-update pipeline
* Add nowcasting using high-frequency indicators

---

## 8. Conclusion

The SARIMA-based inflation forecasting framework provides a **robust, interpretable, and policy-relevant** view of India’s inflation trajectory. Forecasted YoY inflation averaging **~2.7%** suggests continued disinflation, permitting the RBI to maintain a balanced and data-driven policy stance.

This research-grade system lays the groundwork for extensions into VAR modeling, scenario stress-testing, and integrated macro-policy dashboards.
