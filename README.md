[![python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-complete-green.svg)]
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)]

# India Inflation Forecasting System (CPI + WPI)

A research-grade inflation forecasting pipeline modeled after the analytical standards of the **Reserve Bank of India (RBI)**, **ISI**, **IIT**, and global macro-research groups. The project includes:

* Complete CPI & WPI data ingestion (2012–2025)
* Cleaning & structuring of official MOSPI/DPIIT data
* Extensive EDA (trend, YoY/MoM, decomposition, rolling windows)
* Cross-correlation & seasonal diagnostics
* SARIMA-based forecasting engine
* 12‑month baseline forecast + confidence intervals
* Policy insights aligned with Monetary Policy Report standards

---

## 🚀 Project Structure

```
India-Inflation-Forecasting/
├── data/                 # Raw & cleaned CPI/WPI
├── notebooks/            # Step-by-step Jupyter workflows
├── src/                  # Production-grade Python modules
├── plots/                # Auto-generated charts
├── outputs/              # Forecast CSVs
├── reports/              # Policy-grade analysis
├── README.md
└── requirements.txt
```

---

## 📌 Features

### **1. Data Pipeline (Production Grade)**

* Automated cleaning for CPI Annex-VI format
* Automated cleaning for WPI multi-year matrix structure
* Type-safe functions with logging & error-handling

### **2. EDA & Insights**

* Trend charts for CPI, WPI
* YoY & MoM inflation
* Seasonal decomposition
* Rolling averages
* CPI–WPI correlation + heatmap
* WPI leading indicator analysis

### **3. Modeling Framework**

* Manual SARIMA (ACF/PACF informed)
* Auto ARIMA (AIC-driven)
* Diagnostics (Ljung–Box, residual plots)
* Train–test evaluation (RMSE/MAE/MAPE)

### **4. Forecast Engine**

* SARIMA(0,1,1)(1,0,1,12) selected as best model
* 12‑month forecast with 95% CI
* YoY inflation projections
* Exportable CSV + PNG

### **5. Policy Insights (RBI‑Style)**

* Disinflation path assessment
* Risk drivers (food, oil, FX, fiscal)
* Repo‑rate stance recommendations
* Transmission analysis

---

## 🛠 Installation

### Create environment

```
conda create -n india_inflation python=3.10
conda activate india_inflation
pip install -r requirements.txt
```

### Run notebooks

* Open Jupyter Lab or VS Code
* Run `1_data_cleaning.ipynb`
* Then run `2_EDA_and_Modeling.ipynb`

---

## 📈 Results Summary

### **Baseline Forecast (Next 12 Months)**

* YoY inflation range: **0.76% → 3.69%**
* Mean YoY inflation: **2.72%**
* Interpreted as **soft disinflationary conditions**
* Suggests **hold or accommodative stance** unless risks materialize

---

## 📊 Core Files in `src/`

* **data_processing.py** — Raw → Clean dataset pipelines
* **modeling.py** — SARIMA fitting, auto-arima, diagnostics, metrics
* **forecasting.py** — Forecasts, CI bands, YoY generation, scenarios
* **utils.py** — Logging, formatting, shared helpers

---

## 📦 Outputs

* `outputs/cpi_forecast_12m.csv`
* `outputs/cpi_forecast_12m_yoy.csv`
* `plots/*` (all charts)

---

## 📘 Author

**Naman Narendra Choudhary**

* B.Tech (ECE)
* Aspiring quant, macro researcher, and future IIM/Harvard/Stanford MBA
* Research-driven mindset blending **engineering + finance + macro policy**

---

## ⭐ Acknowledgements

* MOSPI & DPIIT (CPI/WPI sources)
* RBI Monetary Policy Reports for analytical inspiration
