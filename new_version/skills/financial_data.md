# SKILL: Financial Data Microstructure, Modularity & Target Definition

## Context
You are acting as a Data Engineer and Quantitative Finance Assistant. The project is a generalized asset direction predictor. The pipeline must dynamically fetch a Primary Asset (e.g., BTC, SPY, GLD) and a parameterized stack of Secondary Features (e.g., On-chain data, VIX, Interest Rates, DXY, M2SL).

## Trigger
Apply these rules strictly whenever creating or modifying `src/data_loader.py`, `src/features.py`, and any data-merging scripts.

## Core Directives

### 1. Modular Fetching Architecture
- **Primary Asset:** Create a distinct function (e.g., `fetch_primary_asset(ticker)`) that strictly pulls OHLCV data via yfinance.
- **Secondary Features:** Create a flexible function or routing mechanism (e.g., `fetch_secondary_features(feature_list)`) that can pull data from various sources (Coinmetrics for crypto on-chain, FRED/yfinance for macro indicators like DXY, VIX, Federal Funds Rate).
- Never hardcode the asset ticker or the feature stack in the core functions. They must be parameterized to allow the Master Notebook to orchestrate different experiments.

### 2. Log Returns & Target Calculation (Strict Formula)
- The target direction MUST be calculated using log returns, following industry quantitative standards.
- First, calculate the daily log return: `r_t = ln(Close_t) - ln(Close_{t-1})`.
- Second, create the target variable `Price_Direction` by shifting the log return backward: The target at time `T` represents the sign of `r_{T+1}`.
- If `r_{T+1} > 0`, Direction = 1 (UP). If `r_{T+1} <= 0`, Direction = 0 (DOWN).
- Ensure `r_t` is also kept as an independent predictive feature, but obviously do not include the shifted `r_{T+1}` as a feature, to prevent fatal data leakage.

### 3. Asynchronous Frequencies & Calendar Merging
- **Macro Data vs Daily Markets:** Macro data (like M2SL - Money Supply) is often reported monthly, whereas OHLCV is daily. 
- When merging datasets of different frequencies, use `forward fill` (`ffill()`) for the lower frequency macroeconomic features so they populate the daily rows, but NEVER backward fill (`bfill()`). We can only know the macro data after it is published.
- All datetime indices must be standardized to timezone-aware UTC prior to the final merge.

### 4. Idempotent Caching
- When the user queries a specific combination (e.g., "GLD" + "VIX" + "DXY"), the data loader should cache the raw downloads in `data/raw/` to prevent redundant API calls on subsequent Notebook runs.