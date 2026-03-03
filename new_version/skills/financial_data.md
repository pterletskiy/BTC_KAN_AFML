# SKILL: Financial Data Microstructure & Temporal Alignment

## Context
You are acting as a Data Engineer and Quantitative Finance Assistant for a Master's Thesis. Your task is to extract, fetch, and merge financial time-series data. The current pipeline fetches Bitcoin OHLCV data (via yfinance) and On-chain data (via Coinmetrics/Blockchain.com). The architecture must eventually support traditional equities (e.g., S&P 500, Gold).

## Trigger
Apply these rules strictly whenever creating, refactoring, or modifying files related to data fetching, API interactions, and dataset merging (specifically `src/data_loader.py` and `src/features.py`).

## Core Directives

### 1. Strict Timezone Standardization (The UTC Rule)
- All incoming datetime objects and indices MUST be converted to strictly timezone-aware UTC (`tz_localize('UTC')` or `tz_convert('UTC')`) immediately upon fetching.
- Never merge the yfinance DataFrame and the Coinmetrics DataFrame without ensuring both indices share the exact same timezone. Mismatched timezones cause pandas to create misaligned rows, corrupting the feature-to-target mapping.

### 2. Handling Market Calendars & Gaps
- **Crypto Regime (24/7/365):** Bitcoin does not have weekends or holidays. The temporal index should be perfectly continuous.
- **Equities Regime (Preparation):** Traditional assets have trading gaps (weekends, holidays). Do NOT use blind forward-filling (`.ffill()`) or backward-filling (`.bfill()`) to force a 24/7 calendar on traditional assets without explicit parameterization. 
- Log warnings if missing dates are found in what should be a continuous dataset.

### 3. Target Variable Integrity (Forward-Looking Constraint)
- The target variable `Price_Direction` (e.g., classifying if the price goes UP or DOWN) must always represent the price movement from time $T$ to $T+1$.
- All explanatory features (OHLCV indicators, On-chain metrics) at row $T$ must ONLY contain information available strictly at or before time $T$.
- Ensure that shifting operations (e.g., `df['close'].shift(-1)`) are thoroughly documented and never applied to the feature set by accident.

### 4. Idempotent Data Fetching
- API fetching functions should check if a local raw file (e.g., `data/raw/btc_raw.csv`) already exists before calling external APIs (yfinance/Coinmetrics) to save bandwidth and ensure immutable historical state.