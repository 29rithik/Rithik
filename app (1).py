#!/usr/bin/env python3
"""Rithik Risk — complete single-file Streamlit application.

Python 3.10 or newer. No other project files, assets or sample CSVs are needed.

Install the packages once:
    python -m pip install "streamlit>=1.50,<2" "pandas>=2.1,<3" "numpy>=1.24,<3" "plotly>=5.24,<7" "yfinance>=0.2.54,<2"

Run:
    python app.py

Or:
    python -m streamlit run app.py --server.address 0.0.0.0

Codespaces: open port 8501 in the Ports panel.
Render: choose a Python web service; use the install command above as its build
command and `python app.py` as its start command. The launcher honors PORT.

Features: portfolio and benchmark history, beta/volatility/Sharpe/drawdown,
covariance risk contributions, correlation network, holding and debt scenarios,
ten optional FRED series, CSV downloads, and a session journal with JSON restore.
Demo prices and macro values are simulated, never claimed to be live data.
Choose Enter stocks manually to enter tickers and percentage weights and fetch
Yahoo Finance adjusted daily prices; CSV uploads remain available. Manual mode
requires yfinance and internet access. Journal notes require export to persist.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
import csv
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from zoneinfo import ZoneInfo

try:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "A required Python package is missing. Run this once, then start again:\n"
        'python -m pip install "streamlit>=1.50,<2" "pandas>=2.1,<3" "numpy>=1.24,<3" "plotly>=5.24,<7"'
    ) from exc

TRADING_DAYS = 252
MIN_RETURNS = 21


class DataValidationError(ValueError):
    """A user-readable data problem, with no credential-bearing details."""


def clean_ticker(value: Any) -> str:
    """Normalize a single Yahoo-format symbol without changing its identity."""
    if not isinstance(value, str):
        raise DataValidationError("Enter a ticker symbol for every stock.")
    ticker = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,19}", ticker):
        raise DataValidationError("Use one Yahoo-format ticker per row, such as AAPL, MSFT or BRK-B.")
    return ticker


def manual_holdings(rows: pd.DataFrame) -> pd.DataFrame:
    """Convert the editor's percent weights to the existing holdings contract."""
    if not isinstance(rows, pd.DataFrame) or not {"Ticker", "Weight (%)"}.issubset(rows.columns):
        raise DataValidationError("Enter ticker symbols and weights in the portfolio table.")
    records = []
    for _, row in rows.iterrows():
        symbol, weight = row["Ticker"], row["Weight (%)"]
        blank_symbol = pd.isna(symbol) or not str(symbol).strip()
        blank_weight = pd.isna(weight) or not str(weight).strip()
        if blank_symbol and (blank_weight or str(weight).strip() in {"0", "0.0"}):
            continue
        if blank_symbol:
            raise DataValidationError("A row has a weight but no ticker. Enter the ticker or remove that row.")
        ticker = clean_ticker(symbol)
        if blank_weight:
            raise DataValidationError(f"Enter a percentage weight for {ticker}.")
        records.append({"ticker": ticker, "name": ticker, "sector": "Unclassified", "weight": weight})
    if not records:
        raise DataValidationError("Add at least one stock and its percentage weight.")
    if len(records) > 50:
        raise DataValidationError("Use up to 50 holdings per portfolio.")
    return read_holdings_csv(pd.DataFrame(records).to_csv(index=False))


def downloaded_close_prices(raw: pd.DataFrame, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Extract adjusted Close from yfinance's flat or MultiIndex output."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        raise DataValidationError("Yahoo Finance returned no price history. Check your tickers and internet connection, or retry later.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = None
        for level in range(raw.columns.nlevels):
            if "Close" in raw.columns.get_level_values(level):
                close = raw.xs("Close", axis=1, level=level)
                break
        if close is None or isinstance(close.columns, pd.MultiIndex):
            raise DataValidationError("The provider response did not contain usable adjusted closing prices.")
    elif "Close" in raw.columns and len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise DataValidationError("The provider response did not contain usable adjusted closing prices.")
    close = close.copy()
    close.columns = [str(column).strip().upper() for column in close.columns]
    if close.columns.has_duplicates:
        raise DataValidationError("The provider returned duplicate price columns. Retry the download.")
    missing = [ticker for ticker in tickers if ticker not in close or close[ticker].dropna().empty]
    if missing:
        raise DataValidationError("No usable prices for: " + ", ".join(missing) + ". Check the symbols or retry later. No holdings were removed or replaced.")
    close = close.loc[:, list(tickers)]
    try:
        index = pd.DatetimeIndex(pd.to_datetime(close.index))
        if index.tz is not None:
            index = index.tz_localize(None)
        close.index = index.normalize()
        close.index.name = "date"
        if close.index.hasnans or close.index.has_duplicates:
            raise ValueError("Invalid dates")
        for ticker in tickers:
            close[ticker] = pd.to_numeric(close[ticker], errors="raise")
            values = close[ticker].dropna().to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError("Invalid prices")
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataValidationError("The downloaded history has invalid prices or dates. Retry or use adjusted-price CSV data.") from exc
    return close.sort_index().dropna(how="all")


def fetch_stock_prices(tickers: tuple[str, ...], years: int, end_date: str) -> pd.DataFrame:
    """Download daily adjusted closes; no silent fallback or forward filling."""
    try:
        import yfinance as yf
    except ImportError:
        raise DataValidationError('Manual entry needs one extra package. In your terminal run: python -m pip install "yfinance>=0.2.54,<2"') from None
    try:
        start = (pd.Timestamp(end_date) - pd.DateOffset(years=years)).date().isoformat()
        raw = yf.download(
            tickers=list(tickers), start=start, end=end_date, interval="1d",
            auto_adjust=True, back_adjust=False, actions=False, repair=False,
            keepna=True, threads=False, progress=False, timeout=20,
            group_by="column", multi_level_index=True, ignore_tz=True,
        )
    except Exception:
        raise DataValidationError("Could not download Yahoo Finance history. Check your connection and symbols, then retry. The provider may temporarily limit requests; CSV uploads remain available.") from None
    prices = downloaded_close_prices(raw, tickers)
    prices.attrs["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    return prices


def manual_request_key(holdings: pd.DataFrame, benchmark: str, years: int) -> tuple:
    """Include weights so edits never display results for an older allocation."""
    return (tuple(zip(holdings["ticker"].tolist(), holdings["weight"].tolist())), benchmark, years)


def manual_portfolio_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Editable holdings, explicit load action, and per-session downloaded data."""
    st.caption("Edit tickers and percentage weights below. Add stocks using the empty row at the bottom; select a row to delete it.")
    st.caption("The starting rows are examples. Replace them with your holdings. Use prices in one currency; no currency conversion is applied.")
    initial = pd.DataFrame({"Ticker": ["GIS", "FRT", "MO", "CI", "D"], "Weight (%)": [30.0, 20.0, 15.0, 20.0, 15.0]})
    if "manual_editor_base" not in st.session_state:
        st.session_state["manual_editor_base"] = initial
    elif "manual_holdings_editor" not in st.session_state and "manual_last_rows" in st.session_state:
        # Widget state is cleared when changing source; retain the user's draft.
        st.session_state["manual_editor_base"] = st.session_state["manual_last_rows"].copy().reset_index(drop=True)
    entered = st.data_editor(
        st.session_state["manual_editor_base"], num_rows="dynamic", hide_index=True, width="stretch", key="manual_holdings_editor",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo-format symbols, for example AAPL or BRK-B.", required=True),
            "Weight (%)": st.column_config.NumberColumn("Weight (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", required=True),
        },
    )
    st.session_state["manual_last_rows"] = entered.copy()
    displayed_total = pd.to_numeric(entered["Weight (%)"], errors="coerce").sum()
    st.caption(f"Entered weights: {displayed_total:g}% · target: 100%")
    benchmark_text = st.text_input("Benchmark ticker", value=st.session_state.get("manual_saved_benchmark", "SPY"), key="manual_benchmark", help="Comparison only. This does not add the benchmark to your holdings.")
    st.session_state["manual_saved_benchmark"] = benchmark_text
    history_options = [1, 3, 5, 10]
    years = st.selectbox("History to download", history_options, index=history_options.index(st.session_state.get("manual_saved_years", 5)), format_func=lambda value: f"{value} year{'s' if value != 1 else ''}", key="manual_history_years")
    st.session_state["manual_saved_years"] = years
    load_clicked = st.button("Load / update portfolio", type="primary", key="load_manual_portfolio")
    st.caption("Downloads Yahoo Finance daily adjusted prices, excluding today's session. No API key is needed. Prices may be delayed or revised.")
    try:
        holdings = manual_holdings(entered)
        benchmark = clean_ticker(benchmark_text)
    except DataValidationError as exc:
        (st.error if load_clicked else st.info)(str(exc))
        st.stop()
    request_key = manual_request_key(holdings, benchmark, years)
    end_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    symbols = tuple(sorted(set(holdings["ticker"].tolist() + [benchmark])))
    download_key = (symbols, years, end_date)
    saved = st.session_state.get("manual_portfolio")
    if load_clicked:
        try:
            if saved is not None and saved["download_key"] == download_key and saved["request_key"] != request_key:
                prices = saved["prices"].copy()
            else:
                with st.spinner("Loading adjusted stock and benchmark history…"):
                    prices = fetch_stock_prices(symbols, years, end_date)
            # Reject incomplete histories before replacing a working snapshot.
            analyze_portfolio(prices, holdings, benchmark, lookback=None)
            saved = {"request_key": request_key, "download_key": download_key, "prices": prices}
            st.session_state["manual_portfolio"] = saved
        except DataValidationError as exc:
            st.error(str(exc))
            st.stop()
    if saved is None:
        st.info("Enter your stocks and weights, then click Load / update portfolio.")
        st.stop()
    if saved["request_key"] != request_key:
        st.info("Your entries changed. Click Load / update portfolio to calculate the new portfolio.")
        st.stop()
    prices = saved["prices"].copy()
    st.caption("Price retrieval: " + prices.attrs.get("retrieved_at", "unknown") + ". Click Load / update portfolio again to refresh unchanged entries.")
    export = holdings.copy()
    export["weight"] *= 100
    st.download_button("Download my holdings", csv_bytes(export, False), "my-holdings.csv", "text/csv")
    return prices, holdings, benchmark


def _csv_frame(source: Any) -> pd.DataFrame:
    if hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, bytes):
        raw = source
    elif isinstance(source, str) and ("\n" in source or "\r" in source):
        raw = source
    else:
        raw = Path(source).read_bytes()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8-sig")
    raw = raw.lstrip("\ufeff")
    try:
        header = next(csv.reader(StringIO(raw)))
    except (StopIteration, csv.Error) as exc:
        raise DataValidationError("The CSV is empty or has an invalid header.") from exc
    normalized = [str(h).strip().upper() for h in header]
    if not normalized or any(not h for h in normalized):
        raise DataValidationError("Every CSV column needs a nonempty name.")
    if len(normalized) != len(set(normalized)):
        raise DataValidationError("Duplicate column names are not allowed, including case-only differences.")
    try:
        frame = pd.read_csv(StringIO(raw))
    except Exception as exc:
        raise DataValidationError("Could not read the CSV. Use a UTF-8 comma-separated file.") from exc
    frame.columns = [str(h).strip() for h in frame.columns]
    if frame.empty:
        raise DataValidationError("The CSV has no data rows.")
    return frame


def read_prices_csv(source: Any) -> pd.DataFrame:
    """Read date,TICKER,... daily adjusted prices; blanks stay missing."""
    frame = _csv_frame(source)
    frame.columns = [c.upper() for c in frame.columns]
    if "DATE" not in frame or len(frame.columns) < 2:
        raise DataValidationError("Prices need a date column and at least one ticker column.")
    dates = pd.to_datetime(frame.pop("DATE"), format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise DataValidationError("Every price date must be valid YYYY-MM-DD text.")
    if dates.duplicated().any():
        raise DataValidationError("Duplicate price dates are not allowed.")
    for column in frame.columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & values.isna()
        if invalid.any():
            raise DataValidationError(f"{column}: prices must be numeric or blank.")
        present = values.dropna()
        if present.empty:
            raise DataValidationError(f"{column}: all prices are missing.")
        if (present <= 0).any() or not np.isfinite(present.to_numpy()).all():
            raise DataValidationError(f"{column}: every supplied price must be finite and greater than zero.")
        frame[column] = values.astype(float)
    frame.index = pd.DatetimeIndex(dates, name="date")
    return frame.sort_index()


def read_holdings_csv(source: Any) -> pd.DataFrame:
    """Read long-only weights in PERCENT units: 20 or 20%, not 0.20."""
    frame = _csv_frame(source)
    frame.columns = [c.lower() for c in frame.columns]
    if len(frame.columns) != 4 or set(frame.columns) != {"ticker", "name", "sector", "weight"}:
        raise DataValidationError("Holdings CSV must have ticker,name,sector,weight columns. Weights use percentages; 20 or 20% means 20%.")
    if frame["ticker"].isna().any():
        raise DataValidationError("Every holding needs a ticker.")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    if frame["ticker"].eq("").any() or frame["ticker"].duplicated().any():
        raise DataValidationError("Holding tickers must be nonempty and unique.")
    weight_text = frame["weight"].astype(str).str.strip()
    if not weight_text.str.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)%?").all():
        raise DataValidationError("Each weight must be a number optionally followed by one percent sign, such as 20 or 20%.")
    text = weight_text.str.removesuffix("%")
    numeric = pd.to_numeric(text, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any() or (numeric > 100).any():
        raise DataValidationError("Weights must be finite percentages between 0 and 100; short positions are not supported.")
    total = float(numeric.sum())
    if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=0.0001):
        raise DataValidationError(f"Weights sum to {total:g}%. Use percentages totaling 100 (for example, 20 or 20%). Weights are not silently normalized.")
    frame["weight"] = numeric.astype(float) / 100.0
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame["name"] = frame["name"].where(frame["name"].ne(""), frame["ticker"])
    frame["sector"] = frame["sector"].fillna("").astype(str).str.strip().replace("", "Unclassified")
    return frame[["ticker", "name", "sector", "weight"]].reset_index(drop=True)


def maximum_drawdown(wealth: pd.Series) -> dict[str, Any]:
    """Wealth MUST include the starting value, so first-period losses count."""
    if wealth.empty or wealth.isna().any() or (wealth <= 0).any():
        raise DataValidationError("Drawdown requires a positive wealth series including its starting value.")
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    trough = drawdown.idxmin()
    maximum = float(drawdown.loc[trough])
    if maximum >= -1e-14:
        return {"value": 0.0, "peak": None, "trough": None, "recovery": None, "series": drawdown}
    before = wealth.loc[:trough]
    peak_value = float(before.max())
    peak = before.index[np.flatnonzero(np.isclose(before.to_numpy(), peak_value, rtol=1e-12, atol=0))[-1]]
    recoveries = wealth.loc[trough:]
    recovered = recoveries[recoveries >= peak_value * (1 - 1e-12)]
    recovery = recovered.index[0] if not recovered.empty else None
    return {"value": maximum, "peak": peak, "trough": trough, "recovery": recovery, "series": drawdown}


def _beta(returns: pd.Series, benchmark: pd.Series) -> float:
    variance = float(benchmark.var(ddof=1))
    return float(returns.cov(benchmark) / variance) if variance > 1e-18 else float("nan")


def return_metrics(returns: pd.Series, benchmark: pd.Series, wealth: pd.Series, annual_risk_free: float = 0.0) -> dict[str, float]:
    if annual_risk_free <= -1 or not math.isfinite(annual_risk_free):
        raise DataValidationError("The annual risk-free assumption must be finite and greater than -100%.")
    n = len(returns)
    if n < 2:
        raise DataValidationError("At least two returns are required.")
    daily_rf = (1.0 + annual_risk_free) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - daily_rf
    sample_std = float(returns.std(ddof=1))
    ending = float(wealth.iloc[-1] / wealth.iloc[0])
    return {
        "total_return": ending - 1.0,
        "annualized_return": ending ** (TRADING_DAYS / n) - 1.0,
        "annualized_volatility": sample_std * math.sqrt(TRADING_DAYS),
        "beta": _beta(returns, benchmark),
        "sharpe": float(excess.mean() / sample_std * math.sqrt(TRADING_DAYS)) if sample_std > 1e-14 else float("nan"),
        "maximum_drawdown": maximum_drawdown(wealth)["value"],
        "benchmark_correlation": float(returns.corr(benchmark)) if sample_std > 1e-14 and benchmark.std(ddof=1) > 1e-14 else float("nan"),
    }


@dataclass
class PortfolioResult:
    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    returns: pd.Series
    benchmark_returns: pd.Series
    wealth: pd.DataFrame
    correlations: pd.DataFrame
    asset_metrics: pd.DataFrame
    metrics: dict[str, float]
    drawdown: dict[str, Any]
    warnings: list[str]
    removed_rows: int
    periods: int
    mode: str
    benchmark: str


def analyze_portfolio(prices: pd.DataFrame, holdings: pd.DataFrame, benchmark: str, lookback: int | None = 252, mode: str = "daily", annual_risk_free: float = 0.0) -> PortfolioResult:
    """Align all requested prices, then calculate transparent historical metrics.

    Common complete rows are used, never filled. Missing dates can make returns
    span multiple sessions; the returned warning explains that annualization is
    approximate in that case. Lookback counts RETURN observations.
    """
    if mode not in {"daily", "buy_hold"}:
        raise DataValidationError("Mode must be daily or buy_hold.")
    if lookback is not None and (not isinstance(lookback, int) or lookback < MIN_RETURNS):
        raise DataValidationError(f"Lookback must be at least {MIN_RETURNS} return observations.")
    tickers = holdings["ticker"].tolist()
    if not tickers or len(tickers) != len(set(tickers)):
        raise DataValidationError("Holdings must contain unique ticker symbols.")
    weights = holdings.set_index("ticker")["weight"].reindex(tickers).astype(float)
    if not np.isfinite(weights.to_numpy()).all() or (weights < 0).any() or (weights > 1).any() or not math.isclose(float(weights.sum()), 1, abs_tol=1e-6, rel_tol=0):
        raise DataValidationError("Analytical weights must be decimal values between 0 and 1 totaling 1.")
    required = list(dict.fromkeys(tickers + [benchmark]))
    missing = [ticker for ticker in required if ticker not in prices]
    if missing:
        raise DataValidationError("Missing price columns: " + ", ".join(missing))
    if prices.index.has_duplicates:
        raise DataValidationError("Duplicate price dates are not allowed.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataValidationError("Price rows must have a daily DatetimeIndex.")
    selected = prices[required].sort_index()
    present = selected.to_numpy(dtype=float)
    finite_values = present[~np.isnan(present)]
    if (finite_values <= 0).any() or not np.isfinite(finite_values).all():
        raise DataValidationError("Prices must be positive finite numbers or missing.")
    aligned = selected.dropna(how="any")
    if lookback is not None:
        aligned = aligned.tail(lookback + 1)
    if len(aligned) < MIN_RETURNS + 1:
        raise DataValidationError(f"Need at least {MIN_RETURNS + 1} common price dates ({MIN_RETURNS} returns). Found {len(aligned)}.")
    in_window = selected.loc[aligned.index[0]:aligned.index[-1]]
    removed = len(in_window) - len(aligned)
    warnings = []
    if weights.eq(0).any():
        warnings.append("Zero-weight holdings still participate in common-date alignment. Remove them if you do not want their missing prices to shorten the analysis period.")
    if removed:
        warnings.append(f"Removed {removed} incomplete price rows inside this analysis window. Returns bridge the retained dates and can span multiple market sessions. Rebalancing occurs on retained dates, and 252-observation annualization is approximate. No prices were filled.")
    n = len(aligned) - 1
    if n < 60:
        warnings.append(f"Only {n} return observations are available. Beta, volatility, correlation and Sharpe can be unstable in short samples.")
    if lookback and n < lookback:
        warnings.append(f"Requested {lookback} returns; only {n} are available after alignment.")
    calendar_gaps = aligned.index.to_series().diff().dt.total_seconds().dropna() / 86400
    if calendar_gaps.median() > 4:
        raise DataValidationError("Prices appear weekly or monthly. Upload daily adjusted prices; 252-day annualization requires daily data.")
    wide_gaps = int(calendar_gaps.gt(4).sum())
    if wide_gaps:
        warnings.append(f"{wide_gaps} gap(s) exceed four calendar days. Each retained interval is treated as one trading-day return; daily annualization may be unreliable for sparse data.")
    returns = aligned.pct_change(fill_method=None).iloc[1:]
    asset_returns = returns[tickers]
    benchmark_returns = returns[benchmark].rename("Benchmark")
    normalized = aligned[tickers] / aligned[tickers].iloc[0]
    if mode == "daily":
        portfolio_returns = asset_returns.mul(weights, axis="columns").sum(axis=1).rename("Portfolio")
        portfolio_wealth = pd.concat([pd.Series([1.0], index=aligned.index[:1]), (1 + portfolio_returns).cumprod()])
    else:
        portfolio_wealth = normalized.mul(weights, axis="columns").sum(axis=1)
        portfolio_returns = portfolio_wealth.pct_change(fill_method=None).iloc[1:].rename("Portfolio")
    portfolio_wealth.name = "Portfolio"
    benchmark_wealth = (aligned[benchmark] / aligned[benchmark].iloc[0]).rename("Benchmark")
    wealth = pd.concat([portfolio_wealth, benchmark_wealth], axis=1)
    metric = return_metrics(portfolio_returns, benchmark_returns, portfolio_wealth, annual_risk_free)
    rows = []
    for ticker in tickers:
        item = return_metrics(asset_returns[ticker], benchmark_returns, normalized[ticker], annual_risk_free)
        rows.append({"ticker": ticker, "weight": float(weights[ticker]), **item})
    return PortfolioResult(
        prices=aligned,
        asset_returns=asset_returns,
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        wealth=wealth,
        correlations=returns[required].corr(),
        asset_metrics=pd.DataFrame(rows).set_index("ticker"),
        metrics=metric,
        drawdown=maximum_drawdown(portfolio_wealth),
        warnings=warnings,
        removed_rows=removed,
        periods=n,
        mode=mode,
        benchmark=benchmark,
    )


def correlation_edges(correlations: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise DataValidationError("The correlation threshold must be between 0 and 1.")
    edges = []
    labels = correlations.columns.tolist()
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            correlation = float(correlations.loc[left, right])
            if math.isfinite(correlation) and abs(correlation) >= threshold:
                edges.append({"asset_a": left, "asset_b": right, "correlation": correlation, "relationship": "positive" if correlation >= 0 else "negative"})
    return pd.DataFrame(edges, columns=["asset_a", "asset_b", "correlation", "relationship"])


def weighted_scenario(holdings: pd.DataFrame, shocks_percent: dict[str, float]) -> pd.DataFrame:
    rows = []
    for row in holdings.itertuples():
        if row.ticker not in shocks_percent:
            raise DataValidationError(f"Missing scenario shock for {row.ticker}.")
        shock = float(shocks_percent[row.ticker])
        if not math.isfinite(shock) or shock < -100:
            raise DataValidationError("Scenario stock returns must be finite and cannot be below -100%.")
        rows.append({"ticker": row.ticker, "weight_percent": row.weight * 100, "shock_percent": shock, "contribution_percentage_points": row.weight * shock})
    return pd.DataFrame(rows)


def financing_scenario(principal: float, change_bps: float, months: float, tax_rate: float, diluted_shares: float) -> dict[str, float]:
    values = [principal, change_bps, months, tax_rate, diluted_shares]
    if not all(math.isfinite(v) for v in values) or principal < 0 or not 0 <= months <= 12 or not 0 <= tax_rate <= 1 or diluted_shares <= 0:
        raise DataValidationError("Use nonnegative repricing principal, 0–12 months, a 0–100% tax rate, and positive shares.")
    interest = principal * change_bps / 10000 * months / 12
    net_income_change = -interest * (1 - tax_rate)
    return {"incremental_interest": interest, "net_income_change": net_income_change, "eps_change": net_income_change / diluted_shares}


def make_demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic synthetic weekday data, not observed security prices."""
    rng = np.random.default_rng(2707)
    dates = pd.bdate_range("2022-11-01", "2026-10-30")
    market = rng.normal(0.0003, 0.009, len(dates) - 1)
    market[110:128] -= 0.006
    market[565:580] -= 0.005
    tickers = ["GIS", "FRT", "MO", "CI", "D"]
    betas = [0.45, 0.85, 0.5, 0.65, 0.7]
    bases = [65, 95, 44, 220, 55]
    data = {}
    for ticker, beta, base in zip(tickers, betas, bases):
        daily = beta * market + rng.normal(0.00012, 0.006, len(market))
        data[ticker] = base * np.r_[1.0, np.cumprod(1 + daily)]
    data["SPY"] = 400 * np.r_[1.0, np.cumprod(1 + market)]
    prices = pd.DataFrame(data, index=dates)
    prices.index.name = "date"
    holdings = pd.DataFrame({
        "ticker": tickers,
        "name": ["General Mills", "Federal Realty", "Altria", "The Cigna Group", "Dominion Energy"],
        "sector": ["Consumer Staples", "Real Estate", "Consumer Staples", "Health Care", "Utilities"],
        "weight": [0.30, 0.20, 0.15, 0.20, 0.15],
    })
    return prices, holdings


FRED_SERIES = {
    "PCEPILFE": {"name": "Core PCE inflation", "category": "Inflation", "frequency": "Monthly", "units": "Index, 2017=100; seasonally adjusted", "transform": "yoy", "provider": "BEA"},
    "UNRATE": {"name": "Unemployment rate", "category": "Growth & labor", "frequency": "Monthly", "units": "Percent; seasonally adjusted", "transform": "rate", "provider": "BLS"},
    "PAYEMS": {"name": "Nonfarm payrolls", "category": "Growth & labor", "frequency": "Monthly", "units": "Thousands of persons; seasonally adjusted", "transform": "jobs", "provider": "BLS"},
    "ICSA": {"name": "Initial unemployment claims", "category": "Growth & labor", "frequency": "Weekly", "units": "Persons; seasonally adjusted", "transform": "claims", "provider": "U.S. Employment and Training Administration"},
    "DGS10": {"name": "10-year Treasury yield", "category": "Rates & credit", "frequency": "Daily", "units": "Percent; not seasonally adjusted", "transform": "yield", "provider": "Federal Reserve Board"},
    "DGS2": {"name": "2-year Treasury yield", "category": "Rates & credit", "frequency": "Daily", "units": "Percent; not seasonally adjusted", "transform": "yield", "provider": "Federal Reserve Board"},
    "T10Y3M": {"name": "10-year minus 3-month Treasury spread", "category": "Rates & credit", "frequency": "Daily", "units": "Percentage points; not seasonally adjusted", "transform": "spread", "provider": "Federal Reserve Bank of St. Louis"},
    "BAMLH0A0HYM2": {"name": "U.S. high-yield option-adjusted spread", "category": "Rates & credit", "frequency": "Daily", "units": "Percent; not seasonally adjusted", "transform": "spread", "provider": "ICE Data Indices, LLC"},
    "INDPRO": {"name": "Industrial production", "category": "Growth & labor", "frequency": "Monthly", "units": "Index, 2017=100; seasonally adjusted", "transform": "yoy", "provider": "Federal Reserve Board"},
    "PCEC96": {"name": "Real personal consumption", "category": "Growth & labor", "frequency": "Monthly", "units": "Billions of chained 2017 dollars; seasonally adjusted annual rate", "transform": "yoy", "provider": "BEA"},
}


def fred_summary(series_id: str, series: pd.Series) -> dict[str, Any]:
    """Transform level series without confusing a percentage with a basis point."""
    if series_id not in FRED_SERIES:
        raise DataValidationError("Unknown macro series.")
    series = series.dropna().sort_index().astype(float)
    if series.empty:
        raise DataValidationError("No numeric macro observations are available.")
    meta = FRED_SERIES[series_id]
    kind = meta["transform"]
    if meta["frequency"] == "Monthly":
        series.index = series.index.to_period("M").to_timestamp()
        if series.index.has_duplicates:
            raise DataValidationError("Monthly series contains duplicate observation months.")
        series = series.asfreq("MS")  # Missing months stay missing; shifts remain calendar-month shifts.
    elif meta["frequency"] == "Weekly":
        series = series.asfreq("7D")  # Do not call four observations four weeks across missing weeks.
    if kind == "yoy":
        chart = (series / series.shift(12) - 1) * 100
        momentum = ((series / series.shift(3)) ** 4 - 1) * 100
        unit, secondary_label = "% YoY", "3-month annualized"
        secondary = float(momentum.iloc[-1])
    elif kind == "jobs":
        chart = series.diff()
        unit, secondary_label = "thousand jobs added", "3-month mean, thousand"
        secondary = float(chart.rolling(3).mean().iloc[-1])
    elif kind == "claims":
        chart = series.rolling(4).mean() / 1000
        unit, secondary_label = "thousand, 4-week mean", "Change vs 4 observations ago, thousand"
        secondary = float(chart.diff(4).iloc[-1])
    elif kind == "rate":
        chart = series.copy()
        unit, secondary_label = "%", "3-month change, percentage points"
        secondary = float(chart.diff(3).iloc[-1])
    elif kind == "yield":
        chart = series.copy()
        unit, secondary_label = "%", "21-observation change, basis points"
        secondary = float(chart.diff(21).iloc[-1] * 100)
    else:
        chart = series * 100
        unit, secondary_label = "basis points", "21-observation change, basis points"
        secondary = float(chart.diff(21).iloc[-1])
    if pd.isna(chart.iloc[-1]):
        raise DataValidationError("The latest observation lacks complete comparison history for this transformation.")
    valid_chart = chart.dropna()
    if valid_chart.empty:
        raise DataValidationError("Not enough history for this indicator's transformation.")
    return {"value": float(valid_chart.iloc[-1]), "unit": unit, "secondary": secondary, "secondary_label": secondary_label, "date": valid_chart.index[-1], "chart": valid_chart}


def fetch_fred_series(series_id: str, api_key: str, observation_start: str = "2010-01-01") -> pd.Series:
    """Fetch FRED in memory. Errors deliberately omit URLs and key-bearing text."""
    if series_id not in FRED_SERIES:
        raise DataValidationError("Choose one of the configured macro series.")
    if not api_key.strip():
        raise DataValidationError("A FRED API key is required for live macro data.")
    query = urllib.parse.urlencode({"series_id": series_id, "api_key": api_key.strip(), "file_type": "json", "observation_start": observation_start})
    request = urllib.request.Request("https://api.stlouisfed.org/fred/series/observations?" + query, headers={"User-Agent": "MacroPortfolioRisk/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise DataValidationError(f"FRED returned HTTP {exc.code} for {series_id}. Check your key, connection and provider availability.") from None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        raise DataValidationError(f"Could not retrieve {series_id} from FRED. Check your connection and try again.") from None
    observations = payload.get("observations")
    if not observations:
        raise DataValidationError(f"FRED returned no observations for {series_id}.")
    frame = pd.DataFrame(observations)
    series = pd.Series(pd.to_numeric(frame["value"], errors="coerce").to_numpy(), index=pd.to_datetime(frame["date"]), name=series_id).dropna()
    series.attrs["retrieved_at"] = datetime.now(timezone.utc).isoformat()
    return series


def make_demo_macro() -> dict[str, pd.Series]:
    """Every value is simulated. Series labels describe the intended data feed."""
    rng = np.random.default_rng(402)
    monthly = pd.date_range("2015-01-01", "2026-10-01", freq="MS")
    weekly = pd.date_range("2015-01-03", "2026-10-24", freq="W-SAT")
    daily = pd.bdate_range("2015-01-01", "2026-10-30")
    months = len(monthly)
    result = {
        "PCEPILFE": pd.Series(100 * np.cumprod(1 + rng.normal(.0022, .0008, months)), index=monthly),
        "UNRATE": pd.Series(4.1 + .35 * np.sin(np.arange(months) / 10) + rng.normal(0, .05, months), index=monthly),
        "PAYEMS": pd.Series(140000 + np.cumsum(rng.normal(170, 70, months)), index=monthly),
        "ICSA": pd.Series(220000 + 20000 * np.sin(np.arange(len(weekly)) / 18) + rng.normal(0, 7000, len(weekly)), index=weekly),
        "INDPRO": pd.Series(100 * np.cumprod(1 + rng.normal(.0012, .003, months)), index=monthly),
        "PCEC96": pd.Series(12000 * np.cumprod(1 + rng.normal(.002, .0015, months)), index=monthly),
    }
    cycle = np.sin(np.arange(len(daily)) / 160)
    result["DGS10"] = pd.Series(3.6 + .6 * cycle + rng.normal(0, .025, len(daily)), index=daily)
    result["DGS2"] = pd.Series(3.8 + .8 * cycle + rng.normal(0, .035, len(daily)), index=daily)
    result["T10Y3M"] = pd.Series(-.15 + .5 * cycle + rng.normal(0, .04, len(daily)), index=daily)
    result["BAMLH0A0HYM2"] = pd.Series(3.7 - .5 * cycle + rng.normal(0, .04, len(daily)), index=daily)
    for key, value in result.items():
        value.name = key
    return result


COLORS = ["#2b8a9c", "#8c7dd8", "#cb8c38", "#609b71", "#cb6f78", "#748899"]


def plot_style(fig, height=340):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=15, t=25, b=15),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="sans-serif"), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0), colorway=COLORS)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)")
    return fig


def pct(value, digits=1):
    return "Unavailable" if not math.isfinite(float(value)) else f"{value:+.{digits}%}"


def number(value, digits=2):
    return "Unavailable" if not math.isfinite(float(value)) else f"{value:.{digits}f}"


def date_text(value):
    return "—" if value is None else pd.Timestamp(value).strftime("%b %d, %Y")


def csv_bytes(frame, index=True):
    return frame.to_csv(index=index).encode("utf-8")




def portfolio_risk_shares(asset_returns: pd.DataFrame, holdings: pd.DataFrame) -> pd.Series:
    """Euler variance shares for fixed initial weights; no clipping of negatives."""
    weights = holdings.set_index("ticker")["weight"].astype(float)
    covariance = asset_returns[weights.index].cov()
    marginal = covariance @ weights
    variance = float(weights @ marginal)
    if not math.isfinite(variance) or variance <= 1e-18:
        return pd.Series(float("nan"), index=weights.index, name="risk_share")
    return (weights * marginal / variance).rename("risk_share")


def demo_data():
    """Built into this file: no examples directory or network request needed."""
    return make_demo_data()


def main():
    st.set_page_config(page_title="Rithik Risk | Macro & Portfolio Analytics", page_icon="📊", layout="wide")
    st.markdown("""<style>
      .block-container { max-width: 1480px; padding-top: 2rem; }
      [data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.20); border-radius: 14px; padding: 16px 20px; }
      [data-testid="stMetricLabel"] { letter-spacing: .01em; }
      [data-baseweb="tab"] { font-weight: 600; }
      h1 { letter-spacing: -.035em; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.15); }
    </style>""", unsafe_allow_html=True)
    st.title("Rithik Risk")
    st.caption("A transparent research workspace for historical portfolio risk, economic context and explicit scenario assumptions.")

    with st.sidebar:
        st.caption("RITHIK RISK · MACRO & PORTFOLIO ANALYTICS")
        st.header("Research setup")
        source = st.radio("Price data", ["Synthetic demo", "Enter stocks manually", "Upload CSVs"])
        if source == "Enter stocks manually":
            prices, holdings, benchmark = manual_portfolio_inputs()
        elif source == "Upload CSVs":
            st.caption("Daily adjusted prices and fully invested, long-only weights in percent units.")
            uploaded_prices = st.file_uploader("Adjusted-price CSV", type=["csv"], key="prices_upload")
            uploaded_holdings = st.file_uploader("Holdings CSV", type=["csv"], key="holdings_upload")
            if uploaded_prices is None or uploaded_holdings is None:
                st.info("Upload both files to calculate your portfolio. The synthetic example is available in the selector.")
                demo_p, demo_h = demo_data()
                st.download_button("Price CSV example", csv_bytes(demo_p.round(6)), "demo-prices.csv", "text/csv")
                example_holdings = demo_h.copy()
                example_holdings["weight"] *= 100
                st.download_button("Holdings CSV example", csv_bytes(example_holdings, False), "demo-holdings.csv", "text/csv")
                st.stop()
            try:
                prices = read_prices_csv(uploaded_prices)
                holdings = read_holdings_csv(uploaded_holdings)
            except (DataValidationError, UnicodeDecodeError, OSError) as exc:
                st.error(str(exc))
                st.stop()
        else:
            try:
                prices, holdings = demo_data()
            except (DataValidationError, OSError) as exc:
                st.error(f"The supplied demo CSV is invalid: {exc}")
                st.stop()
        if source != "Enter stocks manually":
            tickers = prices.columns.tolist()
            benchmark = st.selectbox("Benchmark", tickers, index=tickers.index("SPY") if "SPY" in tickers else len(tickers) - 1, help="Comparison only. Changing this does not change your portfolio holdings.")
        lookback_label = st.selectbox("Lookback · return observations", ["126 · about 6 months", "252 · about 1 year", "756 · about 3 years", "All available"], index=1)
        lookback = None if lookback_label == "All available" else int(lookback_label.split(" ")[0])
        mode_label = st.selectbox("Portfolio construction", ["Constant weights · rebalance each observation", "Buy and hold · weights drift"])
        mode = "daily" if mode_label.startswith("Constant") else "buy_hold"
        rf = st.number_input("Annual risk-free assumption (%)", min_value=-95.0, max_value=100.0, value=0.0, step=0.25, help="Constant annual assumption used only in the Sharpe ratio. Converted to a compound daily rate.") / 100
        st.caption("No trading, optimization, automatic price filling or stock-return forecasts.")

    try:
        result = analyze_portfolio(prices, holdings, benchmark, lookback, mode, rf)
    except DataValidationError as exc:
        st.error(str(exc))
        st.stop()

    if source == "Synthetic demo":
        st.warning("SYNTHETIC DEMO · Every example price, holding weight and derived risk statistic is simulated. These are not historical company results or actual PMG holdings.")
    elif source == "Enter stocks manually":
        st.info("MANUAL PORTFOLIO · Yahoo Finance daily adjusted prices with your entered starting weights. This is a hypothetical historical portfolio, not a transaction-based account record. FRED macro data is controlled separately in the Macro tab.")
    else:
        st.info("UPLOADED DATA · Results reconstruct the supplied weights over the selected history. They are not an actual portfolio track record unless the inputs and rebalance assumptions match it.")

    st.caption(f"{date_text(result.prices.index[0])} to {date_text(result.prices.index[-1])} · {result.periods:,} aligned returns · Benchmark: {benchmark} · {len(holdings)} holdings")
    for message in result.warnings:
        st.warning(message)

    portfolio_tab, macro_tab, connections_tab, scenarios_tab, notes_tab = st.tabs(["Portfolio", "Macro", "Connections", "Scenarios", "Notes"])

    with portfolio_tab:
        cells = st.columns(3)
        cells[0].metric("Total return", pct(result.metrics["total_return"]), help="Compound change over the selected window, before fees, taxes and trading costs.")
        cells[1].metric("Annualized volatility", f'{result.metrics["annualized_volatility"]:.1%}', help="Sample standard deviation of portfolio returns × square root of 252.")
        cells[2].metric(f"Beta vs {benchmark}", number(result.metrics["beta"]), help="Sample covariance with the benchmark divided by the benchmark's sample variance. Historical association, not a causal forecast.")
        cells = st.columns(3)
        cells[0].metric("Maximum drawdown", f'{result.metrics["maximum_drawdown"]:.1%}', help="Largest fall from a prior high, including the initial starting wealth.")
        cells[1].metric("Annualized compound return", pct(result.metrics["annualized_return"]), help="Ending wealth^(252 / number of returns) − 1; a 252-observation CAGR convention.")
        cells[2].metric("Sharpe ratio", number(result.metrics["sharpe"]), help="Mean daily return in excess of the converted risk-free assumption divided by sample daily volatility, × sqrt(252).")
        st.subheader("Growth of 100")
        equity = result.wealth.mul(100).rename(columns={"Benchmark": f"Benchmark ({benchmark})"}).rename_axis("Date").reset_index()
        st.plotly_chart(plot_style(px.line(equity, x="Date", y=["Portfolio", f"Benchmark ({benchmark})"], labels={"value": "Indexed wealth", "variable": "Series"})), width="stretch")
        left, right = st.columns([1.6, 1])
        with left:
            st.subheader("Portfolio drawdown")
            drawdown = result.drawdown["series"].mul(100).rename("Drawdown (%)").rename_axis("Date").reset_index()
            st.plotly_chart(plot_style(px.area(drawdown, x="Date", y="Drawdown (%)", color_discrete_sequence=["#cb6f78"]), 260), width="stretch")
        with right:
            st.subheader("Largest drawdown interval")
            st.write(f'**Prior peak:** {date_text(result.drawdown["peak"])}')
            st.write(f'**Trough:** {date_text(result.drawdown["trough"])}')
            recovery = result.drawdown["recovery"]
            st.write(f'**Recovery:** {date_text(recovery) if recovery is not None else "Not recovered within the selected window" if result.metrics["maximum_drawdown"] < 0 else "No drawdown"}')
            st.caption("Recovery means wealth first regained the prior peak. It may occur after the selected window, which this calculation cannot observe.")
        allocation, risk_share = st.columns(2)
        with allocation:
            st.subheader("Starting allocation")
            allocation_chart = px.pie(holdings, names="ticker", values="weight", hole=0.65, color_discrete_sequence=COLORS)
            allocation_chart.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{percent}<extra></extra>")
            st.plotly_chart(plot_style(allocation_chart, 300), width="stretch")
        with risk_share:
            st.subheader("What drives the risk?")
            contributions = portfolio_risk_shares(result.asset_returns, holdings)
            if contributions.notna().any():
                chart = contributions.mul(100).rename("Share of variance (%)").rename_axis("Ticker").reset_index()
                st.plotly_chart(plot_style(px.bar(chart, x="Ticker", y="Share of variance (%)", color_discrete_sequence=[COLORS[1]]), 300), width="stretch")
            else:
                st.info("Risk shares are unavailable because modeled variance is zero.")
            st.caption("Sample covariance with constant starting weights. Shares sum to 100% when variance is positive; negative shares can reflect diversification. In buy-and-hold mode, this is a constant-weight reference.")
        st.subheader("Holding-level risk")
        shown = holdings.set_index("ticker")[["name", "sector"]].join(result.asset_metrics)
        shown["risk_share"] = portfolio_risk_shares(result.asset_returns, holdings)
        if mode == "buy_hold":
            starting_weights = holdings.set_index("ticker")["weight"]
            relative_prices = result.prices[starting_weights.index].iloc[-1] / result.prices[starting_weights.index].iloc[0]
            ending_values = relative_prices * starting_weights
            shown["ending_weight"] = ending_values / ending_values.sum()
        formats = {column: "{:.1%}" for column in ["weight", "total_return", "annualized_return", "annualized_volatility", "maximum_drawdown"]}
        formats.update({"risk_share": "{:.1%}", "ending_weight": "{:.1%}"})
        formats.update({"beta": "{:.2f}", "sharpe": "{:.2f}", "benchmark_correlation": "{:.2f}"})
        st.dataframe(shown.style.format(formats, na_rep="Unavailable"), width="stretch")
        st.caption("Weights are beginning weights in buy-and-hold mode. The metrics table shows individual-security statistics from the same aligned dates.")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download risk metrics", csv_bytes(shown), "portfolio-risk-metrics.csv", "text/csv")
        d2.download_button("Download wealth history", csv_bytes(result.wealth), "portfolio-wealth-history.csv", "text/csv")
        d3.download_button("Download aligned prices", csv_bytes(result.prices), "aligned-prices.csv", "text/csv")
        with st.expander("Methodology and input limitations"):
            st.markdown("""
    - **Daily returns:** adjusted price today / previous retained adjusted price − 1. The app never forward-fills prices. Common complete rows are used for every holding and the benchmark.
    - **Constant weights:** weighted asset returns each retained observation, with frictionless rebalancing. **Buy and hold:** shares implied by starting weights are held fixed; weights drift.
    - **Volatility and beta:** sample estimates (`ddof=1`), with 252 observations per year. A zero-variance benchmark makes beta unavailable.
    - **Drawdown:** wealth / running maximum wealth − 1, with initial wealth included. Annualized compound return uses the same 252-observation convention.
    - **Sharpe:** a user-supplied constant annual risk-free rate is compounded to a daily rate. No Treasury observation is automatically treated as the historical cash return.
    - Upload daily, consistently adjusted prices in one base currency and confirm how your provider treats dividends and splits. Missing dates can produce multi-session returns and approximate annualization; warnings are visible.
    - Results exclude fees, slippage, taxes, cash flows and changing holdings. Today's weights applied to past prices form a hypothetical historical portfolio. An index or ETF benchmark should fit the investment mandate.
    """)

    with macro_tab:
        st.subheader("Economic context")
        st.caption("Observation dates are economic reference periods, not publication dates. Sources have different release schedules and revision policies.")
        macro_mode = st.radio("Macro source", ["Synthetic illustration", "Live FRED"], horizontal=True)
        if macro_mode == "Synthetic illustration":
            st.warning("SYNTHETIC MACRO · These values illustrate the interface and transformations; they are not current economic readings.")
            macro_data = make_demo_macro()
            failures = {}
        else:
            secret_key = os.environ.get("FRED_API_KEY", "")
            key = st.text_input("FRED API key", type="password", value=secret_key, help="Used in this running session or read from the optional FRED_API_KEY environment variable. The app does not write credentials to files, notes or downloads.")
            st.caption("[Get your own FRED key](https://fred.stlouisfed.org/docs/api/api_key.html). A live request failure is shown explicitly; no synthetic replacement is inserted.")
            if st.button("Refresh the 10 FRED series", disabled=not bool(key.strip())):
                fresh, failures = {}, {}
                with st.spinner("Retrieving official series through FRED…"):
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        futures = {pool.submit(fetch_fred_series, series_id, key): series_id for series_id in FRED_SERIES}
                        for future in as_completed(futures):
                            series_id = futures[future]
                            try:
                                fresh[series_id] = future.result()
                            except Exception as exc:
                                failures[series_id] = str(exc) if isinstance(exc, DataValidationError) else "Could not retrieve this series. Check the connection and retry."
                st.session_state["fred_data"] = fresh
                st.session_state["fred_failures"] = failures
                st.session_state["fred_retrieved"] = datetime.now(timezone.utc).isoformat()
            macro_data = st.session_state.get("fred_data", {})
            failures = st.session_state.get("fred_failures", {})
            if macro_data:
                st.caption("Last retrieval: " + st.session_state.get("fred_retrieved", "unknown") + ". Retrieval time is not the publication time.")
            elif not failures:
                st.info("Enter your key and refresh to load live macro data. Price analysis remains available without a FRED key.")
            for series_id, error in failures.items():
                st.error(f"{series_id}: {error}")
        category = st.selectbox("Indicator group", ["All indicators", "Growth & labor", "Inflation", "Rates & credit"])
        visible = [sid for sid, meta in FRED_SERIES.items() if category == "All indicators" or meta["category"] == category]
        columns = st.columns(2)
        summary_rows = []
        for position, sid in enumerate(visible):
            if sid not in macro_data:
                continue
            meta = FRED_SERIES[sid]
            with columns[position % 2]:
                try:
                    summary = fred_summary(sid, macro_data[sid])
                except DataValidationError as exc:
                    st.warning(f"{sid}: {exc}")
                    continue
                st.markdown(f'**{meta["name"]}**')
                st.metric(summary["unit"], number(summary["value"], 2))
                st.caption(f'{summary["secondary_label"]}: {number(summary["secondary"], 2)} · Observation: {date_text(summary["date"])}')
                history = summary["chart"].tail(504 if meta["frequency"] == "Daily" else 104 if meta["frequency"] == "Weekly" else 60)
                graph_data = history.rename(summary["unit"]).rename_axis("Date").reset_index()
                st.plotly_chart(plot_style(px.line(graph_data, x="Date", y=summary["unit"]), 235), width="stretch", key="macro_chart_" + sid)
                st.caption(f'[{sid} · {meta["provider"]}](https://fred.stlouisfed.org/series/{sid}) · {meta["frequency"]} · Original units: {meta["units"]}')
                summary_rows.append({"series_id": sid, "name": meta["name"], "observation_date": str(summary["date"].date()), "value": summary["value"], "units": summary["unit"], "comparison_value": summary["secondary"], "comparison_definition": summary["secondary_label"], "source_mode": macro_mode})
        if summary_rows:
            st.download_button("Download displayed macro snapshot", csv_bytes(pd.DataFrame(summary_rows), False), "macro-snapshot.csv", "text/csv")
        with st.expander("Transformations, timing and usage rights"):
            st.markdown("""
    Core PCE, industrial production and real personal consumption display **12-month percentage changes**. Their secondary values are **3-month annualized changes**: `100 × ((level / level three months ago)^4 − 1)`.

    Payrolls display the **monthly change in thousands of jobs**; claims display a **four-observation moving average in thousands**. Treasury yields are percentages. The yield-curve spread and high-yield spread display **basis points**, converting FRED's percentage/percentage-point units by multiplying by 100. A 21-observation comparison is approximately one trading month, not necessarily a calendar month.

    The live feed requests the latest available vintage. Revisions, release lags and market expectations matter. Do not use this snapshot as a point-in-time backtest. Use ALFRED vintages and actual availability dates for that work. Macro levels are not consensus surprises or automatic trading signals.

    FRED provides access to sources with different usage terms. The ICE high-yield series can carry restrictions on redistribution. Review its series notes and provider terms before publishing or distributing fetched data.
    """)

    with connections_tab:
        st.subheader("Return relationships")
        st.caption("Pairwise Pearson correlations of returns over the same aligned window. Similar movement can reduce diversification, but it does not establish causality.")
        include_benchmark = st.checkbox("Include benchmark in the network", value=False)
        labels = holdings["ticker"].tolist()
        if include_benchmark and benchmark not in labels:
            labels.append(benchmark)
        matrix = result.correlations.loc[labels, labels]
        heatmap = go.Figure(go.Heatmap(z=matrix.to_numpy(), x=labels, y=labels, zmin=-1, zmax=1, zmid=0, colorscale="RdBu", reversescale=True, text=np.round(matrix.to_numpy(), 2), texttemplate="%{text}", hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>"))
        heatmap.update_layout(height=max(310, len(labels) * 36), margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(heatmap, width="stretch")
        threshold = st.slider("Show connections with absolute correlation at least", 0.0, 1.0, 0.60, 0.05)
        edges = correlation_edges(matrix, threshold)
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        positions = {label: (float(np.cos(angle)), float(np.sin(angle))) for label, angle in zip(labels, angles)}
        graph = go.Figure()
        for edge in edges.itertuples():
            x1, y1 = positions[edge.asset_a]
            x2, y2 = positions[edge.asset_b]
            graph.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines", line=dict(width=1 + 3 * abs(edge.correlation), color="#2b8a9c" if edge.correlation >= 0 else "#cb6f78", dash="solid" if edge.correlation >= 0 else "dash"), hoverinfo="text", text=f"{edge.asset_a} / {edge.asset_b}: {edge.correlation:.2f}", showlegend=False))
        graph.add_trace(go.Scatter(x=[positions[k][0] for k in labels], y=[positions[k][1] for k in labels], mode="markers+text", text=labels, textposition="top center", marker=dict(size=20, color="#8c7dd8"), hoverinfo="text", showlegend=False))
        graph.update_layout(height=330, xaxis=dict(visible=False, range=[-1.3, 1.3]), yaxis=dict(visible=False, range=[-1.3, 1.3], scaleanchor="x", scaleratio=1), margin=dict(l=15, r=15, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(graph, width="stretch")
        st.caption("Teal solid lines: positive relationships. Rose dashed lines: negative relationships. Thickness encodes absolute correlation. Node positions are a display layout; distance has no statistical meaning.")
        if edges.empty:
            st.info("No asset pairs meet this threshold in the selected window.")
        else:
            st.dataframe(edges.style.format({"correlation": "{:.2f}"}), width="stretch", hide_index=True)
        st.download_button("Download correlations", csv_bytes(matrix), "return-correlations.csv", "text/csv")

    with scenarios_tab:
        st.subheader("Explicit scenario assumptions")
        st.caption("These are illustrative sensitivity calculations. They do not assign probabilities or predict future returns.")
        manual_tab, beta_tab, financing_tab = st.tabs(["Holding shocks", "Market-beta approximation", "Debt repricing"])
        with manual_tab:
            initial = st.number_input("Illustrative starting portfolio value ($)", min_value=0.0, value=100000.0, step=1000.0)
            shocks = holdings[["ticker", "weight"]].copy()
            shocks.columns = ["Ticker", "Weight (%)"]
            shocks["Weight (%)"] *= 100
            shocks["Assumed return (%)"] = 0.0
            edited = st.data_editor(shocks, hide_index=True, disabled=["Ticker", "Weight (%)"], column_config={"Assumed return (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=1000.0, step=1.0)}, width="stretch", key="shock_editor_" + "_".join(holdings["ticker"].tolist()))
            try:
                shock_map = dict(zip(edited["Ticker"], edited["Assumed return (%)"]))
                scenario = weighted_scenario(holdings, shock_map)
                impact = float(scenario["contribution_percentage_points"].sum()) / 100
                left, middle, right = st.columns(3)
                left.metric("Weighted portfolio change", pct(impact))
                middle.metric("Dollar change", f"${initial * impact:,.0f}")
                right.metric("Illustrative ending value", f"${initial * (1 + impact):,.0f}")
                st.dataframe(scenario, hide_index=True, width="stretch")
                st.download_button("Download holding scenario", csv_bytes(scenario, False), "holding-shock-scenario.csv", "text/csv")
            except (DataValidationError, ValueError, TypeError) as exc:
                st.error(str(exc))
            st.caption("One-period weighted price/total-return shocks using the entered weights; no rebalancing within the scenario. Company-specific assumptions should come from research, not the macro chart alone.")
        with beta_tab:
            market_shock = st.slider("Assumed benchmark move (%)", -40.0, 40.0, -10.0, 1.0)
            approximate = result.asset_metrics[["weight", "beta"]].copy()
            approximate["assumed_market_move_percent"] = market_shock
            approximate["beta_implied_move_percent"] = approximate["beta"] * market_shock
            portfolio_approx = result.metrics["beta"] * market_shock
            st.metric("Portfolio beta × assumed market move", f"{portfolio_approx:+.1f}%" if math.isfinite(portfolio_approx) else "Unavailable")
            st.dataframe(approximate.style.format({"weight": "{:.1%}", "beta": "{:.2f}", "beta_implied_move_percent": "{:+.1f}"}, na_rep="Unavailable"), width="stretch")
            st.warning("This is a first-order historical-beta assumption. It excludes intercepts, idiosyncratic shocks, nonlinear behavior and changing correlations. It is not a modeled macro scenario, a maximum-loss estimate or a forecast. Buy-and-hold portfolio beta describes its historical return path, not today's asset-weighted sensitivity.")
        with financing_tab:
            a, b = st.columns(2)
            principal_m = a.number_input("Debt actually repricing ($ millions)", min_value=0.0, value=400.0, step=25.0)
            change_bps = b.number_input("Change in borrowing cost (basis points)", value=150.0, step=25.0)
            months = a.number_input("Months exposed in the modeled year", min_value=0.0, max_value=12.0, value=12.0, step=1.0)
            tax_rate = b.number_input("Assumed effective tax rate (%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
            shares_m = a.number_input("Diluted shares (millions)", min_value=0.001, value=100.0, step=10.0)
            financing = financing_scenario(principal_m * 1e6, change_bps, months, tax_rate / 100, shares_m * 1e6)
            a1, a2, a3 = st.columns(3)
            a1.metric("Incremental interest", f'${financing["incremental_interest"] / 1e6:+.2f}m')
            a2.metric("Net-income change", f'${financing["net_income_change"] / 1e6:+.2f}m')
            a3.metric("EPS change", f'${financing["eps_change"]:+.3f}')
            st.caption("Assumes the stated balance reprices for the stated months, full interest deductibility, no hedges, and no other changes. A Treasury shock need not equal the change in company borrowing costs. Do not also subtract this interest from an unlevered FCFF model that accounts for financing through WACC.")

    with notes_tab:
        st.subheader("Analyst decision journal")
        st.caption("Saved notes stay in this Streamlit session and may be lost when you reload or disconnect. Download the JSON archive to preserve them and upload it later to restore them. Notes and API keys are never included together.")
        if "analyst_notes" not in st.session_state:
            st.session_state["analyst_notes"] = []
        title = st.text_input("Note title", placeholder="Rates review · what changed and what needs checking")
        body = st.text_area("Research note", height=180, placeholder="Development:\nAffected holdings / thesis assumptions:\nEvidence and source links:\nScenario assumptions:\nDecision / next action:\nNext review:")
        if st.button("Save note to this session", disabled=not bool(title.strip() and body.strip())):
            st.session_state["analyst_notes"].append({"id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(), "title": title.strip(), "body": body.strip(), "price_source": source, "benchmark": benchmark, "portfolio_mode": mode, "window_start": str(result.prices.index[0].date()), "window_end": str(result.prices.index[-1].date())})
            st.success("Note saved in this session. Download the archive to preserve it after the session ends.")
        notes = st.session_state["analyst_notes"]
        if notes:
            export = {"schema_version": 1, "notes": notes}
            markdown = "# Analyst risk journal\n\n" + "\n\n---\n\n".join(f'## {note["title"]}\n\n{note["created_at"]} · {note.get("price_source", "unspecified")}\n\n{note["body"]}' for note in notes)
            x, y = st.columns(2)
            x.download_button("Download journal JSON", json.dumps(export, indent=2), "analyst-journal.json", "application/json")
            y.download_button("Download journal Markdown", markdown, "analyst-journal.md", "text/markdown")
            for note in reversed(notes):
                with st.expander(note["title"] + " · " + note["created_at"][:10]):
                    st.caption(f'{note.get("price_source", "unspecified")} · {note.get("benchmark", "")} · {note.get("window_start", "")} to {note.get("window_end", "")}')
                    st.text(note["body"])
        restore = st.file_uploader("Restore a journal JSON archive", type=["json"], key="restore_notes")
        if restore is not None and st.button("Import notes, skipping duplicate IDs"):
            try:
                payload = json.loads(restore.getvalue())
                if not isinstance(payload, dict):
                    raise ValueError("Unsupported journal format.")
                imported = payload["notes"]
                if payload.get("schema_version") != 1 or not isinstance(imported, list) or len(imported) > 1000:
                    raise ValueError("Unsupported journal format or too many notes.")
                existing_ids = {note["id"] for note in st.session_state["analyst_notes"]}
                valid = []
                for note in imported:
                    if not isinstance(note, dict) or not all(isinstance(note.get(field), str) for field in ["id", "title", "body", "created_at"]):
                        raise ValueError("Every note needs string id, title, body and created_at fields.")
                    if len(note["body"]) > 100000:
                        raise ValueError("A note is too large.")
                    if note["id"] not in existing_ids:
                        valid.append({key: value for key, value in note.items() if key in {"id", "title", "body", "created_at", "price_source", "benchmark", "portfolio_mode", "window_start", "window_end"}})
                        existing_ids.add(note["id"])
                st.session_state["analyst_notes"].extend(valid)
                st.success(f"Imported {len(valid)} notes. They will appear on the next interaction.")
            except (KeyError, ValueError, TypeError, UnicodeDecodeError):
                st.error("Could not import this archive. Use a valid journal JSON file exported by this app.")

    st.caption("Research tool · Demo prices are simulated. Manual entry uses Yahoo Finance adjusted daily prices; CSV mode uses your supplied prices. The Macro tab separately identifies simulated or live FRED observations. Downloaded market data is provided for personal research; review provider terms before redistribution.")


def launch():
    """Support both `python app.py` and the standard Streamlit CLI."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is not None:
        main()
        return
    port_text = os.environ.get("PORT") or "8501"
    try:
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        raise SystemExit("PORT must be a number from 1 to 65535.") from None
    command = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        "--server.address=0.0.0.0", f"--server.port={port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
        "--theme.base=light", "--theme.primaryColor=#6D4EC7",
        "--theme.backgroundColor=#F8FAFC", "--theme.secondaryBackgroundColor=#FFFFFF",
    ]
    print(f"Starting Rithik Risk on port {port}. Open that port in Codespaces, or visit http://localhost:{port} locally.", flush=True)
    try:
        raise SystemExit(subprocess.call(command))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    launch()
