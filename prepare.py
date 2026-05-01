"""Frozen data prep + utilities. AGENT MUST NOT MODIFY THIS FILE.

Downloads 1-minute OHLCV bars for a small fixed universe via yfinance into a
local cache, then exposes a deterministic train/eval split + paper-broker
simulator + metrics that experiment.py and evaluator.py both rely on.

The cache, the universe, the split, the broker, and the metrics are
intentionally frozen so every experiment is comparable across runs.

FEATURE ENGINEERING IS NOT FROZEN — it lives in experiment.py so the agent
can iterate on which features to compute and feed the model.
"""
from __future__ import annotations
import os
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(os.path.expanduser("~/.cache/trading-autoresearch"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---- v6: top-20 by liquidity (5 ETFs + 15 mega/large-caps). ----
UNIVERSE = [
    # broad-market & sector ETFs
    "SPY", "QQQ", "IWM", "EEM", "XLF",
    # mega-cap tech / consumer
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC", "NFLX",
    # high-volume large-caps
    "BAC", "F", "COIN", "PLTR", "NIO",
]

# ---- v6: extended to 6 years for much longer eval window. ----
DAYS = 2190                  # 6 years of 1-min bars (Alpaca IEX serves back to ~2016)
# v7: day-based eval window (90 days). User directive: train on 6 years of
# bars, evaluate on the LAST 90 calendar days. Switching from a fraction-based
# split (which was 0.20 → ~438 days) gives a tighter, fixed eval horizon that
# matches a typical "last quarter" out-of-sample window.
EVAL_DAYS = 90
SEED = 0                     # for any deterministic shuffles in evaluator

# ---- Fixed economic constants for the simulator (the broker is the evaluator) ----
STARTING_CASH_USD = 50_000.0
NOTIONAL_PER_SYMBOL_USD = 1_000.0
FEE_PER_TRADE_USD = 1.0      # IBKR Pro Fixed minimum
SLIPPAGE_BPS = 2.0           # per side
MIN_TRADE_NOTIONAL_USD = 100.0

# ---- Multi-seed eval (no wall-time budget — let each seed finish on this hardware) ----
N_SEEDS = 3                  # 3 seeds for fast iteration on Alpaca year-of-data


# ----------------------------------------------------------------------
# Data download (cached)
# ----------------------------------------------------------------------

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


# Alpaca symbols cannot include indices like "^VIX". Use yfinance fallback for those.
_ALPACA_INVALID = {"^VIX", "^DXY", "^GSPC"}


def _fetch_via_yfinance(symbol: str, days: int) -> pd.DataFrame:
    """Fallback for indices Alpaca doesn't serve. Capped at ~30 days by yfinance."""
    import yfinance as yf
    from datetime import timedelta
    days = min(days, 28)
    print(f"[prepare] {symbol} (yfinance fallback, max ~30d): downloading {days}d …", flush=True)
    end = datetime.now(timezone.utc)
    chunks = []
    cur_end = end
    days_left = days
    while days_left > 0:
        chunk_days = min(7, days_left)
        chunk_start = cur_end - timedelta(days=chunk_days)
        df = yf.download(symbol, start=chunk_start.strftime("%Y-%m-%d"),
                         end=cur_end.strftime("%Y-%m-%d"), interval="1m",
                         auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            chunks.append(df)
        cur_end = chunk_start
        days_left -= chunk_days
    if not chunks:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    df = pd.concat(chunks).sort_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns=str.lower).reset_index()
    df = df.rename(columns={"datetime": "timestamp", "Datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["timestamp"]).dropna()


def _fetch_via_alpaca(symbol: str, days: int) -> pd.DataFrame:
    """Alpaca free IEX feed — 1-min bars, supports years of history."""
    from datetime import timedelta
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not (key and secret):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed

    client = StockHistoricalDataClient(key, secret)
    end = datetime.now(timezone.utc) - timedelta(minutes=16)   # IEX needs >15min lag on free
    start = end - timedelta(days=days)
    print(f"[prepare] {symbol} (Alpaca IEX): downloading {days}d 1m bars …", flush=True)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start, end=end,
        feed=DataFeed.IEX,
        adjustment="raw",
    )
    resp = client.get_stock_bars(req)
    df = resp.df
    if df is None or df.empty:
        raise RuntimeError(f"Alpaca returned no data for {symbol}")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["timestamp"]).dropna()


def fetch_bars(symbol: str, force: bool = False) -> pd.DataFrame:
    """Download 1-min bars, cached on disk. Alpaca primary, yfinance for indices."""
    p = _cache_path(symbol)
    if p.exists() and not force:
        return pd.read_parquet(p)
    if symbol in _ALPACA_INVALID:
        df = _fetch_via_yfinance(symbol, DAYS)
    else:
        try:
            df = _fetch_via_alpaca(symbol, DAYS)
        except Exception as e:
            print(f"[prepare] {symbol}: Alpaca failed ({e}); falling back to yfinance", flush=True)
            df = _fetch_via_yfinance(symbol, DAYS)
    df.to_parquet(p, index=False)
    print(f"[prepare] {symbol}: {len(df):,} bars → {p}")
    return df


def prepare_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Ensure all universe data is cached. Returns {symbol: bars_df}."""
    return {s: fetch_bars(s, force=force) for s in UNIVERSE}


def split(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/eval split. Eval = last EVAL_DAYS calendar days.

    Day-based slicing (rather than a fixed fraction) ensures the eval window
    has a stable temporal length regardless of how much training data is
    available — important for cross-experiment comparability.
    """
    if len(bars) == 0:
        return bars, bars
    bars = bars.sort_values("timestamp").reset_index(drop=True)
    last_ts = bars["timestamp"].iloc[-1]
    cutoff = last_ts - pd.Timedelta(days=EVAL_DAYS)
    train = bars[bars["timestamp"] < cutoff].reset_index(drop=True)
    eval_ = bars[bars["timestamp"] >= cutoff].reset_index(drop=True)
    return train, eval_


# ----------------------------------------------------------------------
# Frozen broker — DO NOT modify. This is the ground-truth evaluator.
# ----------------------------------------------------------------------

@dataclass
class _Position:
    qty: float = 0.0
    last_price: float = 0.0


class PaperBroker:
    """IBKR-style paper broker: per-trade $1 fee, 2bps slippage, capital cap."""

    def __init__(self) -> None:
        self.cash = STARTING_CASH_USD
        self.positions: dict[str, _Position] = {}
        self.n_trades = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        # (ts, symbol, side) — side is "BUY" or "SELL" based on sign(delta)
        self.trades: list[tuple[pd.Timestamp, str, str]] = []

    def _pos(self, sym: str) -> _Position:
        if sym not in self.positions:
            self.positions[sym] = _Position()
        return self.positions[sym]

    def update(self, symbol: str, price: float, ts: pd.Timestamp, target_frac: float) -> dict:
        """target_frac in [-1, 1]. Position notional = NOTIONAL_PER_SYMBOL_USD * target_frac."""
        p = self._pos(symbol)
        p.last_price = price
        target_qty = (target_frac * NOTIONAL_PER_SYMBOL_USD) / max(price, 1e-9)
        delta = target_qty - p.qty
        notional_delta = abs(delta) * price
        result = {"delta": 0.0, "fee": 0.0, "skipped": None}

        if notional_delta < MIN_TRADE_NOTIONAL_USD:
            return result

        # capital check
        cash_after = self.cash - delta * price - FEE_PER_TRADE_USD
        new_short = max(0.0, -(p.qty + delta)) * price
        old_short = max(0.0, -p.qty) * p.last_price
        if (cash_after - 0.5 * (new_short - old_short)) < 0:
            result["skipped"] = "no_capital"
            return result

        side = 1.0 if delta > 0 else -1.0
        fill_price = price * (1.0 + side * SLIPPAGE_BPS * 1e-4)
        slip_cost = abs(delta) * abs(fill_price - price)
        self.cash -= delta * fill_price + FEE_PER_TRADE_USD
        p.qty = target_qty
        self.n_trades += 1
        self.total_fees += FEE_PER_TRADE_USD
        self.total_slippage += slip_cost
        self.trades.append((ts, symbol, "BUY" if delta > 0 else "SELL"))
        result["delta"] = delta
        result["fee"] = FEE_PER_TRADE_USD
        return result

    def equity(self, prices: dict[str, float]) -> float:
        e = self.cash
        for sym, p in self.positions.items():
            e += p.qty * prices.get(sym, p.last_price)
        return e

    def mark_to_market(self, ts: pd.Timestamp, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.equity_curve.append((ts, eq))
        return eq


# ----------------------------------------------------------------------
# Metrics — also frozen. The contract between experiment and evaluator.
# ----------------------------------------------------------------------

def sharpe_ratio(equity_curve: list[tuple[pd.Timestamp, float]],
                 periods_per_year: float = 252 * 6.5 * 60) -> float:
    """Per-bar Sharpe annualized (1-min bars → 252 trading days × 6.5h × 60min/year)."""
    eq = np.array([v for _, v in equity_curve], dtype=np.float64)
    if eq.size < 3:
        return 0.0
    rets = np.diff(eq) / eq[:-1]
    if rets.std(ddof=1) == 0:
        return 0.0
    return float(rets.mean() / rets.std(ddof=1) * math.sqrt(periods_per_year))


def max_drawdown_pct(equity_curve: list[tuple[pd.Timestamp, float]]) -> float:
    eq = np.array([v for _, v in equity_curve], dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def bootstrap_sharpe_ci(equity_curve, n: int = 200, q: float = 0.05) -> tuple[float, float]:
    """Bootstrap a (q, 1-q) confidence interval on the Sharpe."""
    eq = np.array([v for _, v in equity_curve], dtype=np.float64)
    if eq.size < 10:
        return 0.0, 0.0
    rets = np.diff(eq) / eq[:-1]
    rng = np.random.default_rng(SEED)
    sharpes = []
    periods_per_year = 252 * 6.5 * 60
    for _ in range(n):
        sample = rng.choice(rets, size=rets.size, replace=True)
        if sample.std(ddof=1) > 0:
            sharpes.append(sample.mean() / sample.std(ddof=1) * math.sqrt(periods_per_year))
    if not sharpes:
        return 0.0, 0.0
    arr = np.array(sharpes)
    return float(np.quantile(arr, q)), float(np.quantile(arr, 1 - q))


if __name__ == "__main__":
    prepare_all()
    print(f"\n[prepare] universe={UNIVERSE}  cache={CACHE_DIR}")
