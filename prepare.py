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

# ---- Fixed universe (do not modify in agent loop). 5 highly liquid names. ----
UNIVERSE = ["SPY", "QQQ", "NVDA", "AAPL", "TSLA"]

# ---- Fixed window. yfinance free tier exposes ~30d of 1-min bars. ----
DAYS = 28                    # calendar days fetched
EVAL_FRACTION = 0.30         # last 30% used as held-out eval
SEED = 0                     # for any deterministic shuffles in evaluator

# ---- Fixed economic constants for the simulator (the broker is the evaluator) ----
STARTING_CASH_USD = 50_000.0
NOTIONAL_PER_SYMBOL_USD = 1_000.0
FEE_PER_TRADE_USD = 1.0      # IBKR Pro Fixed minimum
SLIPPAGE_BPS = 2.0           # per side
MIN_TRADE_NOTIONAL_USD = 100.0

# ---- Multi-seed eval (no wall-time budget — let each seed finish on this hardware) ----
N_SEEDS = 3                  # multi-seed runs to combat eval noise


# ----------------------------------------------------------------------
# Data download (cached)
# ----------------------------------------------------------------------

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


def fetch_bars(symbol: str, force: bool = False) -> pd.DataFrame:
    """Download 1-min bars for a symbol via yfinance, cached on disk.

    yfinance limits 1m data to 7-day windows per request; we chunk DAYS / 7 calls.
    """
    p = _cache_path(symbol)
    if p.exists() and not force:
        return pd.read_parquet(p)

    import yfinance as yf
    from datetime import timedelta
    print(f"[prepare] downloading {symbol} 1m × {DAYS}d via yfinance (chunked) …", flush=True)
    end = datetime.now(timezone.utc)
    chunks: list[pd.DataFrame] = []
    cur_end = end
    days_left = DAYS
    while days_left > 0:
        chunk_days = min(7, days_left)
        chunk_start = cur_end - timedelta(days=chunk_days)
        df = yf.download(
            symbol,
            start=chunk_start.strftime("%Y-%m-%d"),
            end=cur_end.strftime("%Y-%m-%d"),
            interval="1m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["timestamp"]).dropna()
    df.to_parquet(p, index=False)
    print(f"[prepare] {symbol}: {len(df):,} bars → {p}")
    return df


def prepare_all(force: bool = False) -> dict[str, pd.DataFrame]:
    """Ensure all universe data is cached. Returns {symbol: bars_df}."""
    return {s: fetch_bars(s, force=force) for s in UNIVERSE}


def split(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/eval split. Eval is the LAST EVAL_FRACTION of bars."""
    n = len(bars)
    cut = int(n * (1.0 - EVAL_FRACTION))
    return bars.iloc[:cut].reset_index(drop=True), bars.iloc[cut:].reset_index(drop=True)


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
