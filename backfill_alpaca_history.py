"""Backfill older Alpaca stock bars into the local 1-minute parquet cache.

The frozen prepare.py path downloads a fixed recent window. This script is for
explicit historical extensions: choose symbols, start/end dates, and feed, then
merge the returned bars into ~/.cache/trading-autoresearch/{SYMBOL}_1m.parquet.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from prepare import CACHE_DIR, UNIVERSE
from experiment import CONTEXT_SYMBOLS, EXTENDED_UNIVERSE, HOLDOUT_UNIVERSE
from top500_universe import load_top500_symbols


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


def _env(name: str, fallback: str = "") -> str:
    return os.environ.get(name) or os.environ.get(f"APCA_{name}") or fallback


def _parse_dt(text: str) -> datetime:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        raw = [s.strip().upper() for part in args.symbols for s in part.split(",") if s.strip()]
    else:
        raw = []
        if args.top500:
            raw.extend(load_top500_symbols())
        if args.cached_all:
            raw.extend(sorted(p.name.removesuffix("_1m.parquet") for p in CACHE_DIR.glob("*_1m.parquet")))
        if args.default_universe:
            raw.extend(UNIVERSE)
            raw.extend(CONTEXT_SYMBOLS)
            raw.extend(EXTENDED_UNIVERSE)
            raw.extend(HOLDOUT_UNIVERSE)
    seen: set[str] = set()
    out = [s for s in raw if not (s in seen or seen.add(s))]
    if args.start_at > 1:
        out = out[args.start_at - 1:]
    if args.limit > 0:
        out = out[: args.limit]
    return out


def _client() -> object:
    load_dotenv(Path(__file__).parent / ".env")
    key = _env("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID", "")
    secret = _env("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("set ALPACA_API_KEY/ALPACA_SECRET_KEY or APCA_API_KEY_ID/APCA_API_SECRET_KEY")
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(key, secret)


def _fetch(client: object, symbol: str, start: datetime, end: datetime, feed: str, adjustment: str) -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=feed_map[feed],
        adjustment=adjustment,
    )
    resp = client.get_stock_bars(req)
    df = resp.df
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].drop_duplicates(subset=["timestamp"]).dropna()


def _merge_cache(symbol: str, new_bars: pd.DataFrame, replace: bool) -> dict:
    path = _cache_path(symbol)
    if replace or not path.exists():
        merged = new_bars.copy()
        old_rows = 0
    else:
        old = pd.read_parquet(path)
        old_rows = len(old)
        merged = pd.concat([old, new_bars], ignore_index=True)
    if merged.empty:
        return {"symbol": symbol, "old_rows": old_rows, "new_rows": len(new_bars), "merged_rows": 0, "path": str(path)}
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return {
        "symbol": symbol,
        "old_rows": int(old_rows),
        "new_rows": int(len(new_bars)),
        "merged_rows": int(len(merged)),
        "start": str(merged["timestamp"].min()),
        "end": str(merged["timestamp"].max()),
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=[], help="symbols or comma-separated symbol groups")
    parser.add_argument("--top500", action="store_true")
    parser.add_argument("--cached-all", action="store_true")
    parser.add_argument("--default-universe", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--start", required=True, help="inclusive UTC/RFC3339 start, e.g. 2016-01-01")
    parser.add_argument("--end", default="", help="exclusive UTC/RFC3339 end; defaults to now minus 16 minutes")
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--adjustment", choices=["raw", "split", "dividend", "all"], default="raw")
    parser.add_argument("--replace", action="store_true", help="replace cache instead of merging")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between symbols")
    args = parser.parse_args()

    symbols = _symbols(args)
    if not symbols:
        raise SystemExit("no symbols selected")
    start = _parse_dt(args.start)
    end = _parse_dt(args.end) if args.end else (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).to_pydatetime()
    client = _client()
    ok = failed = 0
    started = time.time()
    for idx, symbol in enumerate(symbols, start=1):
        try:
            bars = _fetch(client, symbol, start, end, args.feed, args.adjustment)
            stats = _merge_cache(symbol, bars, replace=args.replace)
            print(f"[alpaca-backfill] {idx:04d}/{len(symbols)} {symbol}: {stats}", flush=True)
            ok += 1
        except Exception as exc:
            print(f"[alpaca-backfill] {idx:04d}/{len(symbols)} {symbol}: failed ({exc})", flush=True)
            failed += 1
        if args.sleep > 0 and idx < len(symbols):
            time.sleep(args.sleep)
    print(f"[alpaca-backfill] done ok={ok} failed={failed} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
