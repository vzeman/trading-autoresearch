"""Download full-history Alpaca 1-minute bars for a symbol universe.

This is a chunked/resumable variant of backfill_alpaca_history.py. It writes
the same cache format:

    ~/.cache/trading-autoresearch/{SYMBOL}_1m.parquet

Use SIP for deep history. Alpaca returns OHLCV minute bars, not individual
trade ticks.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from prepare import CACHE_DIR


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


def parse_utc(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def symbols_from_args(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        raw = [s.strip().upper() for part in args.symbols for s in part.split(",") if s.strip()]
    else:
        dataset = pd.read_parquet(args.dataset, columns=["symbol"])
        raw = sorted(dataset["symbol"].astype(str).str.upper().unique())
    if args.include_spy and "SPY" not in raw:
        raw = ["SPY"] + raw
    seen: set[str] = set()
    out = [s for s in raw if not (s in seen or seen.add(s))]
    if args.limit > 0:
        out = out[: args.limit]
    if args.start_at > 1:
        out = out[args.start_at - 1 :]
    return out


def client() -> object:
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("set ALPACA_API_KEY/ALPACA_SECRET_KEY or APCA_API_KEY_ID/APCA_API_SECRET_KEY")
    from alpaca.data.historical import StockHistoricalDataClient

    return StockHistoricalDataClient(key, secret)


def fetch_bars(client_: object, symbol: str, start: pd.Timestamp, end: pd.Timestamp, feed: str, adjustment: str) -> pd.DataFrame:
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    feed_map = {"iex": DataFeed.IEX, "sip": DataFeed.SIP}
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Minute),
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        feed=feed_map[feed],
        adjustment=adjustment,
    )
    resp = client_.get_stock_bars(req)
    df = resp.df
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("timestamp")


def existing_bounds(symbol: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    path = cache_path(symbol)
    if not path.exists():
        return None, None, 0
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return None, None, 0
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max()), int(len(df))


def merge_cache(symbol: str, new_parts: list[pd.DataFrame]) -> dict:
    path = cache_path(symbol)
    frames = []
    old_rows = 0
    if path.exists():
        old = pd.read_parquet(path)
        old_rows = int(len(old))
        frames.append(old)
    frames.extend(part for part in new_parts if not part.empty)
    if not frames:
        return {"symbol": symbol, "old_rows": old_rows, "new_rows": 0, "merged_rows": old_rows, "path": str(path)}
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True).astype("datetime64[ns, UTC]")
    merged = merged.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return {
        "symbol": symbol,
        "old_rows": old_rows,
        "new_rows": int(sum(len(part) for part in new_parts)),
        "merged_rows": int(len(merged)),
        "start": str(merged["timestamp"].min()) if not merged.empty else "",
        "end": str(merged["timestamp"].max()) if not merged.empty else "",
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024**2, 2),
    }


def chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = []
    cur = start
    while cur < end:
        nxt = min(cur + pd.DateOffset(months=months), end)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_top10_volume_valuation_algo.parquet")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--include-spy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="", help="defaults to now minus 16 minutes")
    parser.add_argument("--feed", choices=["iex", "sip"], default="sip")
    parser.add_argument("--adjustment", choices=["raw", "split", "dividend", "all"], default="raw")
    parser.add_argument("--chunk-months", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-at", type=int, default=1)
    parser.add_argument("--skip-covered", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--manifest", default="checkpoints/data_download/full_alpaca_1m_manifest.json")
    args = parser.parse_args()

    start = parse_utc(args.start)
    end = parse_utc(args.end) if args.end else pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)
    ranges = chunk_ranges(start, end, args.chunk_months)
    symbols = symbols_from_args(args)
    if not symbols:
        raise SystemExit("no symbols selected")

    print(
        f"[full-1m] symbols={len(symbols)} feed={args.feed} adjustment={args.adjustment} "
        f"start={start} end={end} chunks={len(ranges)} cache={CACHE_DIR}",
        flush=True,
    )
    c = client()
    manifest = {"args": vars(args), "symbols": {}, "started_at": str(pd.Timestamp.now(tz="UTC"))}
    started = time.time()
    ok = failed = 0
    for idx, symbol in enumerate(symbols, start=1):
        old_start, old_end, old_rows = existing_bounds(symbol)
        if args.skip_covered and old_start is not None and old_start <= start and old_end is not None and old_end >= end - pd.Timedelta(days=1):
            print(f"[full-1m] {idx:03d}/{len(symbols)} {symbol}: already covered rows={old_rows} {old_start}->{old_end}", flush=True)
            ok += 1
            continue
        print(f"[full-1m] {idx:03d}/{len(symbols)} {symbol}: existing rows={old_rows} {old_start}->{old_end}", flush=True)
        parts: list[pd.DataFrame] = []
        symbol_failed = False
        for chunk_idx, (chunk_start, chunk_end) in enumerate(ranges, start=1):
            attempt = 0
            while True:
                try:
                    bars = fetch_bars(c, symbol, chunk_start, chunk_end, args.feed, args.adjustment)
                    parts.append(bars)
                    print(
                        f"[full-1m] {symbol} chunk {chunk_idx:03d}/{len(ranges)} "
                        f"{chunk_start.date()}->{chunk_end.date()} rows={len(bars):,}",
                        flush=True,
                    )
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > args.retries:
                        print(f"[full-1m] {symbol} chunk {chunk_idx}: failed after retries ({exc})", flush=True)
                        symbol_failed = True
                        break
                    wait = max(args.sleep, 1.0) * attempt
                    print(f"[full-1m] {symbol} chunk {chunk_idx}: retry {attempt}/{args.retries} after {wait:.1f}s ({exc})", flush=True)
                    time.sleep(wait)
            if symbol_failed:
                break
            if args.sleep > 0:
                time.sleep(args.sleep)
        if symbol_failed:
            failed += 1
            continue
        stats = merge_cache(symbol, parts)
        manifest["symbols"][symbol] = stats
        write_manifest(Path(args.manifest), manifest)
        print(f"[full-1m] {idx:03d}/{len(symbols)} {symbol}: merged {stats}", flush=True)
        ok += 1
    manifest["finished_at"] = str(pd.Timestamp.now(tz="UTC"))
    manifest["ok"] = ok
    manifest["failed"] = failed
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    write_manifest(Path(args.manifest), manifest)
    print(f"[full-1m] done ok={ok} failed={failed} elapsed={time.time() - started:.1f}s manifest={args.manifest}", flush=True)


if __name__ == "__main__":
    main()
