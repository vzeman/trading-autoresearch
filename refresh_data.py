"""Refresh local Alpaca/yfinance bar cache without running training."""
from __future__ import annotations

import argparse
import time

from prepare import UNIVERSE, fetch_bars
from experiment import CONTEXT_SYMBOLS, EXTENDED_UNIVERSE, HOLDOUT_UNIVERSE
from top500_universe import load_top500_symbols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="redownload even when cache exists")
    parser.add_argument("--no-extended", action="store_true", help="skip extended training universe")
    parser.add_argument("--no-context", action="store_true", help="skip context symbols")
    parser.add_argument("--no-holdout", action="store_true", help="skip holdout symbols")
    parser.add_argument("--top500", action="store_true", help="include current S&P 500 constituents")
    parser.add_argument("--top500-only", action="store_true", help="fetch only current S&P 500 constituents")
    parser.add_argument("--limit", type=int, default=0, help="optional symbol limit after de-duplication")
    parser.add_argument("--start-at", type=int, default=1, help="1-based index to resume symbol refresh")
    args = parser.parse_args()

    symbols: list[str] = []
    if args.top500 or args.top500_only:
        top500 = load_top500_symbols()
        print(f"[refresh] loaded {len(top500)} current S&P 500 symbols", flush=True)
        symbols.extend(top500)
    if not args.top500_only:
        symbols.extend(UNIVERSE)
        if not args.no_context:
            symbols.extend(CONTEXT_SYMBOLS)
        if not args.no_extended:
            symbols.extend(EXTENDED_UNIVERSE)
        if not args.no_holdout:
            symbols.extend(HOLDOUT_UNIVERSE)

    seen = set()
    unique_symbols = [s for s in symbols if not (s in seen or seen.add(s))]
    if args.start_at > 1:
        unique_symbols = unique_symbols[args.start_at - 1:]
    if args.limit > 0:
        unique_symbols = unique_symbols[:args.limit]
    started = time.time()
    ok = 0
    failed = 0
    for idx, symbol in enumerate(unique_symbols, start=1):
        try:
            bars = fetch_bars(symbol, force=args.force)
            last_ts = bars["timestamp"].max() if len(bars) else "n/a"
            print(f"[refresh] {idx:03d}/{len(unique_symbols)} {symbol}: {len(bars):,} bars, last={last_ts}", flush=True)
            ok += 1
        except Exception as exc:
            print(f"[refresh] {idx:03d}/{len(unique_symbols)} {symbol}: failed ({exc})", flush=True)
            failed += 1
    print(f"[refresh] done ok={ok} failed={failed} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
