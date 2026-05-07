from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch

from experiment import (
    CHECKPOINT_DIR,
    CONTEXT_PATCHES,
    D_FF,
    D_MODEL,
    DROPOUT,
    HORIZONS_MINUTES,
    N_HEADS,
    N_LAYERS,
    PATCH_LEN,
    PRED_HORIZON,
    PatchTransformer,
    USE_FEATURES,
    featurize,
    fetch_context,
)
from prepare import UNIVERSE, fetch_bars


DEFAULT_PORT = int(os.environ.get("RECOMMEND_UI_PORT", "3007"))
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_HORIZON_IDXS = (3, 4)
DEFAULT_REFRESH = os.environ.get("RECOMMEND_REFRESH_DATA", "0") == "1"


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="300">
  <title>Trading Autoresearch Recommendations</title>
  <style>
    :root { color-scheme: light dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif; }
    body { margin: 0; padding: 28px; background: #0b1020; color: #e6edf7; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 22px; }
    h1 { margin: 0; font-size: 24px; }
    .muted { color: #94a3b8; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
    .card { background: #111827; border: 1px solid #243044; border-radius: 14px; padding: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.25); }
    .rank { color: #94a3b8; font-size: 13px; }
    .symbol { font-size: 28px; font-weight: 800; margin: 4px 0; }
    .score { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }
    .pill { background: #0b1220; border: 1px solid #263246; border-radius: 10px; padding: 8px; }
    .pill b { display: block; font-size: 16px; }
    svg { width: 100%; height: 150px; display: block; margin-top: 10px; }
    .line { fill: none; stroke: #22c55e; stroke-width: 2.2; }
    .axis { stroke: #334155; stroke-width: 1; }
    .warn { color: #fbbf24; }
    button { background: #2563eb; color: white; border: 0; border-radius: 10px; padding: 9px 12px; cursor: pointer; }
    button:disabled { opacity: .6; cursor: progress; }
    a { color: #93c5fd; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Top 5 Buy Candidates</h1>
      <div class="muted">Exp200-style signal: median seed forecast score over 4h + 1d. Auto-refresh: 5 minutes.</div>
    </div>
    <button id="reload">Reload now</button>
  </header>
  <div id="status" class="muted">Loading recommendations…</div>
  <div id="cards" class="grid"></div>
  <script>
    const fmtPct = v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;
    const fmtNum = v => `${v >= 0 ? '+' : ''}${v.toFixed(4)}`;
    function sparkline(points) {
      if (!points || points.length < 2) return '<div class="muted">No 12-month chart data.</div>';
      const w = 640, h = 150, pad = 8;
      const ys = points.map(p => p.close);
      const min = Math.min(...ys), max = Math.max(...ys);
      const span = Math.max(max - min, 1e-9);
      const d = points.map((p, i) => {
        const x = pad + i * (w - 2 * pad) / (points.length - 1);
        const y = h - pad - ((p.close - min) / span) * (h - 2 * pad);
        return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
      return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="12-month price chart">
        <line class="axis" x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}"></line>
        <path class="line" d="${d}"></path>
      </svg>`;
    }
    async function load() {
      const btn = document.getElementById('reload');
      btn.disabled = true;
      document.getElementById('status').textContent = 'Computing latest recommendations…';
      try {
        const res = await fetch('/api/recommendations?top=5');
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        document.getElementById('status').innerHTML =
          `Updated ${data.generated_at}. Data source: ${data.refresh_data ? 'fresh fetch' : 'local cache'}.
           <span class="warn">Research signal only; not financial advice.</span>`;
        document.getElementById('cards').innerHTML = data.recommendations.map(item => `
          <section class="card">
            <div class="rank">#${item.rank}</div>
            <div class="symbol">${item.symbol}</div>
            <div class="muted">Last bar: ${item.last_timestamp} · Last close: $${item.last_close.toFixed(2)}</div>
            <div class="score">
              <div class="pill"><span class="muted">Score</span><b>${fmtNum(item.score)}</b></div>
              <div class="pill"><span class="muted">4h forecast</span><b>${fmtPct(item.forecast_4h_pct)}</b></div>
              <div class="pill"><span class="muted">1d forecast</span><b>${fmtPct(item.forecast_1d_pct)}</b></div>
            </div>
            <div class="muted">12-month change: ${fmtPct(item.change_12m_pct)}</div>
            ${sparkline(item.chart)}
          </section>
        `).join('');
      } catch (err) {
        document.getElementById('status').innerHTML = `<span class="warn">Failed: ${err.message}</span>`;
      } finally {
        btn.disabled = false;
      }
    }
    document.getElementById('reload').addEventListener('click', load);
    load();
  </script>
</body>
</html>
"""


def _load_models(seed_ids: tuple[int, ...], device: str) -> list[PatchTransformer]:
    models: list[PatchTransformer] = []
    for seed in seed_ids:
        path = CHECKPOINT_DIR / f"last_seed{seed}.pt"
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location=device)
        use_features = ckpt.get("use_features", USE_FEATURES)
        if list(use_features) != list(USE_FEATURES):
            continue
        model = PatchTransformer(
            n_features=len(USE_FEATURES),
            patch_len=int(ckpt.get("patch_len", PATCH_LEN)),
            context_patches=int(ckpt.get("context_patches", CONTEXT_PATCHES)),
            d_model=int(ckpt.get("d_model", D_MODEL)),
            n_heads=int(ckpt.get("n_heads", N_HEADS)),
            n_layers=int(ckpt.get("n_layers", N_LAYERS)),
            d_ff=int(ckpt.get("d_ff", D_FF)),
            dropout=float(ckpt.get("dropout", DROPOUT)),
            pred_horizon=int(ckpt.get("pred_horizon", PRED_HORIZON)),
            horizons_minutes=list(ckpt.get("horizons_minutes", HORIZONS_MINUTES)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError("No compatible checkpoints found in checkpoints/last_seed{0,1,2}.pt")
    return models


def _chart_points(bars: pd.DataFrame) -> list[dict]:
    ts = pd.to_datetime(bars["timestamp"], utc=True)
    cutoff = ts.max() - pd.Timedelta(days=365)
    chart = bars.loc[ts >= cutoff, ["timestamp", "close"]].copy()
    if len(chart) > 260:
        idx = np.linspace(0, len(chart) - 1, 260).astype(int)
        chart = chart.iloc[idx]
    return [
        {"ts": str(row.timestamp), "close": float(row.close)}
        for row in chart.itertuples(index=False)
    ]


def build_recommendations(top_n: int, refresh_data: bool, seed_ids: tuple[int, ...]) -> dict:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    context = fetch_context(force=refresh_data)
    models = _load_models(seed_ids, device=device)
    rows: list[dict] = []
    for symbol in UNIVERSE:
        if symbol == "SPY":
            continue
        try:
            bars = fetch_bars(symbol, force=refresh_data)
            feat = featurize(bars, context=context)
            context_len = models[0].context_len
            if len(feat) < context_len:
                continue
            window = feat[USE_FEATURES].to_numpy(np.float32)[-context_len:]
            xb = torch.from_numpy(window[None, :, :]).to(device)
            seed_scores = []
            seed_4h = []
            seed_1d = []
            with torch.no_grad():
                for model in models:
                    mean, log_std = model.forward_multi_horizon(xb)
                    sharpe = mean / (torch.exp(log_std) + 1e-12)
                    score = float(sharpe[0, DEFAULT_HORIZON_IDXS[0]] + sharpe[0, DEFAULT_HORIZON_IDXS[1]])
                    seed_scores.append(score)
                    seed_4h.append(float(torch.expm1(mean[0, DEFAULT_HORIZON_IDXS[0]]) * 100.0))
                    seed_1d.append(float(torch.expm1(mean[0, DEFAULT_HORIZON_IDXS[1]]) * 100.0))
            closes = bars["close"].astype(float)
            change_12m = 0.0
            if len(closes) > 1:
                ts = pd.to_datetime(bars["timestamp"], utc=True)
                cutoff = ts.max() - pd.Timedelta(days=365)
                year = bars.loc[ts >= cutoff, "close"].astype(float)
                if len(year) > 1 and float(year.iloc[0]) > 0:
                    change_12m = (float(year.iloc[-1]) / float(year.iloc[0]) - 1.0) * 100.0
            rows.append({
                "symbol": symbol,
                "score": float(np.median(seed_scores)),
                "forecast_4h_pct": float(np.median(seed_4h)),
                "forecast_1d_pct": float(np.median(seed_1d)),
                "last_close": float(closes.iloc[-1]),
                "last_timestamp": str(bars["timestamp"].iloc[-1]),
                "change_12m_pct": float(change_12m),
                "chart": _chart_points(bars),
            })
        except Exception as exc:
            rows.append({
                "symbol": symbol,
                "score": -1e9,
                "forecast_4h_pct": 0.0,
                "forecast_1d_pct": 0.0,
                "last_close": 0.0,
                "last_timestamp": f"error: {exc}",
                "change_12m_pct": 0.0,
                "chart": [],
            })
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)[:top_n]
    for idx, row in enumerate(ranked, start=1):
        row["rank"] = idx
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "refresh_data": refresh_data,
        "strategy": "exp200-style score allocator: 4h + 1d median forecast score across compatible latest seed checkpoints",
        "recommendations": ranked,
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/recommendations"}:
            self._send(200, HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/recommendations":
            query = parse_qs(parsed.query)
            top_n = max(1, min(20, int(query.get("top", ["5"])[0])))
            refresh = query.get("refresh", ["1" if DEFAULT_REFRESH else "0"])[0] in {"1", "true", "yes"}
            try:
                payload = build_recommendations(top_n=top_n, refresh_data=refresh, seed_ids=DEFAULT_SEEDS)
                self._send(200, json.dumps(payload), "application/json")
            except Exception as exc:
                self._send(500, json.dumps({"error": str(exc)}), "application/json")
            return
        self._send(404, "not found", "text/plain")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[recommend-ui] {self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local trading recommendation UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[recommend-ui] open http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
