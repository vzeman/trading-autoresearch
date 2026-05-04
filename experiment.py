"""THE FILE THE AGENT MODIFIES.

Single self-contained file with:
  - FEATURE ENGINEERING (multi-scale returns, EMA distances, vol windows, etc.)
  - Model (PatchTST-style transformer, forecast head + 3-action head)
  - Replay buffers
  - Train loop (supervised pretrain + offline RL pretrain on TRAIN slice)
  - Eval loop on the held-out EVAL slice

`evaluator.py` imports `train_and_eval(seed)` from this file. Everything else
the agent decides — features, architecture, optimizer, hyperparameters,
reward shaping, exploration schedule — as long as the contract holds.

CONTRACT (do not break):
  - `train_and_eval(seed: int) -> (equity_curve, n_trades, total_fees, total_slippage)`
    where equity_curve is broker.equity_curve from the EVAL slice run.
  - Use `prepare.PaperBroker` for fills + fees + slippage.
  - Use `prepare.split(bars)` for the chronological train/eval split. NO leakage.
  - All features must be CAUSAL (only depend on bars at or before time t).
"""
from __future__ import annotations
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Speedup #1: skip pretrain when only post-training params (cap, SWAP_MARGIN, etc.)
# changed between iterations. Loads checkpoints/last_seed{seed}.pt instead.
# Set USE_CACHED_PRETRAIN=1 in the env when launching the driver.
USE_CACHED_PRETRAIN = os.environ.get("USE_CACHED_PRETRAIN", "0") == "1"
# exp81/83: bfloat16 autocast on MPS — was on by default but exp83 measured it
# is actually SLOWER than fp32 on Apple Metal (~75min/seed bf16 vs ~55min/seed fp32).
# Apple's Metal doesn't have specialized bf16 tensor cores like NVIDIA. Leaving as
# opt-in via USE_AMP=1 for future hardware (e.g. M5 / cloud H100).
USE_AMP_PRETRAIN = os.environ.get("USE_AMP", "0") == "1"
from contextlib import nullcontext as _nullcontext

from prepare import (
    UNIVERSE, NOTIONAL_PER_SYMBOL_USD, STARTING_CASH_USD,
    FEE_PER_TRADE_USD, SLIPPAGE_BPS,
    PaperBroker, prepare_all, split, fetch_bars,
)

# ============================================================================
# CONTEXT SYMBOLS — macro + cross-asset signals fetched alongside the universe
# ============================================================================
# These are not traded; their bars are merged-asof onto each universe bar to
# provide cross-asset / macro features.
CONTEXT_SYMBOLS = ["TLT", "UUP", "SPY"]   # 20yr Treasury, USD-index proxy, SPY (cross-asset). exp58: dropped ^VIX (only 27 days of yfinance data).

# ============================================================================
# HOLDOUT UNIVERSE — stocks the model NEVER trains on. After the in-symbol eval
# completes, the same trained model is run on these names' last-90d bars to
# test out-of-symbol generalization. A model that learned real signal should
# produce a comparable (within ~30%) sharpe here vs in-sample.
# ============================================================================
HOLDOUT_UNIVERSE = ["JPM", "WMT", "V", "DIS", "JNJ"]

# ============================================================================
# EXTENDED UNIVERSE — exp56: extends prepare.UNIVERSE (20 names) to ~95 names
# spanning the top of S&P 500. Adding more training data to test whether the
# model generalizes better when it sees a wider set of price dynamics.
# Symbols are fetched lazily via prepare.fetch_bars (cached on disk).
# Names whose cache file doesn't exist yet are silently skipped — useful while
# the download script is still running in background.
# ============================================================================
EXTENDED_UNIVERSE = [
    "UNH", "XOM", "MA", "PG", "HD", "CVX", "LLY", "ABBV", "KO", "PEP",
    "AVGO", "COST", "MCD", "TMO", "MRK", "ACN", "ABT", "NKE", "BA", "ORCL",
    "PFE", "CRM", "DHR", "NEE", "TXN", "VZ", "ADBE", "CMCSA", "BMY", "T",
    "PM", "RTX", "QCOM", "UPS", "HON", "LIN", "IBM", "LOW", "AMT", "AMGN",
    "AMAT", "LMT", "INTU", "GS", "CAT", "SPGI", "NOW", "GE", "BLK", "AXP",
    "ELV", "MS", "MDLZ", "BKNG", "DE", "ADI", "PLD", "ISRG", "MMC", "GILD",
    "MO", "SBUX", "ETN", "ZTS", "REGN", "VRTX", "C", "COF", "SCHW", "CI",
    "DUK", "BSX", "USB", "CB", "SYK",
]

# ============================================================================
# FEATURE ENGINEERING — agent edits freely. Must remain causal.
# ============================================================================

# Available feature names. Add a name here AND implement it in featurize().
# `USE_FEATURES` controls which subset feeds the model.
ALL_FEATURES = [
    # Returns at multiple horizons
    "log_return_1", "log_return_5", "log_return_15", "log_return_60",
    # EMA distances (price vs trend)
    "ema_dev_20", "ema_dev_60",
    # Volume features
    "log_volume_dev_60",
    "vol_z_15", "vol_z_60",
    "signed_log_vol",
    # Realized vol at multiple windows + ratio
    "rv_15", "rv_60", "rv_15_div_60",
    # Range features (intraday volatility proxy)
    "hl_range",
    # Time of day
    "tod_sin", "tod_cos",
    # exp11: cross-asset / macro context (forward-filled to each universe bar)
    # exp58: dropped vix_logret_1 (only 27 days of yfinance data)
    "tlt_logret_1",   # 20yr Treasury ETF — interest-rate signal
    "uup_logret_1",   # USD-index ETF (DXY proxy) — currency macro
    "spy_logret_1",   # SPY return as a market factor (for SPY itself this == log_return_1)
    # exp82: REVERTED exp79's universe-context features — they regressed sharpe
    # +1.549 → +0.055. Probably scale-mismatch (pct_above_ma60 in [0,1] vs others
    # ~1e-4). Re-add later with proper z-score normalization. The featurize() stub
    # init + add_universe_context() helper stay in place for the future retry.
]

USE_FEATURES = [f for f in ALL_FEATURES if f not in {"signed_log_vol", "vol_z_15"}]   # exp4 + exp11: drop 2 noisy + add 4 context


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA via pandas."""
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def fetch_context() -> dict[str, pd.DataFrame]:
    """Fetch & cache the context symbols' bars; return {sym: log_return_series}.

    Each value is a DataFrame with columns ['timestamp', 'logret'] — sorted by ts.
    Used by featurize() via merge_asof to forward-fill onto universe bars.
    """
    out: dict[str, pd.DataFrame] = {}
    for sym in CONTEXT_SYMBOLS:
        try:
            bars = fetch_bars(sym)
        except Exception as e:
            print(f"[context] {sym}: fetch failed ({e}) — feature will be 0", flush=True)
            continue
        df = bars[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)
        c = df["close"].to_numpy(np.float64)
        lr = np.zeros_like(c)
        lr[1:] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
        out[sym] = pd.DataFrame({"timestamp": df["timestamp"], "logret": lr.astype(np.float32)})
        print(f"[context] {sym}: {len(out[sym]):,} bars cached", flush=True)
    return out


_CONTEXT_KEY_TO_FEATURE = {
    "TLT":  "tlt_logret_1",
    "UUP":  "uup_logret_1",
    "SPY":  "spy_logret_1",
}


def featurize(bars: pd.DataFrame, context: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """OHLCV bars → causal feature dataframe + the close price (broker needs it).

    If `context` is provided (mapping context symbol → log-return series),
    those returns are forward-filled onto each universe bar's timestamp via
    backward merge_asof — strictly causal (no future leakage).

    Returns a frame with columns ['timestamp', 'close', *ALL_FEATURES].
    """
    df = bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    vol = df["volume"].to_numpy(np.float64)

    # ---- multi-horizon log returns ----
    def lret(k: int) -> np.ndarray:
        out = np.zeros_like(close)
        out[k:] = np.log(close[k:] / np.maximum(close[:-k], 1e-12))
        return out

    log_return_1 = lret(1)
    log_return_5 = lret(5)
    log_return_15 = lret(15)
    log_return_60 = lret(60)

    # ---- EMA distances ----
    ema_20 = _ema(close, 20)
    ema_60 = _ema(close, 60)
    ema_dev_20 = (close - ema_20) / np.maximum(close, 1e-12)
    ema_dev_60 = (close - ema_60) / np.maximum(close, 1e-12)

    # ---- volume features ----
    log_vol = np.log1p(vol)
    log_vol_mean_60 = pd.Series(log_vol).rolling(60, min_periods=1).mean().to_numpy()
    log_vol_std_15 = pd.Series(log_vol).rolling(15, min_periods=2).std(ddof=0).fillna(1e-6).to_numpy()
    log_vol_mean_15 = pd.Series(log_vol).rolling(15, min_periods=1).mean().to_numpy()
    log_vol_std_60 = pd.Series(log_vol).rolling(60, min_periods=2).std(ddof=0).fillna(1e-6).to_numpy()
    log_volume_dev_60 = log_vol - log_vol_mean_60
    vol_z_15 = (log_vol - log_vol_mean_15) / np.maximum(log_vol_std_15, 1e-6)
    vol_z_60 = (log_vol - log_vol_mean_60) / np.maximum(log_vol_std_60, 1e-6)
    signed_log_vol = log_volume_dev_60 * np.sign(log_return_1)

    # ---- realized vol ----
    rv_15 = pd.Series(log_return_1).rolling(15, min_periods=2).std(ddof=0).fillna(0.0).to_numpy()
    rv_60 = pd.Series(log_return_1).rolling(60, min_periods=2).std(ddof=0).fillna(0.0).to_numpy()
    rv_15_div_60 = rv_15 / np.maximum(rv_60, 1e-9)

    # ---- range features ----
    hl_range = (high - low) / np.maximum(close, 1e-12)

    # ---- time of day (regular session 09:30-16:00 ET, i.e. 23,400 sec) ----
    et = df["timestamp"].dt.tz_convert("America/New_York")
    sec = ((et.dt.hour - 9) * 3600 + (et.dt.minute - 30) * 60 + et.dt.second).to_numpy(np.float64)
    period = 6.5 * 3600
    tod_sin = np.sin(2 * math.pi * sec / period)
    tod_cos = np.cos(2 * math.pi * sec / period)

    feat = pd.DataFrame({
        "timestamp": df["timestamp"],
        "close": df["close"].astype(np.float32),
        "volume": df["volume"].astype(np.float32),   # exp58: raw bar volume for liquidity-aware slippage in simulate_weighted (not a model feature)
        "log_return_1": log_return_1.astype(np.float32),
        "log_return_5": log_return_5.astype(np.float32),
        "log_return_15": log_return_15.astype(np.float32),
        "log_return_60": log_return_60.astype(np.float32),
        "ema_dev_20": ema_dev_20.astype(np.float32),
        "ema_dev_60": ema_dev_60.astype(np.float32),
        "log_volume_dev_60": log_volume_dev_60.astype(np.float32),
        "vol_z_15": vol_z_15.astype(np.float32),
        "vol_z_60": vol_z_60.astype(np.float32),
        "signed_log_vol": signed_log_vol.astype(np.float32),
        "rv_15": rv_15.astype(np.float32),
        "rv_60": rv_60.astype(np.float32),
        "rv_15_div_60": rv_15_div_60.astype(np.float32),
        "hl_range": hl_range.astype(np.float32),
        "tod_sin": tod_sin.astype(np.float32),
        "tod_cos": tod_cos.astype(np.float32),
    })

    # ---- Context features: backward merge_asof (causal) ----
    feat = feat.sort_values("timestamp").reset_index(drop=True)
    for ctx_sym, feat_name in _CONTEXT_KEY_TO_FEATURE.items():
        if context is not None and ctx_sym in context:
            ctx_df = context[ctx_sym].sort_values("timestamp").reset_index(drop=True)
            merged = pd.merge_asof(
                feat[["timestamp"]], ctx_df, on="timestamp", direction="backward",
            )
            feat[feat_name] = merged["logret"].fillna(0.0).astype(np.float32).to_numpy()
        else:
            feat[feat_name] = np.zeros(len(feat), dtype=np.float32)
    # exp79: stub-init the universe-context columns; populated by add_universe_context()
    # after all symbols have been featurized. Keeps featurize() signature unchanged.
    feat["univ_mean_logret_1"] = np.zeros(len(feat), dtype=np.float32)
    feat["univ_disp_logret_1"] = np.zeros(len(feat), dtype=np.float32)
    feat["univ_pct_above_ma60"] = np.zeros(len(feat), dtype=np.float32)
    return feat


def add_universe_context(features: dict[str, pd.DataFrame]) -> None:
    """exp79: compute per-timestep universe aggregates and merge into each symbol's
    frame in-place. Strictly causal (no future leakage) — uses only the same-timestamp
    log_return_1 and ema_dev_60 values that already exist in each symbol's frame.
    """
    if not features:
        return
    rets_by_ts: dict = {}
    above_by_ts: dict = {}
    for sym, f in features.items():
        ts_arr = f["timestamp"].to_numpy()
        ret_arr = f["log_return_1"].to_numpy(np.float32)
        ema_dev_arr = f["ema_dev_60"].to_numpy(np.float32)
        for i in range(len(ts_arr)):
            ts = ts_arr[i]
            r = float(ret_arr[i])
            if not np.isfinite(r):
                continue
            rets_by_ts.setdefault(ts, []).append(r)
            cnt, tot = above_by_ts.get(ts, (0, 0))
            above_by_ts[ts] = (cnt + (1 if ema_dev_arr[i] > 0 else 0), tot + 1)
    if not rets_by_ts:
        return
    sorted_ts = sorted(rets_by_ts.keys())
    means_arr = np.zeros(len(sorted_ts), dtype=np.float32)
    disp_arr = np.zeros(len(sorted_ts), dtype=np.float32)
    breadth_arr = np.zeros(len(sorted_ts), dtype=np.float32)
    for i, ts in enumerate(sorted_ts):
        rets = np.array(rets_by_ts[ts], dtype=np.float32)
        means_arr[i] = float(rets.mean())
        disp_arr[i] = float(rets.std()) if len(rets) > 1 else 0.0
        cnt, tot = above_by_ts[ts]
        breadth_arr[i] = float(cnt / max(tot, 1))
    ctx = pd.DataFrame({
        "timestamp": sorted_ts,
        "univ_mean_logret_1": means_arr,
        "univ_disp_logret_1": disp_arr,
        "univ_pct_above_ma60": breadth_arr,
    })
    for sym, f in features.items():
        merged = pd.merge_asof(
            f[["timestamp"]].sort_values("timestamp").reset_index(drop=True),
            ctx, on="timestamp", direction="backward",
        )
        f["univ_mean_logret_1"] = merged["univ_mean_logret_1"].fillna(0.0).astype(np.float32).to_numpy()
        f["univ_disp_logret_1"] = merged["univ_disp_logret_1"].fillna(0.0).astype(np.float32).to_numpy()
        f["univ_pct_above_ma60"] = merged["univ_pct_above_ma60"].fillna(0.5).astype(np.float32).to_numpy()


# ============================================================================
# HYPERPARAMETERS — agent edits freely
# ============================================================================

PATCH_LEN = 8
CONTEXT_PATCHES = 16            # context window = PATCH_LEN * CONTEXT_PATCHES = 128 bars
D_MODEL = 128                   # exp58+: 96→128 (more capacity for 11-horizon multi-task head)
N_HEADS = 4
N_LAYERS = 4                    # 3→4 (deeper for multi-horizon)
D_FF = 256                      # 2 × D_MODEL
DROPOUT = 0.1   # exp22 (DROPOUT=0) was disastrous — keep regularization
PRED_HORIZON = 5
# 11-horizon multi-task. 5m → 30d. Same model serves multiple trader profiles.
HORIZONS_MINUTES = [5, 60, 120, 240, 390, 780, 1170, 1560, 1950, 5460, 11700]
RL_REWARD_HORIZON = 3
ACTION_HEAD_HOLD_BIAS = 1.5     # exp10: softmax([-1.5,1.5,-1.5]) ≈ [4.7%,90.6%,4.7%]: be even more selective

PRETRAIN_EPOCHS = 1             # exp88: REVERT 2→1 — exp87 (2 epochs) regressed sharpe +1.55 → +0.34. Classic overfit: NLL went lower (−10.98 vs −9.5) but seed-2 generalization collapsed to −1.14.
# exp67: validate same canonical with N_SEEDS=3 — driver gate compares ci_low against
# the prior best (set by 3-seed exp51); a 1-seed bootstrap CI is unfair. With the exp66
# precompute speedup each seed costs ~120s so 3 seeds is affordable (~6min total).
# exp83: fresh pretrain WITH bf16 autocast (exp81 infra) on the reverted 17-feature
# config. Same seeds as exp71 KEEP but with autocast — should reproduce ~+1.5 sharpe
# in ~110min instead of exp79's 167min (~33% wall-clock savings).
# exp76: bump to N_SEEDS=5 to dilute seed-2 drag. exp71-75 (cached pretrain) consistently
# saw seed 2 produce a single losing trade ($-428 / -0.43 sharpe) that pulled the median
# below the gate. Adding 2 more seeds should give a more robust median (~+1.5 if the
# signal is real) and tighten ci_low for a clean KEEP.
# exp68: FRESH FULL PRETRAIN with ranking loss actually engaged. exp67 showed median
# 3-seed sharpe = +0.82 < SPY +1.00 — the cached pretrain we'd been reusing was from
# before USE_RANK_LOSS was infrastructure, so the model never actually trained against
# the ranking objective. This iter scraps cached weights (USE_CACHED_PRETRAIN=0) and
# does a fresh full pretrain. Tests whether stale weights were the bottleneck.
# exp63: cross-sectional ranking + standardization (research-backed: CIKM 2025, JFDS 2021).
# Trains the model to predict RELATIVE outperformance vs the universe at each timestep
# rather than absolute returns. Documented ~3× sharpe lift in published comparisons.
USE_CSEC_STANDARDIZATION = True  # subtract per-timestep universe mean from y_mh targets
USE_RANK_LOSS = True             # add pairwise margin ranking loss within each timestep
RANK_MARGIN = 0.05               # margin in pred-units for pairwise ranking
RANK_LOSS_COEF = 1.0             # weight of ranking loss vs Gaussian NLL
TRAIN_LOOKBACK_DAYS = 365       # exp41: subset train slice to last N days. Hypothesis: model trained on full 6yr is too conservative for recent regime → exp40 = 0 trades. Recent-only data should produce more confident predictions.
PRETRAIN_BATCH = 128
PRETRAIN_LR = 3e-4
RL_PRETRAIN_EPOCHS = 1          # offline RL pass(es) on TRAIN slice
RL_LR = 2e-5     # exp7 KEPT setting (known stable, no rogue seeds)
RL_COEF = 1.0
ENTROPY_COEF = 0.005   # exp50: 0.01→0.005. exp49 (SWAP_MARGIN=0.15) had best raw sharpe (+1.548) and PnL (+$3,442) but ci_low -1.523 missed exp47's -1.513 by 0.01 — wider per-seed variance suggested too much exploration. Tighter convergence should narrow CI.
VOL_PENALTY = 0.0   # exp20/21 showed: small penalty=invisible, large penalty=destabilizing. Off.
SPY_ALPHA_COEF = 0.5   # exp51: weight on alpha-vs-SPY bonus added to RL reward. 0=ignore, 1=full alpha. 0.5=balanced (keep absolute reward base + half-weight alpha bonus). Trains the model to seek positions that BEAT the market, not just predict positive returns.

# ============================================================================
# STRATEGY-LEVEL "STICKINESS" — minimum time between portfolio moves.
# Each strategy defines its own. Higher = more committed positions, less churn.
# 1 = current behavior (can change every bar).
# ============================================================================
PRIMARY_MIN_HOLD_BARS = 1     # primary strategy: how many bars to hold before allowing position change
# (Picker already has PICKER_BUY_COOLDOWN_S = 5 min between BUYs.)

# exp26: in a bull market, SELLs systematically lose. Force long-only.
LONG_ONLY = True              # if True: SELL action is treated as HOLD

# exp27: bypass RL action_head entirely — derive action from forecast head's
# predicted Sharpe (mean / std over pred_horizon). The forecast trains with much
# denser signal than REINFORCE and may have real predictive value.
USE_FORECAST_POLICY = True            # exp28: re-test now that we have multi-horizon predictions
FORECAST_BUY_SHARPE_THRESHOLD = 0.5   # exp28 setting (exp30 lowering had no effect)
FORECAST_HORIZON_IDX = 1              # 1h horizon (exp31 ensemble was a disaster)
SGD_BATCH = 64
GRAD_CLIP = 1.0
RL_STEP_EVERY_BARS = 5
# Bound replay buffer for RL pretrain — at 6yr × 20 symbols an unbounded
# buffer reaches ~100 GB (each entry is ~9 KB). 100k entries ≈ 900 MB.
RL_BUFFER_MAX = 100_000


def pick_device() -> str:
    """Picks compute device. CPU is forced on M-series Macs because:
      - Our batch size (≤5 = num universe symbols per timestep) is too small
        for MPS launch overhead to amortize.
      - Empirically MPS is ~6× slower than CPU on this exact workload
        (verified at d_model=96, batch=5).
      - For larger models or batches, MPS would win — re-enable then.
      - Speedup #4: opt-in MPS via TRY_MPS=1 env var so we can re-test with
        current config (larger pretrain batches may now favor MPS).
    """
    if torch.cuda.is_available():
        return "cuda"
    if os.environ.get("TRY_MPS", "0") == "1" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================================
# MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatchTransformer(nn.Module):
    """PatchTST-style. Forecast head (Gaussian) + action head (3-way softmax)."""
    MIN_LOG_STD = -10.0
    MAX_LOG_STD = 2.0

    def __init__(self, n_features: int, patch_len: int, context_patches: int,
                 d_model: int, n_heads: int, n_layers: int, d_ff: int,
                 dropout: float, pred_horizon: int,
                 horizons_minutes: list[int] | None = None) -> None:
        super().__init__()
        self.n_features = n_features
        self.patch_len = patch_len
        self.context_patches = context_patches
        self.pred_horizon = pred_horizon
        self.horizons_minutes = horizons_minutes or []

        in_dim = patch_len * n_features
        self.patch_proj = nn.Linear(in_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=context_patches + 16)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, pred_horizon * 2),
        )
        # exp28: separate multi-horizon head — predicts cumulative log-return
        # Gaussian at each horizon (1m, 1h, 1d, 1w). Independent of pred_horizon
        # (which still trains per-step).
        if self.horizons_minutes:
            self.mh_head = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(),
                nn.Linear(d_ff, len(self.horizons_minutes) * 2),
            )
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, 3),
        )
        nn.init.zeros_(self.action_head[-1].weight)
        self.action_head[-1].bias.data = torch.tensor(
            [-ACTION_HEAD_HOLD_BIAS, ACTION_HEAD_HOLD_BIAS, -ACTION_HEAD_HOLD_BIAS]
        )

    @property
    def context_len(self) -> int:
        return self.patch_len * self.context_patches

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        tokens = self.patch_proj(x.reshape(b, t // self.patch_len, self.patch_len * f))
        tokens = self.pos(tokens)
        n = tokens.size(1)
        mask = torch.triu(torch.ones(n, n, device=tokens.device, dtype=torch.bool), diagonal=1)
        h = self.body(tokens, mask=mask, is_causal=True)
        return self.norm(h[:, -1])

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        out = self.head(h).view(-1, self.pred_horizon, 2)
        mean, log_std = out[..., 0], out[..., 1].clamp(self.MIN_LOG_STD, self.MAX_LOG_STD)
        action_logits = self.action_head(h)
        return mean, log_std, action_logits

    def forward_multi_horizon(self, x: torch.Tensor):
        """Forward pass returning multi-horizon (mean, log_std). Only available
        if `horizons_minutes` was set at construction."""
        if not self.horizons_minutes:
            return None, None
        h = self.encode(x)
        out = self.mh_head(h).view(-1, len(self.horizons_minutes), 2)
        mean = out[..., 0]
        log_std = out[..., 1].clamp(self.MIN_LOG_STD, self.MAX_LOG_STD)
        return mean, log_std

    @staticmethod
    def gaussian_nll(mean, log_std, target) -> torch.Tensor:
        var = torch.exp(2 * log_std)
        return (log_std + 0.5 * (target - mean) ** 2 / var + 0.5 * math.log(2 * math.pi)).mean()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# WINDOWING — on-demand (lazy) batch builder
# ============================================================================
# History note: pre-materializing all training windows as one (N, C, F) float32
# tensor exploded RAM at the 6yr × 20-symbol × 128-context × 18-feature config
# (~100 GB per worker, OOM'd a 96 GB Mac when evaluator ran 3 parallel workers).
# Now we keep only per-symbol (N_bars, F) feature arrays (~1 GB total) and slice
# (B, C, F) batches per minibatch via index gather.

class WindowDataset:
    """Lazy windowed dataset for supervised pretrain.

    Per window index `(sym_idx, i)`:
      - X      = feat[i : i + context_len]                    (C, F)
      - y_step = log_return_1[i + C : i + C + pred_horizon]    (H,)
      - y_mh   = [log(close[i + C + h] / close[i + C])  for h in horizons]
    """

    def __init__(self, features: dict[str, pd.DataFrame], context_len: int,
                 pred_horizon: int, horizons_minutes: list[int] | None) -> None:
        self.context_len = context_len
        self.pred_horizon = pred_horizon
        self.horizons = list(horizons_minutes or [])
        self.max_h = max([pred_horizon] + self.horizons) if self.horizons else pred_horizon
        cols = USE_FEATURES
        self.n_features = len(cols)
        self.feat_arrs: list[np.ndarray] = []
        self.lr_arrs: list[np.ndarray] = []
        self.close_arrs: list[np.ndarray] = []
        # Flat index of valid windows: parallel arrays for cheap fancy-indexing.
        sym_ids: list[int] = []
        starts: list[int] = []
        for sym, feat in features.items():
            arr = feat[cols].to_numpy(np.float32)
            lr = feat["log_return_1"].to_numpy(np.float32)
            cl = feat["close"].to_numpy(np.float32)
            n = len(feat) - context_len - self.max_h
            if n <= 0:
                continue
            local = len(self.feat_arrs)
            self.feat_arrs.append(arr)
            self.lr_arrs.append(lr)
            self.close_arrs.append(cl)
            sym_ids.extend([local] * n)
            starts.extend(range(n))
        self.sym_idx = np.asarray(sym_ids, dtype=np.int32)
        self.start = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.start.shape[0])

    def get_batch(self, idxs: np.ndarray):
        """idxs: 1-D int array. Returns (X, y_step, y_mh|None) numpy float32 arrays."""
        B = idxs.shape[0]
        C = self.context_len
        H = self.pred_horizon
        F = self.n_features
        X = np.empty((B, C, F), np.float32)
        y_step = np.empty((B, H), np.float32)
        y_mh = np.empty((B, len(self.horizons)), np.float32) if self.horizons else None
        for k in range(B):
            idx = idxs[k]
            s = int(self.sym_idx[idx])
            i = int(self.start[idx])
            X[k] = self.feat_arrs[s][i : i + C]
            y_step[k] = self.lr_arrs[s][i + C : i + C + H]
            if y_mh is not None:
                cl = self.close_arrs[s]
                t = i + C
                base = max(float(cl[t]), 1e-12)
                for j, h in enumerate(self.horizons):
                    fc = max(float(cl[t + h]), 1e-12)
                    y_mh[k, j] = math.log(fc / base)
        return X, y_step, y_mh


# ============================================================================
# SUPERVISED PRETRAIN
# ============================================================================

def supervised_pretrain(model: PatchTransformer, train_features: dict[str, pd.DataFrame], device: str):
    """Supervised pretrain with optional cross-sectional ranking loss.

    exp63: per research (CIKM 2025, JFDS 2021) the highest-ROI change for
    SPY-relative alpha is to (a) z-score targets across the universe at each
    timestep, (b) add a pairwise ranking loss within each timestep's symbols.
    Together this trains the model to predict RELATIVE outperformance (the
    only thing that produces alpha vs the index).
    """
    horizons = model.horizons_minutes
    ds = WindowDataset(train_features, model.context_len, model.pred_horizon, horizons)
    n = len(ds)
    if n == 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    model.train()
    # exp80: live training-loss log so we can chart loss falling during long pretrains
    # without waiting for subprocess stdout to flush. Append-only JSONL, one row per
    # batch print (every TRAIN_LOG_EVERY batches). Cleared at the start of each pretrain.
    import json as _json
    seed = int(np.random.get_state()[1][0]) % 100000   # rough seed identifier
    train_log_path = CHECKPOINT_DIR / f"last_train_loss.jsonl"
    try:
        train_log_path.write_text("")  # truncate at start
    except Exception:
        pass
    TRAIN_LOG_EVERY = max(1, n // (PRETRAIN_BATCH * 50))   # ~50 points per epoch
    for ep in range(PRETRAIN_EPOCHS):
        perm = np.random.permutation(n)
        losses, losses_mh, losses_rank = [], [], []
        batch_idx_print = 0
        for i in range(0, n - SGD_BATCH, PRETRAIN_BATCH):
            batch_idxs = perm[i : i + PRETRAIN_BATCH]
            X_np, y_np, y_mh_np = ds.get_batch(batch_idxs)
            xb = torch.from_numpy(X_np).to(device)
            yb = torch.from_numpy(y_np).to(device)
            opt.zero_grad(set_to_none=True)
            # exp81: bfloat16 autocast on MPS — ~1.5× speedup on transformer matmuls.
            # bfloat16 (not fp16) because: same exponent range as fp32 → no underflow
            # on Gaussian NLL log_std; no GradScaler needed (CUDA-only anyway).
            # Disabled for non-MPS (CPU bfloat16 is slower; CUDA prefers fp16+scaler).
            use_amp = USE_AMP_PRETRAIN and device == "mps"
            amp_ctx = torch.autocast(device_type="mps", dtype=torch.bfloat16) if use_amp else _nullcontext()
            with amp_ctx:
                mean, log_std, _ = model(xb)
                loss = PatchTransformer.gaussian_nll(mean, log_std, yb)

                rank_loss_val = 0.0
                if y_mh_np is not None:
                    ymh_b = torch.from_numpy(y_mh_np).to(device)
                    # Group windows by timestamp (start position) — same timestep across symbols
                    batch_starts = ds.start[batch_idxs]
                    if USE_CSEC_STANDARDIZATION or USE_RANK_LOSS:
                        unique_starts, inverse = np.unique(batch_starts, return_inverse=True)
                        inverse_t = torch.from_numpy(inverse.astype(np.int64)).to(device)
                    if USE_CSEC_STANDARDIZATION:
                        # Subtract per-timestep universe mean from targets
                        n_g = len(unique_starts)
                        sums = torch.zeros(n_g, ymh_b.size(1), device=device)
                        counts = torch.zeros(n_g, 1, device=device)
                        sums = sums.index_add(0, inverse_t, ymh_b)
                        counts = counts.index_add(0, inverse_t, torch.ones(ymh_b.size(0), 1, device=device))
                        means = sums / counts.clamp(min=1)
                        ymh_b = ymh_b - means[inverse_t]

                    mh_mean, mh_log_std = model.forward_multi_horizon(xb)
                    loss_mh = PatchTransformer.gaussian_nll(mh_mean, mh_log_std, ymh_b)
                    loss = loss + loss_mh
                    losses_mh.append(float(loss_mh.item()))

                    if USE_RANK_LOSS:
                        # Pairwise margin ranking loss within each timestep group
                        rank_loss = torch.zeros((), device=device)
                        n_groups = 0
                        for g_idx in range(len(unique_starts)):
                            mask = (inverse_t == g_idx)
                            cnt = int(mask.sum().item())
                            if cnt < 2:
                                continue
                            p = mh_mean[mask]      # (cnt, H_horizons)
                            t = ymh_b[mask]        # (cnt, H_horizons)
                            # For each horizon: pairwise margin loss
                            p_diff = p.unsqueeze(0) - p.unsqueeze(1)   # (cnt, cnt, H)
                            t_diff = t.unsqueeze(0) - t.unsqueeze(1)
                            mask_pair = (t_diff > 0).float()
                            loss_pair = torch.relu(RANK_MARGIN - p_diff) * mask_pair
                            denom = mask_pair.sum().clamp(min=1)
                            rank_loss = rank_loss + loss_pair.sum() / denom
                            n_groups += 1
                        if n_groups > 0:
                            rank_loss = rank_loss / n_groups
                            loss = loss + RANK_LOSS_COEF * rank_loss
                            rank_loss_val = float(rank_loss.item())
                            losses_rank.append(rank_loss_val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
            # exp80: live JSONL log every TRAIN_LOG_EVERY batches so charts can be
            # built mid-pretrain (durable on disk, not subprocess stdout).
            batch_idx_print += 1
            if batch_idx_print % TRAIN_LOG_EVERY == 0:
                try:
                    row = {
                        "epoch": ep + 1,
                        "batch": batch_idx_print,
                        "frac_through": float(i / max(1, n - SGD_BATCH)),
                        "nll": float(loss.item()),
                        "nll_running_mean": float(np.mean(losses[-100:])),
                    }
                    if losses_mh:
                        row["mh_nll"] = float(np.mean(losses_mh[-100:]))
                    if losses_rank:
                        row["rank"] = float(np.mean(losses_rank[-100:]))
                    with open(train_log_path, "a") as _fh:
                        _fh.write(_json.dumps(row) + "\n")
                except Exception:
                    pass
        if losses:
            extra = f"  mh_nll={np.mean(losses_mh):.4f}" if losses_mh else ""
            extra += f"  rank={np.mean(losses_rank):.4f}" if losses_rank else ""
            print(f"[pretrain] epoch {ep+1}/{PRETRAIN_EPOCHS}  nll={np.mean(losses):.4f}{extra}", flush=True)


# ============================================================================
# RL PRETRAIN + EVAL — share the same simulation routine
# ============================================================================

ACTION_TO_POS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def simulate(model: PatchTransformer, features: dict[str, pd.DataFrame], device: str,
             learn: bool) -> PaperBroker:
    """Replay merged events through the model, BATCHING all symbols' decisions
    at the same timestamp into a single forward pass.

    Original loop did 1 forward(batch=1) per (sym, ts) event → ~50k tiny GPU
    launches. This version groups events by ts and does 1 forward(batch=N_syms)
    per timestamp → ~10k larger launches. ~5× MPS speedup.

    If learn=True, also accumulates RL transitions and takes periodic
    REINFORCE updates from the buffer.
    """
    portfolio_weight = NOTIONAL_PER_SYMBOL_USD / STARTING_CASH_USD
    round_trip_var_cost = 2.0 * SLIPPAGE_BPS * 1e-4
    fixed_cost_frac = 2.0 * FEE_PER_TRADE_USD / NOTIONAL_PER_SYMBOL_USD

    class _CashTrackingPaperBroker(PaperBroker):
        def __init__(self):
            super().__init__()
            self.cash_curve: list[tuple[pd.Timestamp, float]] = []
        def mark_to_market(self, ts, prices):
            eq = super().mark_to_market(ts, prices)
            self.cash_curve.append((ts, self.cash))
            return eq
    broker = _CashTrackingPaperBroker()
    # last_change_idx tracks the bar index when this symbol's position last changed,
    # so we can enforce PRIMARY_MIN_HOLD_BARS between consecutive position changes.
    sym_state: dict[str, dict] = {s: {"i": -1, "pending": [], "last_pos": 0.0,
                                       "last_change_idx": -10**9} for s in features}

    cols = USE_FEATURES
    feat_arrays: dict[str, np.ndarray] = {s: f[cols].to_numpy(np.float32) for s, f in features.items()}
    close_arrays: dict[str, np.ndarray] = {s: f["close"].to_numpy(np.float32) for s, f in features.items()}
    # exp51: per-symbol SPY 1-min log-returns (forward-filled in featurize), used
    # in the RL reward to compute alpha-vs-SPY over RL_REWARD_HORIZON bars.
    spy_arrays: dict[str, np.ndarray] = {
        s: f["spy_logret_1"].to_numpy(np.float32) if "spy_logret_1" in f.columns
        else np.zeros(len(f), dtype=np.float32)
        for s, f in features.items()
    }

    # Group events by timestamp. One model.forward per group, batch = #symbols ready.
    events_by_ts: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for sym, f in features.items():
        for i, ts in enumerate(f["timestamp"]):
            events_by_ts.setdefault(ts, []).append((sym, i))
    sorted_ts = sorted(events_by_ts.keys())

    opt = torch.optim.AdamW(model.parameters(), lr=RL_LR, weight_decay=0.0) if learn else None
    buf_X: list[np.ndarray] = []
    buf_a: list[int] = []
    buf_r: list[float] = []

    H_max = max(model.pred_horizon, RL_REWARD_HORIZON)
    C = model.context_len
    n_decisions = 0
    ts_count = 0

    for ts in sorted_ts:
        ts_count += 1
        events_here = events_by_ts[ts]

        # Update each symbol's "current bar index" so mark_to_market sees latest prices
        for sym, i_now in events_here:
            sym_state[sym]["i"] = i_now

        # Build batched X for symbols that have enough history at this ts
        batch_X: list[np.ndarray] = []
        batch_meta: list[tuple[str, int]] = []
        for sym, i_now in events_here:
            if i_now >= C - 1:
                batch_X.append(feat_arrays[sym][i_now - C + 1 : i_now + 1])
                batch_meta.append((sym, i_now))

        # ONE forward pass for all symbols ready at this timestamp
        if batch_X:
            with torch.no_grad():
                model.eval()
                xb = torch.from_numpy(np.stack(batch_X)).to(device)
                mean, log_std, alog = model(xb)        # alog: (B, 3)
                if USE_FORECAST_POLICY:
                    # exp28+: action from MULTI-HORIZON head.
                    # FORECAST_HORIZON_IDX:
                    #   >=0 → use that single horizon
                    #   -1  → ensemble: average per-horizon predicted Sharpes
                    if model.horizons_minutes:
                        mh_mean, mh_log_std = model.forward_multi_horizon(xb)
                        if FORECAST_HORIZON_IDX < 0:
                            # Per-horizon Sharpe, then average
                            per_h_sharpe = mh_mean / (torch.exp(mh_log_std) + 1e-12)  # (B, H)
                            pred_sharpe = per_h_sharpe.mean(dim=-1)
                        else:
                            h_mean = mh_mean[:, FORECAST_HORIZON_IDX]
                            h_std = torch.exp(mh_log_std[:, FORECAST_HORIZON_IDX])
                            pred_sharpe = h_mean / (h_std + 1e-12)
                    else:
                        h_mean = mean.sum(dim=-1)
                        h_var = torch.exp(2 * log_std).sum(dim=-1)
                        pred_sharpe = h_mean / torch.sqrt(h_var + 1e-12)
                    # SELL=0, HOLD=1, BUY=2
                    a_idx_t = torch.full_like(pred_sharpe, 1, dtype=torch.long)
                    a_idx_t = torch.where(pred_sharpe > FORECAST_BUY_SHARPE_THRESHOLD,
                                          torch.tensor(2, device=a_idx_t.device), a_idx_t)
                    a_idx_t = torch.where(pred_sharpe < -FORECAST_BUY_SHARPE_THRESHOLD,
                                          torch.tensor(0, device=a_idx_t.device), a_idx_t)
                elif learn:
                    probs = torch.softmax(alog, dim=-1)
                    a_idx_t = torch.distributions.Categorical(probs=probs).sample()
                else:
                    a_idx_t = torch.argmax(alog, dim=-1)
                a_idx_list = a_idx_t.cpu().tolist()
            model.train()

            # Apply each decision (still serial, but only broker bookkeeping — fast)
            for (sym, i_now), a_idx, X_arr in zip(batch_meta, a_idx_list, batch_X):
                st = sym_state[sym]
                # exp26: long-only — collapse SELL (a_idx=0) → HOLD (a_idx=1)
                if LONG_ONLY and a_idx == 0:
                    a_idx = 1
                target = float(ACTION_TO_POS[a_idx])
                # STICKINESS: if not enough bars since last position change, force HOLD
                if (i_now - st["last_change_idx"]) < PRIMARY_MIN_HOLD_BARS:
                    target = st["last_pos"]
                    a_idx = 1   # HOLD
                pos_change = abs(target - st["last_pos"])
                if pos_change > 1e-9:
                    st["last_change_idx"] = i_now
                broker.update(sym, float(close_arrays[sym][i_now]), ts, target)
                st["pending"].append({
                    "X": X_arr.copy(),
                    "a": a_idx, "target": target,
                    "entry": float(close_arrays[sym][i_now]),
                    "i": i_now, "pos_change": pos_change,
                })
                st["last_pos"] = target
                n_decisions += 1

        # mark-to-market once per timestamp (was N times in old loop)
        prices = {s: float(close_arrays[s][st_["i"]]) for s, st_ in sym_state.items() if st_["i"] >= 0}
        broker.mark_to_market(ts, prices)

        # resolve pending whose horizon arrived (all symbols)
        for sym, i_now in events_here:
            st = sym_state[sym]
            close = close_arrays[sym]
            spy_lr = spy_arrays[sym]
            while st["pending"] and st["pending"][0]["i"] + H_max <= i_now:
                p_ = st["pending"].pop(0)
                future_close = float(close[p_["i"] + RL_REWARD_HORIZON])
                log_ret = math.log(max(future_close, 1e-12) / max(p_["entry"], 1e-12))
                cost_charge = 0.5 * (round_trip_var_cost + fixed_cost_frac) * p_["pos_change"]
                pos_ret = p_["target"] * log_ret
                # exp51: SPY-alpha bonus. spy_lr[t+1 : t+1+H] sums to SPY's H-bar
                # log return at this symbol's timestamp. Reward beating it (when long).
                t_after = p_["i"] + 1
                spy_h_ret = float(spy_lr[t_after : t_after + RL_REWARD_HORIZON].sum())
                alpha_bonus = SPY_ALPHA_COEF * (pos_ret - p_["target"] * spy_h_ret)
                reward = portfolio_weight * (pos_ret - cost_charge
                                              - VOL_PENALTY * pos_ret * pos_ret
                                              + alpha_bonus)
                buf_X.append(p_["X"]); buf_a.append(p_["a"]); buf_r.append(reward)

        # Bound replay buffer — trim oldest in batches to avoid per-append cost.
        if learn and len(buf_X) > RL_BUFFER_MAX + 1000:
            buf_X = buf_X[-RL_BUFFER_MAX:]
            buf_a = buf_a[-RL_BUFFER_MAX:]
            buf_r = buf_r[-RL_BUFFER_MAX:]

        # periodic SGD step (counted in timestamps now, not in events)
        if learn and ts_count % RL_STEP_EVERY_BARS == 0 and len(buf_X) >= SGD_BATCH:
            idx = np.random.choice(len(buf_X), size=SGD_BATCH, replace=False)
            Xb = torch.from_numpy(np.stack([buf_X[i] for i in idx])).to(device)
            ab = torch.tensor([buf_a[i] for i in idx], dtype=torch.long, device=device)
            rb = torch.tensor([buf_r[i] for i in idx], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            _, _, logits = model(Xb)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_pi = log_probs.gather(1, ab.unsqueeze(1)).squeeze(1)
            adv = rb - rb.mean()
            ent = -(torch.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
            loss = -(log_pi * adv).mean() - ENTROPY_COEF * ent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

    return broker


# ============================================================================
# SECONDARY STRATEGY: BEST-STOCK PICKER
# ============================================================================
# Different way to USE the model's predictions: at each bar, pick the SINGLE
# best stock to buy (max one buy per cooldown window, fixed $ per buy).
# Sells whole position when model's SELL conviction exceeds threshold.
# Uses its own broker (PickerBroker) so it doesn't interfere with the main
# portfolio simulation — runs in parallel as a second equity curve.
#
# Why interesting:
#   - Different alpha source (concentration on best signal vs spread across all)
#   - Tighter risk per trade ($1k each, vs $1k * many positions)
#   - More similar to a discretionary trader's workflow
#   - Provides a SECOND reward signal that we can later feed back to RL
# ============================================================================

PICKER_POSITION_USD = 1_000.0
PICKER_BUY_COOLDOWN_S = 5 * 60     # max 1 buy per 5 minutes
PICKER_HOLD_BARS = 60              # auto-sell after N bars (1 hour at 1-min bars)
PICKER_MAX_CONCURRENT = 5          # max number of distinct positions held at once

# ============================================================================
# Strategy 3: WEIGHTED — confidence-sized dynamic positions
# Position size = function of (predicted Sharpe, free cash, vol).
# Reserves MIN_CASH_RESERVE_PCT of starting cash for opportunities.
# Up to MAX_NEW_TRADES_PER_TIMESTEP simultaneous buys, each capped at
# MAX_POS_FRACTION_OF_FREE_CASH of free cash.
# ============================================================================
MAX_POS_FRACTION_OF_FREE_CASH = 0.50  # exp47: SWAP + cap 0.50. exp46 (SWAP+0.65) gave best sharpe yet (+1.42) but DD -10.85% over floor on seed 1 only. Drop cap from 0.65 to 0.50 to bring worst-seed DD comfortably under -10%.
MIN_CASH_RESERVE_PCT = 0.81875        # exp157: tighter top2 reserve probe below exp156
MAX_NEW_TRADES_PER_TIMESTEP = 5       # diversify timing
KELLY_SCALE = 0.5                     # half-Kelly (exp33: doubling had no effect — cap saturates)
WEIGHTED_SELL_SHARPE = 0.0            # close any held position whose 1h predicted Sharpe drops below this
WEIGHTED_MIN_TRADE_USD = 100.0        # too small → fee dominates
WEIGHTED_SWAP_MARGIN = 0.15           # exp50: keep 0.15
# exp58: realistic transaction friction (re-applied — was reset by exp57 discard)
VOLUME_IMPACT_BPS_PER_PCT = 50.0      # extra slippage per 1% of bar's $-volume our order represents
VOLUME_IMPACT_MAX_BPS = 200.0         # cap extra slippage at 2%
MAX_BAR_VOLUME_PARTICIPATION = 0.10   # refuse trades > 10% of bar volume


class PickerBroker:
    """Single-name buyer: each buy is a fixed $ amount, throttled by cooldown.

    Buys ADD to the position (can stack multiple buys of same symbol).
    Sells close the entire position. Long-only.
    """
    def __init__(self) -> None:
        self.cash = STARTING_CASH_USD
        self.positions: dict[str, float] = {}        # symbol -> qty
        self.last_prices: dict[str, float] = {}
        self.last_buy_ts: pd.Timestamp | None = None
        self.n_trades = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[tuple[pd.Timestamp, str, str]] = []   # (ts, sym, side)

    def _cooldown_ok(self, ts: pd.Timestamp) -> bool:
        if self.last_buy_ts is None:
            return True
        return (ts - self.last_buy_ts).total_seconds() >= PICKER_BUY_COOLDOWN_S

    def buy(self, sym: str, price: float, ts: pd.Timestamp) -> bool:
        if not self._cooldown_ok(ts):
            return False
        fill_price = price * (1.0 + SLIPPAGE_BPS * 1e-4)
        qty = PICKER_POSITION_USD / max(fill_price, 1e-9)
        cost = qty * fill_price + FEE_PER_TRADE_USD
        if cost > self.cash:
            return False
        slip_cost = qty * (fill_price - price)
        self.cash -= cost
        self.positions[sym] = self.positions.get(sym, 0.0) + qty
        self.last_prices[sym] = price
        self.last_buy_ts = ts
        self.n_trades += 1
        self.total_fees += FEE_PER_TRADE_USD
        self.total_slippage += slip_cost
        self.trades.append((ts, sym, "BUY"))
        return True

    def sell_all(self, sym: str, price: float, ts: pd.Timestamp) -> bool:
        qty = self.positions.get(sym, 0.0)
        if qty <= 1e-9:
            return False
        fill_price = price * (1.0 - SLIPPAGE_BPS * 1e-4)
        proceeds = qty * fill_price - FEE_PER_TRADE_USD
        slip_cost = qty * (price - fill_price)
        self.cash += proceeds
        self.positions[sym] = 0.0
        self.last_prices[sym] = price
        self.n_trades += 1
        self.total_fees += FEE_PER_TRADE_USD
        self.total_slippage += slip_cost
        self.trades.append((ts, sym, "SELL"))
        return True

    def equity(self, prices: dict[str, float]) -> float:
        e = self.cash
        for sym, qty in self.positions.items():
            if qty > 0:
                e += qty * prices.get(sym, self.last_prices.get(sym, 0.0))
        return e

    def mark_to_market(self, ts: pd.Timestamp, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.equity_curve.append((ts, eq))
        if not hasattr(self, "cash_curve"):
            self.cash_curve = []
        self.cash_curve.append((ts, self.cash))
        return eq


def simulate_best_picker(model: PatchTransformer,
                         features: dict[str, pd.DataFrame],
                         device: str) -> PickerBroker:
    """At each timestamp, RANK all ready symbols by P(BUY) (softmax of action head).

    Buy logic: every PICKER_BUY_COOLDOWN_S, buy the TOP-1 ranked symbol's $1k
    position, regardless of absolute level. Pure rank-based — works even when
    HOLD bias makes all P(BUY) values low in absolute terms, because we only
    care which symbol the model likes MOST relative to the others.

    Sell logic: each held position auto-exits after PICKER_HOLD_BARS bars
    (deterministic timer — no model decision needed for exits). This makes
    the picker's behavior independent of SELL-logit calibration.

    Concurrency: max PICKER_MAX_CONCURRENT positions held at once. Beyond
    that, new buys wait for the cooldown AND a freed slot.
    """
    broker = PickerBroker()
    cols = USE_FEATURES
    feat_arrays = {s: f[cols].to_numpy(np.float32) for s, f in features.items()}
    close_arrays = {s: f["close"].to_numpy(np.float32) for s, f in features.items()}

    events_by_ts: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for sym, f in features.items():
        for i, ts in enumerate(f["timestamp"]):
            events_by_ts.setdefault(ts, []).append((sym, i))
    sorted_ts = sorted(events_by_ts.keys())

    C = model.context_len
    last_idx_by_sym: dict[str, int] = {s: -1 for s in features}
    # Track when each held position was bought (bar count since start of replay)
    bought_at_bar: dict[str, int] = {}   # sym -> bar count
    bar_count = 0

    for ts in sorted_ts:
        bar_count += 1
        events_here = events_by_ts[ts]
        for sym, i_now in events_here:
            last_idx_by_sym[sym] = i_now

        # 1) Auto-sell positions held longer than PICKER_HOLD_BARS
        to_close = []
        for sym, bought_bar in bought_at_bar.items():
            if (bar_count - bought_bar) >= PICKER_HOLD_BARS and broker.positions.get(sym, 0.0) > 0:
                to_close.append(sym)
        for sym in to_close:
            price = float(close_arrays[sym][last_idx_by_sym[sym]])
            if broker.sell_all(sym, price, ts):
                bought_at_bar.pop(sym, None)

        # 2) Score every ready symbol; buy top-ranked if cooldown + slot available
        batch_X, batch_meta = [], []
        for sym, i_now in events_here:
            if i_now >= C - 1:
                batch_X.append(feat_arrays[sym][i_now - C + 1 : i_now + 1])
                batch_meta.append((sym, i_now))

        if batch_X and broker._cooldown_ok(ts):
            n_active = sum(1 for q in broker.positions.values() if q > 0)
            if n_active < PICKER_MAX_CONCURRENT:
                with torch.no_grad():
                    model.eval()
                    xb = torch.from_numpy(np.stack(batch_X)).to(device)
                    _, _, alog = model(xb)   # (B, 3): SELL=0, HOLD=1, BUY=2
                    # Rank by softmax P(BUY) — relative scoring, robust to HOLD bias
                    p_buy = torch.softmax(alog, dim=-1)[:, 2].cpu().tolist()
                model.train()
                best_idx = max(range(len(p_buy)), key=lambda k: p_buy[k])
                sym, i_now = batch_meta[best_idx]
                price = float(close_arrays[sym][i_now])
                if broker.buy(sym, price, ts):
                    bought_at_bar[sym] = bar_count

        # Mark to market every bar
        prices = {s: float(close_arrays[s][last_idx_by_sym[s]])
                  for s in features if last_idx_by_sym[s] >= 0}
        broker.mark_to_market(ts, prices)

    return broker


# ============================================================================
# Strategy 3: WEIGHTED — confidence-sized dynamic positions (exp32)
# ============================================================================

class WeightedBroker:
    """Position-sized by % of free cash. Reserves min cash for opportunities.

    No NOTIONAL_PER_SYMBOL constraint — buy in arbitrary $ amounts.
    Long-only. Sells close entire position.
    """
    def __init__(self, starting_cash: float, min_reserve_frac: float = 0.10) -> None:
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.min_reserve_usd = starting_cash * min_reserve_frac
        self.positions: dict[str, float] = {}        # symbol -> qty
        self.last_prices: dict[str, float] = {}
        self.n_trades = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[tuple[pd.Timestamp, str, str]] = []

    def free_cash(self) -> float:
        """Cash available for new buys after subtracting reserve."""
        return max(0.0, self.cash - self.min_reserve_usd)

    def buy_usd(self, sym: str, price: float, ts: pd.Timestamp, dollar_amount: float,
                bar_dollar_volume: float = 0.0) -> bool:
        """Buy dollar_amount worth of sym at price. Returns True if executed.
        exp58: bar_dollar_volume enables liquidity gate + market-impact slippage."""
        if dollar_amount < WEIGHTED_MIN_TRADE_USD:
            return False
        if dollar_amount > self.free_cash():
            return False
        if bar_dollar_volume > 0:
            if dollar_amount / bar_dollar_volume > MAX_BAR_VOLUME_PARTICIPATION:
                return False
        impact_bps = 0.0
        if bar_dollar_volume > 0:
            participation_pct = (dollar_amount / bar_dollar_volume) * 100.0
            impact_bps = min(participation_pct * VOLUME_IMPACT_BPS_PER_PCT, VOLUME_IMPACT_MAX_BPS)
        total_slip_bps = SLIPPAGE_BPS + impact_bps
        fill_price = price * (1.0 + total_slip_bps * 1e-4)
        qty = dollar_amount / max(fill_price, 1e-9)
        cost = qty * fill_price + FEE_PER_TRADE_USD
        if cost > self.cash:
            return False
        slip_cost = qty * (fill_price - price)
        self.cash -= cost
        self.positions[sym] = self.positions.get(sym, 0.0) + qty
        self.last_prices[sym] = price
        self.n_trades += 1
        self.total_fees += FEE_PER_TRADE_USD
        self.total_slippage += slip_cost
        self.trades.append((ts, sym, "BUY"))
        return True

    def sell_all(self, sym: str, price: float, ts: pd.Timestamp,
                 bar_dollar_volume: float = 0.0) -> bool:
        """exp58: bar_dollar_volume enables liquidity gate + impact slippage on sell side."""
        qty = self.positions.get(sym, 0.0)
        if qty <= 1e-9:
            return False
        sell_dollar = qty * price
        if bar_dollar_volume > 0:
            if sell_dollar / bar_dollar_volume > MAX_BAR_VOLUME_PARTICIPATION:
                return False
        impact_bps = 0.0
        if bar_dollar_volume > 0:
            participation_pct = (sell_dollar / bar_dollar_volume) * 100.0
            impact_bps = min(participation_pct * VOLUME_IMPACT_BPS_PER_PCT, VOLUME_IMPACT_MAX_BPS)
        total_slip_bps = SLIPPAGE_BPS + impact_bps
        fill_price = price * (1.0 - total_slip_bps * 1e-4)
        proceeds = qty * fill_price - FEE_PER_TRADE_USD
        slip_cost = qty * (price - fill_price)
        self.cash += proceeds
        self.positions[sym] = 0.0
        self.last_prices[sym] = price
        self.n_trades += 1
        self.total_fees += FEE_PER_TRADE_USD
        self.total_slippage += slip_cost
        self.trades.append((ts, sym, "SELL"))
        return True

    def equity(self, prices: dict[str, float]) -> float:
        e = self.cash
        for sym, qty in self.positions.items():
            if qty > 0:
                e += qty * prices.get(sym, self.last_prices.get(sym, 0.0))
        return e

    def mark_to_market(self, ts: pd.Timestamp, prices: dict[str, float]) -> float:
        eq = self.equity(prices)
        self.equity_curve.append((ts, eq))
        if not hasattr(self, "cash_curve"):
            self.cash_curve = []
        self.cash_curve.append((ts, self.cash))
        return eq


def precompute_predictions(
    model: PatchTransformer,
    features: dict[str, pd.DataFrame],
    device: str,
    *,
    batch_size: int = 4096,
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray]]:
    """exp66 SPEEDUP: precompute model.forward_multi_horizon outputs for every
    valid (symbol, bar_index) pair. Returns {(sym, i_now): (mh_mean[H], mh_log_std[H])}.
    Eliminates ~25k×N_simulators of repeated GPU launches per iteration.
    Memory: ~95 sym × 25k bars × 4 horizons × 2 × 4 bytes ≈ 76 MB.
    """
    preds: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    if not getattr(model, "horizons_minutes", None):
        return preds
    cols = USE_FEATURES
    C = model.context_len
    all_X: list[np.ndarray] = []
    all_meta: list[tuple[str, int]] = []
    for sym, f in features.items():
        arr = f[cols].to_numpy(np.float32)
        n = len(arr)
        for i_now in range(C - 1, n):
            all_X.append(arr[i_now - C + 1 : i_now + 1])
            all_meta.append((sym, i_now))
    if not all_X:
        return preds
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(all_X), batch_size):
            chunk = all_X[i : i + batch_size]
            chunk_meta = all_meta[i : i + batch_size]
            xb = torch.from_numpy(np.stack(chunk)).to(device)
            mh_mean, mh_log_std = model.forward_multi_horizon(xb)
            mh_mean_np = mh_mean.cpu().numpy()
            mh_log_std_np = mh_log_std.cpu().numpy()
            for j, key in enumerate(chunk_meta):
                preds[key] = (mh_mean_np[j], mh_log_std_np[j])
    model.train()
    print(f"[precompute] {len(preds):,} (sym,bar) predictions in {time.time()-t0:.1f}s", flush=True)
    return preds


def _lookup_mh(
    events: list[tuple[str, int]],
    precomputed: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    C: int,
) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
    """Filter events to those with valid context AND a precomputed prediction;
    return (kept_meta, mh_mean[B,H], mh_log_std[B,H]) as numpy arrays."""
    kept: list[tuple[str, int]] = []
    means: list[np.ndarray] = []
    log_stds: list[np.ndarray] = []
    for s, i in events:
        if i < C - 1:
            continue
        v = precomputed.get((s, i))
        if v is None:
            continue
        kept.append((s, i))
        means.append(v[0])
        log_stds.append(v[1])
    if not kept:
        return [], np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    return kept, np.stack(means), np.stack(log_stds)


def simulate_weighted(model: PatchTransformer,
                      features: dict[str, pd.DataFrame],
                      device: str,
                      precomputed_preds: dict | None = None) -> WeightedBroker:
    """Confidence-weighted dynamic-sizing strategy.

    For each timestep:
      1. Predict 1h-horizon Sharpe for each ready symbol via mh_head.
      2. SELL pass: close any held position whose Sharpe < WEIGHTED_SELL_SHARPE.
      3. SWAP pass (exp47): rotate weakest held → strongest unheld when the
         pred_sharpe edge exceeds WEIGHTED_SWAP_MARGIN (covers round-trip cost).
      4. BUY pass: for symbols with positive predicted Sharpe (not already held),
         compute Kelly-like dollar size and execute up to MAX_NEW_TRADES_PER_TIMESTEP.
    No time-based exit — the model decides when to sell.
    """
    broker = WeightedBroker(STARTING_CASH_USD, min_reserve_frac=MIN_CASH_RESERVE_PCT)
    cols = USE_FEATURES
    feat_arrays = {s: f[cols].to_numpy(np.float32) for s, f in features.items()}
    close_arrays = {s: f["close"].to_numpy(np.float32) for s, f in features.items()}
    # exp58: per-bar dollar-volume for liquidity-aware slippage
    volume_arrays = {
        s: (f["volume"].to_numpy(np.float32) if "volume" in f.columns
            else np.zeros(len(f), dtype=np.float32))
        for s, f in features.items()
    }

    events_by_ts: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for sym, f in features.items():
        for i, ts in enumerate(f["timestamp"]):
            events_by_ts.setdefault(ts, []).append((sym, i))
    sorted_ts = sorted(events_by_ts.keys())

    C = model.context_len
    last_idx_by_sym: dict[str, int] = {s: -1 for s in features}

    for ts in sorted_ts:
        events_here = events_by_ts[ts]
        for sym, i_now in events_here:
            last_idx_by_sym[sym] = i_now

        # Get predictions for all ready symbols at this timestamp (one batched forward).
        batch_X, batch_meta = [], []
        for sym, i_now in events_here:
            if i_now >= C - 1:
                batch_X.append(feat_arrays[sym][i_now - C + 1 : i_now + 1])
                batch_meta.append((sym, i_now))

        if batch_X and model.horizons_minutes:
            if precomputed_preds is not None:
                batch_meta, mh_mean_np, mh_log_std_np = _lookup_mh(events_here, precomputed_preds, C)
                if not batch_meta:
                    pred_sharpe_list = []
                else:
                    h_mean = mh_mean_np[:, 1]
                    h_std = np.exp(mh_log_std_np[:, 1])
                    pred_sharpe_list = (h_mean / (h_std + 1e-12)).tolist()
            else:
                with torch.no_grad():
                    model.eval()
                    xb = torch.from_numpy(np.stack(batch_X)).to(device)
                    mh_mean, mh_log_std = model.forward_multi_horizon(xb)
                    h_mean = mh_mean[:, 1]   # 1h horizon
                    h_std = torch.exp(mh_log_std[:, 1])
                    pred_sharpe_list = (h_mean / (h_std + 1e-12)).cpu().tolist()
                model.train()
            sym_to_sharpe = {sym: ps for (sym, _), ps in zip(batch_meta, pred_sharpe_list)}

            # 1) SELL pass — close held positions whose 1h Sharpe dropped below threshold
            for sym, ps in sym_to_sharpe.items():
                if broker.positions.get(sym, 0.0) > 0 and ps < WEIGHTED_SELL_SHARPE:
                    i_now = last_idx_by_sym[sym]
                    px = float(close_arrays[sym][i_now])
                    bar_dv = float(volume_arrays[sym][i_now]) * px
                    broker.sell_all(sym, px, ts, bar_dollar_volume=bar_dv)

            # 1.5) SWAP pass (exp47) — rotate weakest held → strongest unheld
            # only when the pred_sharpe edge clears WEIGHTED_SWAP_MARGIN. Covers
            # the round-trip transaction cost (fee + 2bps slippage × 2) so we
            # only churn when relative model conviction is meaningful.
            held_with_sharpe = [
                (sym, sym_to_sharpe[sym])
                for sym, qty in broker.positions.items()
                if qty > 0 and sym in sym_to_sharpe
            ]
            unheld_with_sharpe = [
                (sym, ps) for (sym, _), ps in zip(batch_meta, pred_sharpe_list)
                if broker.positions.get(sym, 0.0) <= 0 and ps > 0
            ]
            if held_with_sharpe and unheld_with_sharpe:
                weak_sym, weak_ps = min(held_with_sharpe, key=lambda t: t[1])
                strong_sym, strong_ps = max(unheld_with_sharpe, key=lambda t: t[1])
                if (strong_ps - weak_ps) > WEIGHTED_SWAP_MARGIN:
                    i_now_weak = last_idx_by_sym[weak_sym]
                    px_w = float(close_arrays[weak_sym][i_now_weak])
                    bar_dv_w = float(volume_arrays[weak_sym][i_now_weak]) * px_w
                    broker.sell_all(weak_sym, px_w, ts, bar_dollar_volume=bar_dv_w)
                    # BUY pass below will deploy the freed cash to strong_sym
                    # since it's still in pred_sharpe_list with positive ps.

            # 2) BUY pass — for symbols not currently held, size by Kelly
            free_cash_now = broker.free_cash()
            candidates = []
            for (sym, i_now), ps in zip(batch_meta, pred_sharpe_list):
                if ps <= 0:
                    continue
                if broker.positions.get(sym, 0.0) > 0:
                    continue   # already long, don't double up
                base_frac = min(ps * KELLY_SCALE, MAX_POS_FRACTION_OF_FREE_CASH)
                usd_size = base_frac * free_cash_now
                if usd_size >= WEIGHTED_MIN_TRADE_USD:
                    candidates.append((sym, i_now, usd_size, ps))

            candidates.sort(key=lambda t: -t[2])
            candidates = candidates[:MAX_NEW_TRADES_PER_TIMESTEP]

            for sym, i_now, suggested_usd, ps in candidates:
                cap = MAX_POS_FRACTION_OF_FREE_CASH * broker.free_cash()
                actual = min(suggested_usd, cap)
                if actual < WEIGHTED_MIN_TRADE_USD:
                    continue
                price = float(close_arrays[sym][i_now])
                bar_dv = float(volume_arrays[sym][i_now]) * price
                broker.buy_usd(sym, price, ts, actual, bar_dollar_volume=bar_dv)

        prices = {s: float(close_arrays[s][last_idx_by_sym[s]])
                  for s in features if last_idx_by_sym[s] >= 0}
        broker.mark_to_market(ts, prices)

    return broker


# ============================================================================
# MULTI-TRADER-PROFILE SIMULATORS — same model, different time horizons
# ============================================================================
# Profiles share the trained model but differ in:
#   - which horizon prediction they consult
#   - max-hold rule (force-exit when held too long)
#   - selection threshold (absolute pred_sharpe OR rank-percentile)
# Plus a passive top-N picker (no rotation, just pick + hold) and SPY benchmark.

def simulate_profile(
    model: PatchTransformer,
    features: dict[str, pd.DataFrame],
    device: str,
    *,
    horizon_idx: int,
    max_hold_bars: int,
    buy_threshold: float = 0.0,
    sell_threshold: float = 0.0,
    rank_percentile: float = 0.0,   # exp62: if >0, only buy when pred_sharpe is in top (1-rank_pct) of timestep
    name: str = "profile",
    precomputed_preds: dict | None = None,
    cooldown_bars_per_sym: int = 0,    # exp80: min bars between trades on the same symbol (0 = no cap)
) -> WeightedBroker:
    broker = WeightedBroker(STARTING_CASH_USD, min_reserve_frac=MIN_CASH_RESERVE_PCT)
    cols = USE_FEATURES
    feat_arrays = {s: f[cols].to_numpy(np.float32) for s, f in features.items()}
    close_arrays = {s: f["close"].to_numpy(np.float32) for s, f in features.items()}
    volume_arrays = {
        s: (f["volume"].to_numpy(np.float32) if "volume" in f.columns
            else np.zeros(len(f), dtype=np.float32))
        for s, f in features.items()
    }
    events_by_ts: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for sym, f in features.items():
        for i, ts in enumerate(f["timestamp"]):
            events_by_ts.setdefault(ts, []).append((sym, i))
    sorted_ts = sorted(events_by_ts.keys())
    C = model.context_len
    last_idx_by_sym: dict[str, int] = {s: -1 for s in features}
    bar_count = 0
    bought_at_bar: dict[str, int] = {}
    for ts in sorted_ts:
        bar_count += 1
        events_here = events_by_ts[ts]
        for sym, i_now in events_here:
            last_idx_by_sym[sym] = i_now
        for held_sym, q in list(broker.positions.items()):
            if q > 0 and (bar_count - bought_at_bar.get(held_sym, 0)) >= max_hold_bars:
                i_now_h = last_idx_by_sym[held_sym]
                if i_now_h >= 0:
                    px = float(close_arrays[held_sym][i_now_h])
                    bar_dv = float(volume_arrays[held_sym][i_now_h]) * px
                    if broker.sell_all(held_sym, px, ts, bar_dollar_volume=bar_dv):
                        bought_at_bar.pop(held_sym, None)
        batch_X, batch_meta = [], []
        for sym, i_now in events_here:
            if i_now >= C - 1:
                batch_X.append(feat_arrays[sym][i_now - C + 1 : i_now + 1])
                batch_meta.append((sym, i_now))
        if batch_X and model.horizons_minutes:
            if precomputed_preds is not None:
                batch_meta, mh_mean_np, mh_log_std_np = _lookup_mh(events_here, precomputed_preds, C)
                if not batch_meta:
                    pred_sharpe_list = []
                else:
                    hi = min(horizon_idx, mh_mean_np.shape[1] - 1)
                    h_mean = mh_mean_np[:, hi]
                    h_std = np.exp(mh_log_std_np[:, hi])
                    pred_sharpe_list = (h_mean / (h_std + 1e-12)).tolist()
            else:
                with torch.no_grad():
                    model.eval()
                    xb = torch.from_numpy(np.stack(batch_X)).to(device)
                    mh_mean, mh_log_std = model.forward_multi_horizon(xb)
                    hi = min(horizon_idx, mh_mean.size(1) - 1)
                    h_mean = mh_mean[:, hi]
                    h_std = torch.exp(mh_log_std[:, hi])
                    pred_sharpe_list = (h_mean / (h_std + 1e-12)).cpu().tolist()
                model.train()
            sym_to_sharpe = {sym: ps for (sym, _), ps in zip(batch_meta, pred_sharpe_list)}
            # SELL pass — close held when below sell_threshold
            for sym, ps in sym_to_sharpe.items():
                if broker.positions.get(sym, 0.0) > 0 and ps < sell_threshold:
                    i_now = last_idx_by_sym[sym]
                    px = float(close_arrays[sym][i_now])
                    bar_dv = float(volume_arrays[sym][i_now]) * px
                    if broker.sell_all(sym, px, ts, bar_dollar_volume=bar_dv):
                        bought_at_bar.pop(sym, None)
            # exp62: optional rank-percentile gate
            rank_cutoff = None
            if rank_percentile > 0 and pred_sharpe_list:
                import numpy as _np
                rank_cutoff = float(_np.quantile(_np.array(pred_sharpe_list), rank_percentile))
            free_cash_now = broker.free_cash()
            candidates = []
            for (sym, i_now), ps in zip(batch_meta, pred_sharpe_list):
                if ps <= buy_threshold:
                    continue
                if rank_cutoff is not None and ps < rank_cutoff:
                    continue
                if broker.positions.get(sym, 0.0) > 0:
                    continue
                # exp80: per-symbol cooldown — refuse to re-trade SYM until cooldown_bars_per_sym elapsed
                last_traded = bought_at_bar.get(sym, -10**9)
                if cooldown_bars_per_sym > 0 and (bar_count - last_traded) < cooldown_bars_per_sym:
                    continue
                base_frac = min(ps * KELLY_SCALE, MAX_POS_FRACTION_OF_FREE_CASH)
                usd_size = base_frac * free_cash_now
                if usd_size >= WEIGHTED_MIN_TRADE_USD:
                    candidates.append((sym, i_now, usd_size, ps))
            candidates.sort(key=lambda t: -t[3])
            candidates = candidates[:MAX_NEW_TRADES_PER_TIMESTEP]
            for sym, i_now, suggested_usd, ps in candidates:
                cap = MAX_POS_FRACTION_OF_FREE_CASH * broker.free_cash()
                actual = min(suggested_usd, cap)
                if actual < WEIGHTED_MIN_TRADE_USD:
                    continue
                price = float(close_arrays[sym][i_now])
                bar_dv = float(volume_arrays[sym][i_now]) * price
                if broker.buy_usd(sym, price, ts, actual, bar_dollar_volume=bar_dv):
                    bought_at_bar[sym] = bar_count
        prices = {s: float(close_arrays[s][last_idx_by_sym[s]])
                  for s in features if last_idx_by_sym[s] >= 0}
        broker.mark_to_market(ts, prices)
    return broker


def simulate_passive_topn(
    model: PatchTransformer,
    features: dict[str, pd.DataFrame],
    device: str,
    *,
    top_n: int = 10,
    ranking_horizons: tuple = (4, 8, 10),
    name: str = "topn",
    precomputed_preds: dict | None = None,
    rank_vs_spy: bool = False,
) -> WeightedBroker:
    """Pick top-N at first ready bar, equal-weight buy, hold to end."""
    broker = WeightedBroker(STARTING_CASH_USD, min_reserve_frac=MIN_CASH_RESERVE_PCT)
    cols = USE_FEATURES
    feat_arrays = {s: f[cols].to_numpy(np.float32) for s, f in features.items()}
    close_arrays = {s: f["close"].to_numpy(np.float32) for s, f in features.items()}
    volume_arrays = {
        s: (f["volume"].to_numpy(np.float32) if "volume" in f.columns
            else np.zeros(len(f), dtype=np.float32))
        for s, f in features.items()
    }
    events_by_ts: dict[pd.Timestamp, list[tuple[str, int]]] = {}
    for sym, f in features.items():
        for i, ts in enumerate(f["timestamp"]):
            events_by_ts.setdefault(ts, []).append((sym, i))
    sorted_ts = sorted(events_by_ts.keys())
    C = model.context_len
    last_idx_by_sym: dict[str, int] = {s: -1 for s in features}
    picked = False
    for ts in sorted_ts:
        events_here = events_by_ts[ts]
        for sym, i_now in events_here:
            last_idx_by_sym[sym] = i_now
        if not picked:
            ready = [(sym, i_now) for sym, i_now in events_here if i_now >= C - 1]
            if len(ready) >= max(top_n, len(features) // 4):
                if precomputed_preds is not None and model.horizons_minutes:
                    kept, mh_mean_np, mh_log_std_np = _lookup_mh(ready, precomputed_preds, C)
                    if not kept:
                        scores = []
                        ready = []
                    else:
                        ready = kept
                        scores_np = np.zeros(mh_mean_np.shape[0], dtype=np.float32)
                        spy_sharpe = np.zeros(mh_mean_np.shape[1], dtype=np.float32)
                        if rank_vs_spy:
                            for row_idx, (sym, _i_now) in enumerate(ready):
                                if sym == "SPY":
                                    spy_sharpe = mh_mean_np[row_idx] / (np.exp(mh_log_std_np[row_idx]) + 1e-12)
                                    break
                        for hi in ranking_horizons:
                            if hi >= mh_mean_np.shape[1]:
                                continue
                            mu = mh_mean_np[:, hi]
                            sd = np.exp(mh_log_std_np[:, hi])
                            scores_np += (mu / (sd + 1e-12)) - spy_sharpe[hi]
                        scores = scores_np.tolist()
                else:
                    batch_X = [feat_arrays[sym][i_now - C + 1 : i_now + 1] for sym, i_now in ready]
                    with torch.no_grad():
                        model.eval()
                        xb = torch.from_numpy(np.stack(batch_X)).to(device)
                        if model.horizons_minutes:
                            mh_mean, mh_log_std = model.forward_multi_horizon(xb)
                            scores = None
                            spy_sharpe = None
                            if rank_vs_spy:
                                for row_idx, (sym, _i_now) in enumerate(ready):
                                    if sym == "SPY":
                                        spy_sharpe = mh_mean[row_idx] / (torch.exp(mh_log_std[row_idx]) + 1e-12)
                                        break
                            for hi in ranking_horizons:
                                if hi >= mh_mean.size(1):
                                    continue
                                mu = mh_mean[:, hi]
                                sd = torch.exp(mh_log_std[:, hi])
                                s = mu / (sd + 1e-12)
                                if spy_sharpe is not None:
                                    s = s - spy_sharpe[hi]
                                scores = s if scores is None else scores + s
                            scores = scores.cpu().tolist() if scores is not None else [0.0] * len(ready)
                        else:
                            scores = [0.0] * len(ready)
                    model.train()
                ranked = sorted(zip(ready, scores), key=lambda r: -r[1])
                top = ranked[:top_n]
                if top:
                    per_pos = broker.free_cash() / len(top)
                    for (sym, i_now), score in top:
                        if per_pos < WEIGHTED_MIN_TRADE_USD:
                            continue
                        price = float(close_arrays[sym][i_now])
                        bar_dv = float(volume_arrays[sym][i_now]) * price
                        broker.buy_usd(sym, price, ts, per_pos, bar_dollar_volume=bar_dv)
                picked = True
        prices = {s: float(close_arrays[s][last_idx_by_sym[s]])
                  for s in features if last_idx_by_sym[s] >= 0}
        broker.mark_to_market(ts, prices)
    return broker


def simulate_buyhold_spy(features: dict[str, pd.DataFrame]) -> WeightedBroker:
    """Passive baseline: invest all (less reserve) into SPY at first bar, hold."""
    broker = WeightedBroker(STARTING_CASH_USD, min_reserve_frac=MIN_CASH_RESERVE_PCT)
    if not features:
        return broker
    sym = "SPY" if "SPY" in features else next(iter(features))
    f = features[sym]
    closes = f["close"].to_numpy(np.float32)
    volumes = f["volume"].to_numpy(np.float32) if "volume" in f.columns else None
    timestamps = f["timestamp"].tolist()
    if len(closes) < 2:
        return broker
    first_ts = timestamps[0]
    first_px = float(closes[0])
    # exp75: SPY buy-hold is a reference benchmark — bypass the liquidity gate
    # by passing 0.0. Also subtract FEE_PER_TRADE_USD headroom from the order
    # so qty*price + fee ≤ cash (the broker rejects when cost > cash even by $1).
    target_usd = max(0.0, broker.free_cash() - FEE_PER_TRADE_USD - 1.0)
    broker.buy_usd(sym, first_px, first_ts, target_usd, bar_dollar_volume=0.0)
    for i, ts in enumerate(timestamps):
        broker.mark_to_market(ts, {sym: float(closes[i])})
    return broker


# Profile presets. horizon_idx into HORIZONS_MINUTES = [5,60,120,240,390,780,1170,1560,1950,5460,11700]
# 7th field (exp80) = cooldown_bars_per_sym — minimum bars between trades of the same symbol.
PROFILE_PRESETS = [
    # (name, horizon_idx, max_hold_bars, buy_threshold, sell_threshold, rank_percentile, cooldown_bars_per_sym)
    ("intraday",   2, 390,         0.0, 0.0, 0.0, 0),
    ("intraweek",  8, 5 * 390,     0.0, 0.0, 0.0, 0),
    ("intramonth", 10, 30 * 390,   0.0, 0.0, 0.0, 0),
    ("longterm",   10, 10**9,      0.0, 0.0, 0.0, 0),
    # exp80: per-symbol trade-frequency caps — at most one buy per symbol per [day|week|month].
    ("daily_capped",   4, 5 * 390,   0.0, 0.0, 0.0, 390),         # 1 trade/sym/day,   1d horizon
    ("weekly_capped",  8, 30 * 390,  0.0, 0.0, 0.0, 5 * 390),     # 1 trade/sym/week,  5d horizon
    ("monthly_capped", 10, 10**9,    0.0, 0.0, 0.0, 30 * 390),    # 1 trade/sym/month, 30d horizon
]
PASSIVE_TOPN_VARIANTS = [(1,), (3,), (4,), (5,), (10,), (20,)]   # exp89: add top4 between unstable top3 and stable top5


def run_profile_suite(model, eval_feat, device, seed, precomputed_preds=None):
    """exp61+: run all 8 strategies and dump results JSON for the driver.
    exp66: accept precomputed predictions so each profile's per-bar inference is a dict lookup.
    """
    profile_results: dict = {}
    try:
        from prepare import sharpe_ratio as _sr, max_drawdown_pct as _dd
        for preset in PROFILE_PRESETS:
            # exp80: 7th field is optional cooldown_bars_per_sym
            if len(preset) >= 7:
                pname, hidx, max_hold, buy_t, sell_t, rank_pct, cooldown_b = preset
            else:
                pname, hidx, max_hold, buy_t, sell_t, rank_pct = preset
                cooldown_b = 0
            try:
                pb = simulate_profile(
                    model, eval_feat, device,
                    horizon_idx=hidx, max_hold_bars=max_hold,
                    buy_threshold=buy_t, sell_threshold=sell_t,
                    rank_percentile=rank_pct, name=pname,
                    precomputed_preds=precomputed_preds,
                    cooldown_bars_per_sym=cooldown_b,
                )
                end_eq = float(pb.equity_curve[-1][1]) if pb.equity_curve else 0.0
                start_eq = float(pb.equity_curve[0][1]) if pb.equity_curve else float(STARTING_CASH_USD)
                profile_results[pname] = {
                    "sharpe": float(_sr(pb.equity_curve)),
                    "pnl": end_eq - start_eq,
                    "pnl_pct": (end_eq - start_eq) / start_eq * 100 if start_eq else 0.0,
                    "trades": int(pb.n_trades), "dd_pct": float(_dd(pb.equity_curve)),
                    "ending_equity": end_eq, "horizon_minutes": HORIZONS_MINUTES[hidx],
                    "max_hold_bars": int(max_hold) if max_hold < 10**8 else None,
                }
                print(f"[prof-{pname}] sh={profile_results[pname]['sharpe']:+.3f} "
                      f"pnl=${profile_results[pname]['pnl']:+,.2f} trades={profile_results[pname]['trades']}", flush=True)
            except Exception as e:
                print(f"[prof-{pname}] failed: {e}", flush=True)
        for (n,) in PASSIVE_TOPN_VARIANTS:
            try:
                pb = simulate_passive_topn(model, eval_feat, device, top_n=n, name=f"top{n}",
                                           precomputed_preds=precomputed_preds)
                end_eq = float(pb.equity_curve[-1][1]) if pb.equity_curve else 0.0
                start_eq = float(pb.equity_curve[0][1]) if pb.equity_curve else float(STARTING_CASH_USD)
                profile_results[f"top{n}_picker"] = {
                    "sharpe": float(_sr(pb.equity_curve)),
                    "pnl": end_eq - start_eq,
                    "pnl_pct": (end_eq - start_eq) / start_eq * 100 if start_eq else 0.0,
                    "trades": int(pb.n_trades), "dd_pct": float(_dd(pb.equity_curve)),
                    "ending_equity": end_eq, "horizon_minutes": 0,
                }
                print(f"[prof-top{n}] sh={profile_results[f'top{n}_picker']['sharpe']:+.3f} "
                      f"pnl=${profile_results[f'top{n}_picker']['pnl']:+,.2f}", flush=True)
            except Exception as e:
                print(f"[prof-top{n}] failed: {e}", flush=True)
        try:
            spy = simulate_buyhold_spy(eval_feat)
            end_eq = float(spy.equity_curve[-1][1]) if spy.equity_curve else 0.0
            start_eq = float(spy.equity_curve[0][1]) if spy.equity_curve else float(STARTING_CASH_USD)
            profile_results["spy_buyhold"] = {
                "sharpe": float(_sr(spy.equity_curve)),
                "pnl": end_eq - start_eq,
                "pnl_pct": (end_eq - start_eq) / start_eq * 100 if start_eq else 0.0,
                "trades": int(spy.n_trades), "dd_pct": float(_dd(spy.equity_curve)),
                "ending_equity": end_eq,
            }
            print(f"[prof-spy] sh={profile_results['spy_buyhold']['sharpe']:+.3f} "
                  f"pnl=${profile_results['spy_buyhold']['pnl']:+,.2f}", flush=True)
        except Exception as e:
            print(f"[prof-spy] failed: {e}", flush=True)
    except Exception as e:
        print(f"[prof-suite] outer failure: {e}", flush=True)
    try:
        import json as _json
        prof_path = CHECKPOINT_DIR / f"last_seed{seed}_profiles.json"
        prof_path.write_text(_json.dumps(profile_results, indent=2, default=str))
    except Exception as e:
        print(f"[prof-dump] seed {seed} failed: {e}", flush=True)
    # exp80: also dump per-profile equity curves for the multi-profile chart.
    # Subsamples every 30 minutes to keep file size sane.
    try:
        import json as _json
        curves: dict = {}
        def _dump_curve(name: str, broker):
            if not broker.equity_curve:
                return
            pts = broker.equity_curve
            stride = max(1, len(pts) // 600)
            curves[name] = [(str(t), float(v)) for (t, v) in pts[::stride]]
        # re-run profile sims to capture full curves (cheap with precomputed_preds)
        for preset in PROFILE_PRESETS:
            if len(preset) >= 7:
                pname, hidx, max_hold, buy_t, sell_t, rank_pct, cooldown_b = preset
            else:
                pname, hidx, max_hold, buy_t, sell_t, rank_pct = preset
                cooldown_b = 0
            try:
                pb = simulate_profile(model, eval_feat, device,
                                      horizon_idx=hidx, max_hold_bars=max_hold,
                                      buy_threshold=buy_t, sell_threshold=sell_t,
                                      rank_percentile=rank_pct, name=pname,
                                      precomputed_preds=precomputed_preds,
                                      cooldown_bars_per_sym=cooldown_b)
                _dump_curve(pname, pb)
            except Exception:
                pass
        for (n,) in PASSIVE_TOPN_VARIANTS:
            try:
                pb = simulate_passive_topn(model, eval_feat, device, top_n=n, name=f"top{n}",
                                           precomputed_preds=precomputed_preds)
                _dump_curve(f"top{n}_picker", pb)
            except Exception:
                pass
        try:
            spy = simulate_buyhold_spy(eval_feat)
            _dump_curve("spy_buyhold", spy)
        except Exception:
            pass
        curves_path = CHECKPOINT_DIR / f"last_seed{seed}_profile_curves.json"
        curves_path.write_text(_json.dumps(curves, default=str))
    except Exception as e:
        print(f"[prof-curves-dump] seed {seed} failed: {e}", flush=True)


# ============================================================================
# CONTRACT — DO NOT change the signature
# ============================================================================

def train_and_eval(seed: int = 0) -> tuple:
    """Returns (equity_curve, n_trades, total_fees, total_slippage)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()

    bars_by_sym = prepare_all()
    # exp56: extend universe with EXTENDED_UNIVERSE names whose cache exists.
    # fetch_bars is cache-aware; if download script hasn't fetched a name yet
    # it'll either download (slow) or fail — we silently skip failures.
    from prepare import _cache_path
    extended_added = 0
    for sym in EXTENDED_UNIVERSE:
        if sym in bars_by_sym:
            continue
        if _cache_path(sym).exists():
            try:
                bars_by_sym[sym] = fetch_bars(sym)
                extended_added += 1
            except Exception as e:
                print(f"[extended] {sym}: skipped ({e})", flush=True)
    print(f"[experiment] universe size: {len(bars_by_sym)} ({extended_added} extended added beyond UNIVERSE)", flush=True)
    training_universe = list(bars_by_sym.keys())

    context = fetch_context()   # cached after first call
    train_feat: dict[str, pd.DataFrame] = {}
    eval_feat: dict[str, pd.DataFrame] = {}
    for sym in training_universe:
        bars = bars_by_sym[sym]
        tr_bars, ev_bars = split(bars)
        # exp41: subset train slice to last TRAIN_LOOKBACK_DAYS — exp40's full
        # 6yr training produced 0 trades on the 90-day eval; the model became
        # too uncertain on the noisier large dataset. Recent-only training
        # gives more confident predictions for the recent regime.
        if TRAIN_LOOKBACK_DAYS and len(tr_bars) > 0:
            cutoff = tr_bars["timestamp"].iloc[-1] - pd.Timedelta(days=TRAIN_LOOKBACK_DAYS)
            tr_bars = tr_bars[tr_bars["timestamp"] >= cutoff].reset_index(drop=True)
        # featurize each slice independently — strict no-leakage
        if len(tr_bars) > 200:
            train_feat[sym] = featurize(tr_bars, context=context)
        if len(ev_bars) > 50:
            eval_feat[sym] = featurize(ev_bars, context=context)

    # exp79: populate universe-aggregate context features (per-timestep
    # cross-sectional mean/dispersion/breadth) on each symbol's frame.
    # MUST happen after per-symbol featurize so all timestamps are present.
    add_universe_context(train_feat)
    add_universe_context(eval_feat)

    n_features = len(USE_FEATURES)
    model = PatchTransformer(
        n_features=n_features, patch_len=PATCH_LEN, context_patches=CONTEXT_PATCHES,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT, pred_horizon=PRED_HORIZON,
        horizons_minutes=HORIZONS_MINUTES,   # exp28: also predict at 1m/1h/1d/1w
    ).to(device)
    print(f"[experiment] device={device}  features={n_features}  params={model.num_parameters():,}", flush=True)

    cached_ckpt = CHECKPOINT_DIR / f"last_seed{seed}.pt"
    if USE_CACHED_PRETRAIN and cached_ckpt.exists():
        # Speedup: skip both pretrain phases (~25 min savings per iteration).
        # Use when only post-training params changed (cap, SWAP_MARGIN, etc).
        print(f"[experiment] USE_CACHED_PRETRAIN — loading {cached_ckpt}, SKIPPING pretrain", flush=True)
        try:
            ck = torch.load(cached_ckpt, map_location=device)
            model.load_state_dict(ck["state_dict"])
            print(f"[experiment] cached pretrain loaded", flush=True)
        except Exception as e:
            print(f"[experiment] cache load failed ({e}); falling back to full pretrain", flush=True)
            supervised_pretrain(model, train_feat, device)
            for ep in range(RL_PRETRAIN_EPOCHS):
                print(f"[rl_pretrain] epoch {ep+1}/{RL_PRETRAIN_EPOCHS} (encoder-warming)", flush=True)
                _ = simulate(model, train_feat, device, learn=True)
    else:
        # Phase 1: supervised forecast-head pretrain (trains multi-horizon head).
        supervised_pretrain(model, train_feat, device)

        # Phase 2: offline RL on the train slice — KEPT for its side-effect of
        # warming the shared transformer encoder via gradient flow. Removing it
        # in exp39 dropped weighted sharpe +1.79 → +0.97. The action head it
        # trains is unused (primary/picker gone), but the encoder updates are
        # load-bearing for forecast-head confidence in the weighted strategy.
        for ep in range(RL_PRETRAIN_EPOCHS):
            print(f"[rl_pretrain] epoch {ep+1}/{RL_PRETRAIN_EPOCHS} (encoder-warming)", flush=True)
            _ = simulate(model, train_feat, device, learn=True)

    # Phases 3-4 (primary + picker eval) stay removed — they didn't beat passive.

    # exp66 SPEEDUP: precompute model predictions ONCE, share across all simulators.
    # Each simulator becomes pure dict lookup + numpy — no per-bar GPU launch.
    pred_cache = precompute_predictions(model, eval_feat, device)

    # Phase 5: WEIGHTED dynamic-sizing strategy (the only one that works).
    weighted = simulate_weighted(model, eval_feat, device, precomputed_preds=pred_cache)
    # Multi-trader-profile + passive-topn + SPY comparison suite
    run_profile_suite(model, eval_feat, device, seed, precomputed_preds=pred_cache)

    # Save trained weights for this seed. Agent loop promotes last_*.pt → best_*.pt
    # whenever sharpe_ci_low improves on the prior best.
    try:
        ckpt_path = CHECKPOINT_DIR / f"last_seed{seed}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "patch_len": PATCH_LEN, "context_patches": CONTEXT_PATCHES,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS, "d_ff": D_FF,
            "dropout": DROPOUT, "pred_horizon": PRED_HORIZON,
            "horizons_minutes": HORIZONS_MINUTES,
            "use_features": USE_FEATURES,
        }, ckpt_path)
        print(f"[checkpoint] saved {ckpt_path}", flush=True)
    except Exception as e:
        print(f"[checkpoint] save failed: {e}", flush=True)

    # Dump per-seed trade list so the autoresearch driver can include it in
    # the iteration .md file (each BUY/SELL with timestamp + symbol).
    try:
        import json as _json
        trades_path = CHECKPOINT_DIR / f"last_seed{seed}_trades.json"
        end_eq = float(weighted.equity_curve[-1][1]) if weighted.equity_curve else 0.0
        trades_payload = {
            "seed": seed,
            "n_trades": int(weighted.n_trades),
            "starting_cash": float(STARTING_CASH_USD),
            "ending_equity": end_eq,
            "trades": [
                {"ts": str(ts), "symbol": sym, "side": side}
                for (ts, sym, side) in weighted.trades
            ],
        }
        trades_path.write_text(_json.dumps(trades_payload, indent=2))
    except Exception as e:
        print(f"[trades-dump] seed {seed} failed: {e}", flush=True)

    # --- Out-of-symbol HOLDOUT eval (exp50) -------------------------------
    # Run the trained model on stocks it never saw. Generalization check.
    holdout_metrics = {"sharpe": 0.0, "pnl": 0.0, "trades": 0, "dd_pct": 0.0,
                       "symbols": [], "ending_equity": 0.0}
    try:
        from prepare import sharpe_ratio as _sr, max_drawdown_pct as _dd
        holdout_feat: dict[str, pd.DataFrame] = {}
        for sym in HOLDOUT_UNIVERSE:
            try:
                bars = fetch_bars(sym)
            except Exception as e:
                print(f"[holdout] {sym}: fetch failed ({e})", flush=True)
                continue
            _, ev_bars = split(bars)
            if len(ev_bars) > 50:
                holdout_feat[sym] = featurize(ev_bars, context=context)
        # exp79: populate universe context for holdout too (separate aggregation)
        add_universe_context(holdout_feat)
        if holdout_feat:
            holdout_broker = simulate_weighted(model, holdout_feat, device)
            end_eq_h = float(holdout_broker.equity_curve[-1][1]) if holdout_broker.equity_curve else 0.0
            start_eq_h = float(holdout_broker.equity_curve[0][1]) if holdout_broker.equity_curve else 0.0
            holdout_metrics = {
                "sharpe": float(_sr(holdout_broker.equity_curve)),
                "pnl": end_eq_h - start_eq_h,
                "trades": int(holdout_broker.n_trades),
                "dd_pct": float(_dd(holdout_broker.equity_curve)),
                "symbols": list(holdout_feat.keys()),
                "ending_equity": end_eq_h,
            }
            print(f"[holdout] sharpe={holdout_metrics['sharpe']:+.3f} "
                  f"pnl=${holdout_metrics['pnl']:+,.2f} trades={holdout_metrics['trades']} "
                  f"dd={holdout_metrics['dd_pct']:+.2f}% on {holdout_metrics['symbols']}", flush=True)
    except Exception as e:
        print(f"[holdout] failed: {e}", flush=True)

    try:
        import json as _json
        holdout_path = CHECKPOINT_DIR / f"last_seed{seed}_holdout.json"
        holdout_path.write_text(_json.dumps(holdout_metrics, indent=2, default=str))
    except Exception as e:
        print(f"[holdout-dump] seed {seed} failed: {e}", flush=True)

    # exp157: canonical top2 quarter-readiness with 81.875% reserve under SPY-alpha objective.
    canonical_broker = weighted
    try:
        # exp87: REVERT to (3,4) = 4h + 1d combo. Both single-horizon variants
        # (exp85=(4,) and exp86=(3,)) regressed materially. The combo is the
        # local optimum.
        topn_broker = simulate_passive_topn(model, eval_feat, device, top_n=2, name="top2",
                                            ranking_horizons=(3, 4),
                                            precomputed_preds=pred_cache)
        if topn_broker.equity_curve and len(topn_broker.equity_curve) > 5:
            canonical_broker = topn_broker
            print(f"[experiment] canonical = top2_picker (final equity ${topn_broker.equity_curve[-1][1]:,.2f})", flush=True)
    except Exception as e:
        print(f"[experiment] top4 canonical failed ({e}); falling back to weighted", flush=True)
    return (
        canonical_broker.equity_curve, canonical_broker.n_trades,
        canonical_broker.total_fees, 0.0,
        canonical_broker.trades,
        getattr(canonical_broker, "cash_curve", []),
    )


if __name__ == "__main__":
    t0 = time.time()
    result = train_and_eval(seed=0)
    eq, nt, fees, slip, trades = result[:5]
    p_eq, p_nt, p_fees, p_trades = result[5:9]
    w_eq, w_nt, w_fees, w_trades = result[9:13]
    print(f"\n[primary]  bars={len(eq)}  trades={nt}  fees=${fees:.2f}  slippage=${slip:.2f}")
    if eq: print(f"[primary]  equity start=${eq[0][1]:,.2f}  end=${eq[-1][1]:,.2f}")
    print(f"[picker]   bars={len(p_eq)}  trades={p_nt}  fees=${p_fees:.2f}")
    if p_eq: print(f"[picker]   equity start=${p_eq[0][1]:,.2f}  end=${p_eq[-1][1]:,.2f}")
    print(f"[weighted] bars={len(w_eq)}  trades={w_nt}  fees=${w_fees:.2f}")
    if w_eq: print(f"[weighted] equity start=${w_eq[0][1]:,.2f}  end=${w_eq[-1][1]:,.2f}")
    print(f"[experiment] total wall: {time.time()-t0:.1f}s")
