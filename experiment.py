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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

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
CONTEXT_SYMBOLS = ["^VIX", "TLT", "UUP", "SPY"]   # VIX, 20yr Treasury, USD-index proxy, SPY (cross-asset)

# ============================================================================
# HOLDOUT UNIVERSE — stocks the model NEVER trains on. After the in-symbol eval
# completes, the same trained model is run on these names' last-90d bars to
# test out-of-symbol generalization. A model that learned real signal should
# produce a comparable (within ~30%) sharpe here vs in-sample.
# ============================================================================
HOLDOUT_UNIVERSE = ["JPM", "WMT", "V", "DIS", "JNJ"]

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
    "vix_logret_1",   # CBOE Volatility Index — fear gauge
    "tlt_logret_1",   # 20yr Treasury ETF — interest-rate signal
    "uup_logret_1",   # USD-index ETF (DXY proxy) — currency macro
    "spy_logret_1",   # SPY return as a market factor (for SPY itself this == log_return_1)
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
    "^VIX": "vix_logret_1",
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
    return feat


# ============================================================================
# HYPERPARAMETERS — agent edits freely
# ============================================================================

PATCH_LEN = 8
CONTEXT_PATCHES = 16            # context window = PATCH_LEN * CONTEXT_PATCHES = 128 bars
D_MODEL = 96                    # exp11 KEPT; exp14 (64) and exp17 (128) both discarded
N_HEADS = 4
N_LAYERS = 3
D_FF = 192
DROPOUT = 0.1   # exp22 (DROPOUT=0) was disastrous — keep regularization
PRED_HORIZON = 5
# exp28: multi-horizon prediction targets (in 1-min bars)
# 1, 60, 390, 1950 = 1m, 1h, 1d, 1w. Each predicted as cumulative log-return Gaussian.
HORIZONS_MINUTES = [1, 60, 390, 1950]
RL_REWARD_HORIZON = 3
ACTION_HEAD_HOLD_BIAS = 1.5     # exp10: softmax([-1.5,1.5,-1.5]) ≈ [4.7%,90.6%,4.7%]: be even more selective

PRETRAIN_EPOCHS = 1             # exp41: 2→1 — at v7 6yr × 20-sym scale each epoch is ~5h. One epoch is plenty given the increased data.
TRAIN_LOOKBACK_DAYS = 365       # exp41: subset train slice to last N days. Hypothesis: model trained on full 6yr is too conservative for recent regime → exp40 = 0 trades. Recent-only data should produce more confident predictions.
PRETRAIN_BATCH = 128
PRETRAIN_LR = 3e-4
RL_PRETRAIN_EPOCHS = 1          # offline RL pass(es) on TRAIN slice
RL_LR = 2e-5     # exp7 KEPT setting (known stable, no rogue seeds)
RL_COEF = 1.0
ENTROPY_COEF = 0.005   # exp50: 0.01→0.005. exp49 (SWAP_MARGIN=0.15) had best raw sharpe (+1.548) and PnL (+$3,442) but ci_low -1.523 missed exp47's -1.513 by 0.01 — wider per-seed variance suggested too much exploration. Tighter convergence should narrow CI.
VOL_PENALTY = 0.0   # exp20/21 showed: small penalty=invisible, large penalty=destabilizing. Off.

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
    """
    if torch.cuda.is_available():
        return "cuda"
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
    horizons = model.horizons_minutes
    ds = WindowDataset(train_features, model.context_len, model.pred_horizon, horizons)
    n = len(ds)
    if n == 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    model.train()
    for ep in range(PRETRAIN_EPOCHS):
        perm = np.random.permutation(n)   # uses global np.random state seeded by train_and_eval
        losses, losses_mh = [], []
        for i in range(0, n - SGD_BATCH, PRETRAIN_BATCH):
            batch_idxs = perm[i : i + PRETRAIN_BATCH]
            X_np, y_np, y_mh_np = ds.get_batch(batch_idxs)
            xb = torch.from_numpy(X_np).to(device)
            yb = torch.from_numpy(y_np).to(device)
            opt.zero_grad(set_to_none=True)
            mean, log_std, _ = model(xb)
            loss = PatchTransformer.gaussian_nll(mean, log_std, yb)
            if y_mh_np is not None:
                ymh_b = torch.from_numpy(y_mh_np).to(device)
                mh_mean, mh_log_std = model.forward_multi_horizon(xb)
                loss_mh = PatchTransformer.gaussian_nll(mh_mean, mh_log_std, ymh_b)
                loss = loss + loss_mh
                losses_mh.append(float(loss_mh.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        if losses:
            extra = f"  mh_nll={np.mean(losses_mh):.4f}" if losses_mh else ""
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
            while st["pending"] and st["pending"][0]["i"] + H_max <= i_now:
                p_ = st["pending"].pop(0)
                future_close = float(close[p_["i"] + RL_REWARD_HORIZON])
                log_ret = math.log(max(future_close, 1e-12) / max(p_["entry"], 1e-12))
                cost_charge = 0.5 * (round_trip_var_cost + fixed_cost_frac) * p_["pos_change"]
                pos_ret = p_["target"] * log_ret
                reward = portfolio_weight * (pos_ret - cost_charge - VOL_PENALTY * pos_ret * pos_ret)
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
MIN_CASH_RESERVE_PCT = 0.10           # keep 10% of starting cash unspent
MAX_NEW_TRADES_PER_TIMESTEP = 5       # diversify timing
KELLY_SCALE = 0.5                     # half-Kelly (exp33: doubling had no effect — cap saturates)
WEIGHTED_SELL_SHARPE = 0.0            # close any held position whose 1h predicted Sharpe drops below this
WEIGHTED_MIN_TRADE_USD = 100.0        # too small → fee dominates
WEIGHTED_SWAP_MARGIN = 0.15           # exp50: keep 0.15 from exp49 (best PnL +$3,442, sharpe +1.548). DISCARD on exp49 was tiny ci_low miss; reducing entropy should tighten CI.


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

    def buy_usd(self, sym: str, price: float, ts: pd.Timestamp, dollar_amount: float) -> bool:
        """Buy dollar_amount worth of sym at price. Returns True if executed."""
        if dollar_amount < WEIGHTED_MIN_TRADE_USD:
            return False
        if dollar_amount > self.free_cash():
            return False
        fill_price = price * (1.0 + SLIPPAGE_BPS * 1e-4)
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


def simulate_weighted(model: PatchTransformer,
                      features: dict[str, pd.DataFrame],
                      device: str) -> WeightedBroker:
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
                    broker.sell_all(sym, float(close_arrays[sym][i_now]), ts)

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
                    broker.sell_all(weak_sym, float(close_arrays[weak_sym][i_now_weak]), ts)
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
                broker.buy_usd(sym, price, ts, actual)

        prices = {s: float(close_arrays[s][last_idx_by_sym[s]])
                  for s in features if last_idx_by_sym[s] >= 0}
        broker.mark_to_market(ts, prices)

    return broker


# ============================================================================
# CONTRACT — DO NOT change the signature
# ============================================================================

def train_and_eval(seed: int = 0) -> tuple:
    """Returns (equity_curve, n_trades, total_fees, total_slippage)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()

    bars_by_sym = prepare_all()
    context = fetch_context()   # cached after first call
    train_feat: dict[str, pd.DataFrame] = {}
    eval_feat: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
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

    n_features = len(USE_FEATURES)
    model = PatchTransformer(
        n_features=n_features, patch_len=PATCH_LEN, context_patches=CONTEXT_PATCHES,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT, pred_horizon=PRED_HORIZON,
        horizons_minutes=HORIZONS_MINUTES,   # exp28: also predict at 1m/1h/1d/1w
    ).to(device)
    print(f"[experiment] device={device}  features={n_features}  params={model.num_parameters():,}", flush=True)

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

    # Phase 5: WEIGHTED dynamic-sizing strategy (the only one that works).
    weighted = simulate_weighted(model, eval_feat, device)

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

    # Return 6-element tuple: weighted_eq, n_trades, fees, slip, trades, cash_curve.
    return (
        weighted.equity_curve, weighted.n_trades,
        weighted.total_fees, 0.0,
        weighted.trades,
        getattr(weighted, "cash_curve", []),
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
