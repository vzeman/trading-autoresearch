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

import numpy as np
import pandas as pd
import torch
from torch import nn

from prepare import (
    UNIVERSE, NOTIONAL_PER_SYMBOL_USD, STARTING_CASH_USD,
    FEE_PER_TRADE_USD, SLIPPAGE_BPS,
    PaperBroker, prepare_all, split,
)

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
]

USE_FEATURES = [f for f in ALL_FEATURES if f not in {"signed_log_vol", "vol_z_15"}]   # exp4: drop noisy ones


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal EMA via pandas."""
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def featurize(bars: pd.DataFrame) -> pd.DataFrame:
    """OHLCV bars → causal feature dataframe + the close price (broker needs it).

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
    return feat


# ============================================================================
# HYPERPARAMETERS — agent edits freely
# ============================================================================

PATCH_LEN = 8
CONTEXT_PATCHES = 16            # context window = PATCH_LEN * CONTEXT_PATCHES = 128 bars
D_MODEL = 96                    # bumped from 64 — more features now
N_HEADS = 4
N_LAYERS = 3
D_FF = 192
DROPOUT = 0.1
PRED_HORIZON = 5
RL_REWARD_HORIZON = 3
ACTION_HEAD_HOLD_BIAS = 1.0     # softmax([-1,1,-1]) ≈ [13%,73%,13%]: explore but lean HOLD

PRETRAIN_EPOCHS = 2             # supervised forecast pretrain on TRAIN slice
PRETRAIN_BATCH = 128
PRETRAIN_LR = 3e-4
RL_PRETRAIN_EPOCHS = 1          # offline RL pass(es) on TRAIN slice
RL_LR = 3e-5     # exp6: 3× stronger REINFORCE updates so policy actually moves
RL_COEF = 1.0
ENTROPY_COEF = 0.01
SGD_BATCH = 64
GRAD_CLIP = 1.0
RL_STEP_EVERY_BARS = 5


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
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
                 dropout: float, pred_horizon: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.patch_len = patch_len
        self.context_patches = context_patches
        self.pred_horizon = pred_horizon

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

    @staticmethod
    def gaussian_nll(mean, log_std, target) -> torch.Tensor:
        var = torch.exp(2 * log_std)
        return (log_std + 0.5 * (target - mean) ** 2 / var + 0.5 * math.log(2 * math.pi)).mean()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# WINDOWING
# ============================================================================

def windows_from_features(feat: pd.DataFrame, context_len: int, pred_horizon: int):
    cols = USE_FEATURES
    arr = feat[cols].to_numpy(np.float32)
    log_ret = feat["log_return_1"].to_numpy(np.float32)
    n = len(feat) - context_len - pred_horizon
    if n <= 0:
        return np.empty((0, context_len, len(cols)), np.float32), np.empty((0, pred_horizon), np.float32)
    X = np.empty((n, context_len, len(cols)), np.float32)
    y = np.empty((n, pred_horizon), np.float32)
    for i in range(n):
        X[i] = arr[i : i + context_len]
        y[i] = log_ret[i + context_len : i + context_len + pred_horizon]
    return X, y


# ============================================================================
# SUPERVISED PRETRAIN
# ============================================================================

def supervised_pretrain(model: PatchTransformer, train_features: dict[str, pd.DataFrame], device: str):
    Xs, ys = [], []
    for sym, feat in train_features.items():
        X, y = windows_from_features(feat, model.context_len, model.pred_horizon)
        if len(X) > 0:
            Xs.append(X); ys.append(y)
    if not Xs:
        return
    X = torch.from_numpy(np.concatenate(Xs))
    y = torch.from_numpy(np.concatenate(ys))
    n = len(X)
    perm = torch.randperm(n)
    X, y = X[perm], y[perm]
    opt = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=1e-4)
    model.train()
    for ep in range(PRETRAIN_EPOCHS):
        losses = []
        for i in range(0, n - SGD_BATCH, PRETRAIN_BATCH):
            xb = X[i : i + PRETRAIN_BATCH].to(device)
            yb = y[i : i + PRETRAIN_BATCH].to(device)
            opt.zero_grad(set_to_none=True)
            mean, log_std, _ = model(xb)
            loss = PatchTransformer.gaussian_nll(mean, log_std, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(loss.item())
        if losses:
            print(f"[pretrain] epoch {ep+1}/{PRETRAIN_EPOCHS}  nll={np.mean(losses):.4f}", flush=True)


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

    broker = PaperBroker()
    sym_state: dict[str, dict] = {s: {"i": -1, "pending": [], "last_pos": 0.0} for s in features}

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
                _, _, alog = model(xb)        # alog: (B, 3)
                if learn:
                    probs = torch.softmax(alog, dim=-1)
                    a_idx_t = torch.distributions.Categorical(probs=probs).sample()
                else:
                    a_idx_t = torch.argmax(alog, dim=-1)
                a_idx_list = a_idx_t.cpu().tolist()
            model.train()

            # Apply each decision (still serial, but only broker bookkeeping — fast)
            for (sym, i_now), a_idx, X_arr in zip(batch_meta, a_idx_list, batch_X):
                st = sym_state[sym]
                target = float(ACTION_TO_POS[a_idx])
                pos_change = abs(target - st["last_pos"])
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
                reward = portfolio_weight * (p_["target"] * log_ret - cost_charge)
                buf_X.append(p_["X"]); buf_a.append(p_["a"]); buf_r.append(reward)

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
# CONTRACT — DO NOT change the signature
# ============================================================================

def train_and_eval(seed: int = 0) -> tuple:
    """Returns (equity_curve, n_trades, total_fees, total_slippage)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device()

    bars_by_sym = prepare_all()
    train_feat: dict[str, pd.DataFrame] = {}
    eval_feat: dict[str, pd.DataFrame] = {}
    for sym in UNIVERSE:
        bars = bars_by_sym[sym]
        tr_bars, ev_bars = split(bars)
        # featurize each slice independently — strict no-leakage
        if len(tr_bars) > 200:
            train_feat[sym] = featurize(tr_bars)
        if len(ev_bars) > 50:
            eval_feat[sym] = featurize(ev_bars)

    n_features = len(USE_FEATURES)
    model = PatchTransformer(
        n_features=n_features, patch_len=PATCH_LEN, context_patches=CONTEXT_PATCHES,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT, pred_horizon=PRED_HORIZON,
    ).to(device)
    print(f"[experiment] device={device}  features={n_features}  params={model.num_parameters():,}", flush=True)

    # Phase 1: supervised forecast-head pretrain
    supervised_pretrain(model, train_feat, device)

    # Phase 2: offline RL on the train slice (does not touch eval)
    for ep in range(RL_PRETRAIN_EPOCHS):
        print(f"[rl_pretrain] epoch {ep+1}/{RL_PRETRAIN_EPOCHS}", flush=True)
        _ = simulate(model, train_feat, device, learn=True)

    # Phase 3: deterministic eval on the held-out slice
    eval_broker = simulate(model, eval_feat, device, learn=False)
    return (eval_broker.equity_curve, eval_broker.n_trades,
            eval_broker.total_fees, eval_broker.total_slippage)


if __name__ == "__main__":
    t0 = time.time()
    eq, nt, fees, slip = train_and_eval(seed=0)
    print(f"\n[experiment] eval bars={len(eq)}  trades={nt}  fees=${fees:.2f}  slippage=${slip:.2f}")
    if eq:
        print(f"[experiment] equity start=${eq[0][1]:,.2f}  end=${eq[-1][1]:,.2f}")
    print(f"[experiment] total wall: {time.time()-t0:.1f}s")
