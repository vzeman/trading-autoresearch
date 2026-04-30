"""THE FILE THE AGENT MODIFIES.

Single self-contained file with:
  - Model (PatchTST-style transformer, forecast head + 3-action head)
  - Replay buffers
  - Train loop (supervised pretrain on train slice + offline RL pretrain)
  - Policy (acts on the action head; portfolio-weighted reward)

`evaluator.py` imports `train_and_eval(seed)` from this file. Everything else
the agent decides — architecture, optimizer, hyperparameters, features used,
reward shaping, exploration schedule, anything — as long as the contract holds:

CONTRACT (do not break):
  - `train_and_eval(seed: int) -> (equity_curve, n_trades, total_fees, total_slippage)`
    where equity_curve is what the broker accumulated during the EVAL slice.
  - Use `prepare.PaperBroker` as-is for fills + fees + slippage. No custom broker.
  - Use `prepare.split(...)` for train/eval split. No leakage.
  - Use the FROZEN constants in `prepare.py` (cash, fees, etc.).

If you need to change the contract, talk to the human (edit program.md).
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from prepare import (
    UNIVERSE, FEATURE_NAMES, N_FEATURES, NOTIONAL_PER_SYMBOL_USD,
    STARTING_CASH_USD, FEE_PER_TRADE_USD, SLIPPAGE_BPS,
    PaperBroker, prepare_all, featurize, split,
)

# ============================================================================
# HYPERPARAMETERS — agent edits freely
# ============================================================================

PATCH_LEN = 8
CONTEXT_PATCHES = 16            # context window = PATCH_LEN * CONTEXT_PATCHES = 128 bars
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128
DROPOUT = 0.1
PRED_HORIZON = 5                # supervised: predict next-5 bars
RL_REWARD_HORIZON = 3           # RL: reward measured over next-3 bars
ACTION_HEAD_HOLD_BIAS = 3.0     # initial logit bias toward HOLD (anti-churn)

PRETRAIN_EPOCHS = 3             # supervised forecast pretrain on TRAIN slice
PRETRAIN_BATCH = 128
PRETRAIN_LR = 3e-4
RL_PRETRAIN_EPOCHS = 1          # offline RL pass(es) on TRAIN slice
RL_LR = 1e-5
RL_COEF = 1.0
ENTROPY_COEF = 0.01
EWC_LAMBDA = 0.0                # 0 = no EWC anchor; agent can turn this on
SGD_BATCH = 64
GRAD_CLIP = 1.0
RL_STEP_EVERY_BARS = 5          # backward pass cadence inside RL replay
USE_FEATURES = list(FEATURE_NAMES)  # agent can drop features


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
# DATA WINDOWS
# ============================================================================

def windows_from_features(feat: pd.DataFrame, context_len: int, pred_horizon: int):
    cols = USE_FEATURES
    arr = feat[cols].to_numpy(np.float32)
    log_ret = feat["log_return"].to_numpy(np.float32)
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
             learn: bool, anchor: dict | None = None) -> PaperBroker:
    """Replay merged events through the model. If learn=True, also train action head."""
    portfolio_weight = NOTIONAL_PER_SYMBOL_USD / STARTING_CASH_USD
    round_trip_var_cost = 2.0 * SLIPPAGE_BPS * 1e-4
    fixed_cost_frac = 2.0 * FEE_PER_TRADE_USD / NOTIONAL_PER_SYMBOL_USD

    broker = PaperBroker()
    sym_state: dict[str, dict] = {s: {"bars": [], "pending": [], "last_pos": 0.0} for s in features}

    # merge events chronologically
    rows: list[tuple[pd.Timestamp, str, dict]] = []
    cols = USE_FEATURES
    for sym, feat in features.items():
        arr = feat[cols].to_numpy(np.float32)
        for i, ts in enumerate(feat["timestamp"]):
            rows.append((ts, sym, {"i": i, "feat": arr, "ts": ts,
                                   "close": _approx_close_from_logret(feat, i)}))
    rows.sort(key=lambda r: r[0])

    opt = None
    if learn:
        opt = torch.optim.AdamW(model.parameters(), lr=RL_LR, weight_decay=0.0)

    # RL replay buffer (in-memory, reset per call)
    buf_X: list[np.ndarray] = []
    buf_a: list[int] = []
    buf_r: list[float] = []

    H_max = max(model.pred_horizon, RL_REWARD_HORIZON)
    n_decisions = 0

    for ev_i, (ts, sym, payload) in enumerate(rows):
        st = sym_state[sym]
        st["bars"].append(payload)
        if len(st["bars"]) > model.context_len * 4:
            drop = len(st["bars"]) - model.context_len * 4
            st["bars"] = st["bars"][drop:]
            for p in st["pending"]:
                p["idx"] -= drop

        if len(st["bars"]) >= model.context_len + 1:
            arr = payload["feat"]
            i_now = payload["i"]
            if i_now >= model.context_len - 1:
                # window of features ending at this bar
                X = arr[i_now - model.context_len + 1 : i_now + 1].copy()
                with torch.no_grad():
                    model.eval()
                    xb = torch.from_numpy(X).unsqueeze(0).to(device)
                    _, _, alog = model(xb)
                    if learn:
                        a_idx = int(torch.distributions.Categorical(probs=torch.softmax(alog[0], dim=-1)).sample().item())
                    else:
                        a_idx = int(torch.argmax(alog[0]).item())
                model.train()
                target = float(ACTION_TO_POS[a_idx])
                pos_change = abs(target - st["last_pos"])
                broker.update(sym, payload["close"], ts, target)
                st["pending"].append({"X": X, "a": a_idx, "target": target,
                                      "entry": payload["close"], "idx": len(st["bars"]) - 1,
                                      "pos_change": pos_change})
                st["last_pos"] = target
                n_decisions += 1

        # mark portfolio
        prices = {s: ss["bars"][-1]["close"] for s, ss in sym_state.items() if ss["bars"]}
        broker.mark_to_market(ts, prices)

        # resolve pending
        while st["pending"] and st["pending"][0]["idx"] + H_max < len(st["bars"]):
            p_ = st["pending"].pop(0)
            future_close = st["bars"][p_["idx"] + RL_REWARD_HORIZON]["close"]
            log_ret = math.log(max(future_close, 1e-12) / max(p_["entry"], 1e-12))
            cost_charge = 0.5 * (round_trip_var_cost + fixed_cost_frac) * p_["pos_change"]
            reward = portfolio_weight * (p_["target"] * log_ret - cost_charge)
            buf_X.append(p_["X"]); buf_a.append(p_["a"]); buf_r.append(reward)

        # periodic SGD step (only if learn=True)
        if learn and (ev_i + 1) % RL_STEP_EVERY_BARS == 0 and len(buf_X) >= SGD_BATCH:
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


def _approx_close_from_logret(feat: pd.DataFrame, i: int) -> float:
    """Reconstruct close from cumulative log returns (drops absolute price level scale).

    We need true closes for broker fills. featurize() drops them, so we re-derive
    here from log returns + a normalized starting price of 100.
    """
    if not hasattr(_approx_close_from_logret, "_cache"):
        _approx_close_from_logret._cache = {}
    cache = _approx_close_from_logret._cache
    key = id(feat)
    if key not in cache:
        log_ret = feat["log_return"].to_numpy(np.float64)
        cum = np.cumsum(log_ret)
        cache[key] = 100.0 * np.exp(cum)
    return float(cache[key][i])


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
        feat = featurize(bars_by_sym[sym])
        tr, ev = split(feat)
        if len(tr) > 200 and len(ev) > 50:
            train_feat[sym] = tr
            eval_feat[sym] = ev

    n_features = len(USE_FEATURES)
    model = PatchTransformer(
        n_features=n_features, patch_len=PATCH_LEN, context_patches=CONTEXT_PATCHES,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=DROPOUT, pred_horizon=PRED_HORIZON,
    ).to(device)
    print(f"[experiment] device={device}  params={model.num_parameters():,}", flush=True)

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
