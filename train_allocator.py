"""Train a second-stage allocator on top of world-model predictions.

The base world model predicts candidate outcomes. This script learns a compact
planner score from those predictions and simple action metadata, then evaluates
the learned score with the existing planner logic.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_world_model import evaluate_planner, load_frame, predict
from train_world_model import action_keys, pick_device


NUMERIC_FEATURES = [
    "pred_portfolio_return",
    "pred_max_drawdown",
    "pred_path_vol",
    "pred_future_alpha_vs_spy",
    "pred_future_min_asset_return",
    "pred_future_asset_max_drawdown",
    "pred_profit_label",
    "pred_beat_spy_label",
    "pred_asset_crash_label",
    "pred_severe_adverse_label",
    "pred_rank_top_quartile",
    "price",
    "horizon_bars",
    "current_position_frac",
    "target_position_frac",
    "trade_notional",
    "fees",
    "slippage",
]

CONTEXT_FEATURES = [
    "feat_tlt_logret_1",
    "feat_uup_logret_1",
    "feat_spy_logret_1",
    "feat_spy_logret_60",
    "feat_spy_logret_240",
    "feat_spy_logret_390",
    "feat_spy_logret_2730",
    "feat_spy_logret_5460",
    "feat_spy_logret_10920",
    "feat_spy_logret_16380",
    "state_ret_30m",
    "state_vol_30m",
    "state_volume_z_30m",
    "state_ret_2h",
    "state_vol_2h",
    "state_volume_z_2h",
    "state_ret_1d",
    "state_vol_1d",
    "state_volume_z_1d",
    "state_ret_5d",
    "state_vol_5d",
    "state_volume_z_5d",
    "state_ret_20d",
    "state_vol_20d",
    "state_volume_z_20d",
    "state_drawdown_5d",
    "xsec_universe_count",
    "xsec_ret_30m_rank_pct",
    "xsec_ret_30m_minus_median",
    "xsec_ret_30m_up_frac",
    "xsec_ret_2h_rank_pct",
    "xsec_ret_2h_minus_median",
    "xsec_ret_2h_up_frac",
    "xsec_ret_1d_rank_pct",
    "xsec_ret_1d_minus_median",
    "xsec_ret_1d_up_frac",
    "xsec_vol_1d_rank_pct",
    "xsec_volume_z_1d_rank_pct",
]


@dataclass(frozen=True)
class AllocatorConfig:
    train_data: str
    test_data: str
    world_checkpoint: str
    output: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    n_layers: int
    dropout: float
    val_fraction: float
    val_gap_days: float
    limit_rows: int
    min_horizon_bars: int
    max_horizon_bars: int
    top_quantile: float
    feature_mode: str
    utility_mode: str
    train_entry_only: bool
    extra_roundtrip_bps: float
    drawdown_penalty: float
    volatility_penalty: float
    seed: int
    device: str


class AllocatorModel(nn.Module):
    def __init__(self, n_features: int, n_symbols: int, n_actions: int, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        symbol_dim = min(32, max(8, int(np.ceil(np.sqrt(max(n_symbols, 1))) * 2)))
        action_dim = min(16, max(4, int(np.ceil(np.sqrt(max(n_actions, 1))) * 2)))
        self.symbol_emb = nn.Embedding(n_symbols + 1, symbol_dim)
        self.action_emb = nn.Embedding(n_actions + 1, action_dim)
        layers: list[nn.Module] = []
        dim = n_features + symbol_dim + action_dim
        for _ in range(max(1, n_layers)):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.utility_head = nn.Linear(hidden_dim, 1)
        self.top_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor, action_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([x, self.symbol_emb(symbol_id), self.action_emb(action_id)], dim=-1)
        h = self.trunk(z)
        return self.utility_head(h).squeeze(-1), self.top_head(h).squeeze(-1)


def filter_frame(df: pd.DataFrame, min_horizon: int, max_horizon: int) -> pd.DataFrame:
    if min_horizon > 0:
        df = df[df["horizon_bars"] >= min_horizon]
    if max_horizon > 0:
        df = df[df["horizon_bars"] <= max_horizon]
    if df.empty:
        raise RuntimeError("no rows left after horizon filtering")
    return df.reset_index(drop=True)


def entry_only_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        (df["action"].astype(str) == "buy")
        & (df["current_position_frac"].astype(float) == 0.0)
        & (df["target_position_frac"].astype(float) > 0.0)
    ].copy()
    if out.empty:
        raise RuntimeError("no entry-only buy rows left after filtering")
    return out.reset_index(drop=True)


def split_masks(df: pd.DataFrame, val_fraction: float, val_gap_days: float) -> tuple[np.ndarray, np.ndarray]:
    time_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    timestamps = pd.to_datetime(df[time_col], utc=True)
    cutoff = timestamps.quantile(max(0.0, min(1.0, 1.0 - val_fraction)))
    train_cutoff = cutoff - pd.Timedelta(days=max(0.0, val_gap_days))
    train_mask = (timestamps <= train_cutoff).to_numpy()
    val_mask = (timestamps > cutoff).to_numpy()
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError("empty allocator train/validation split")
    return train_mask, val_mask


def _stress_adjusted_return(df: pd.DataFrame, extra_roundtrip_bps: float) -> pd.Series:
    target_frac = df["target_position_frac"].astype(float) if "target_position_frac" in df.columns else 1.0
    extra_cost = target_frac * max(0.0, extra_roundtrip_bps) * 1e-4
    return df["portfolio_return"].astype(float) - extra_cost


def add_targets(
    df: pd.DataFrame,
    top_quantile: float,
    utility_mode: str = "default",
    extra_roundtrip_bps: float = 0.0,
    drawdown_penalty: float = 0.25,
    volatility_penalty: float = 0.10,
) -> pd.DataFrame:
    out = df.copy()
    target_frac_for_crash = out.get("target_position_frac", pd.Series(1.0, index=out.index)).astype(float).clip(lower=0.01)
    if "future_min_asset_return" not in out.columns:
        derived_dd = out.get("max_drawdown", pd.Series(0.0, index=out.index)).astype(float) / target_frac_for_crash
        asset_ret = out.get("future_asset_return", out.get("portfolio_return", pd.Series(0.0, index=out.index))).astype(float)
        out["future_min_asset_return"] = np.minimum(asset_ret, derived_dd.clip(lower=-1.0, upper=0.0)).astype(np.float32)
    if "future_asset_max_drawdown" not in out.columns:
        out["future_asset_max_drawdown"] = (
            out.get("max_drawdown", pd.Series(0.0, index=out.index)).astype(float) / target_frac_for_crash
        ).clip(lower=-1.0, upper=0.0).astype(np.float32)
    if "asset_crash_label" not in out.columns:
        asset_ret = out.get("future_asset_return", pd.Series(0.0, index=out.index)).astype(float)
        out["asset_crash_label"] = (
            (asset_ret < -0.02)
            | (out["future_min_asset_return"].astype(float) < -0.025)
            | (out["future_asset_max_drawdown"].astype(float) < -0.025)
        ).astype(np.float32)
    if "severe_adverse_label" not in out.columns:
        out["severe_adverse_label"] = (
            (out["future_min_asset_return"].astype(float) < -0.04)
            | (out["future_asset_max_drawdown"].astype(float) < -0.04)
        ).astype(np.float32)
    time_col = "decision_timestamp" if "decision_timestamp" in out.columns else "timestamp"
    group_key = out[time_col].astype(str) + "|" + out["horizon_bars"].astype(str)
    if utility_mode == "default":
        target_return = out["portfolio_return"].astype(float)
        alpha = out["future_alpha_vs_spy"].astype(float)
        profit_label = out["profit_label"].astype(float)
        beat_label = out["beat_spy_label"].astype(float)
        dd_penalty = 0.25
        vol_penalty = 0.10
    elif utility_mode in {"stress_adjusted", "stress_convex", "tradable_stress", "crash_averse"}:
        target_return = _stress_adjusted_return(out, extra_roundtrip_bps)
        alpha = target_return - out["future_spy_return"].astype(float)
        profit_label = (target_return > 0.0).astype(float)
        beat_label = (alpha > 0.0).astype(float)
        dd_penalty = drawdown_penalty
        vol_penalty = volatility_penalty
    else:
        raise ValueError(f"unknown utility mode: {utility_mode}")

    out["allocator_target_return"] = target_return.astype(np.float32)
    out["allocator_target_alpha"] = alpha.astype(np.float32)
    out["allocator_target_profit_label"] = profit_label.astype(np.float32)
    out["allocator_target_beat_spy_label"] = beat_label.astype(np.float32)
    rank_target = target_return
    if utility_mode == "stress_convex":
        raw_return = out["portfolio_return"].astype(float)
        raw_alpha = out["future_alpha_vs_spy"].astype(float)
        rank_target = (
            target_return
            + 0.35 * raw_return.clip(lower=0.0)
            + 0.25 * raw_alpha.clip(lower=0.0)
        )
    elif utility_mode == "tradable_stress":
        rank_target = (
            target_return
            + 0.50 * alpha.clip(lower=-0.05, upper=0.15)
            + 0.08 * profit_label
            + 0.12 * beat_label
            + 0.65 * out["max_drawdown"].astype(float).clip(lower=-0.25, upper=0.0)
            - 0.20 * out["path_vol"].astype(float).clip(lower=0.0, upper=0.20)
        )
    elif utility_mode == "crash_averse":
        max_dd = out["max_drawdown"].astype(float)
        path_vol = out["path_vol"].astype(float)
        asset_ret = out["future_asset_return"].astype(float) if "future_asset_return" in out.columns else target_return
        target_frac = out["target_position_frac"].astype(float).clip(lower=0.0, upper=1.0)
        dd_breach = (max_dd < -0.0125).astype(float)
        deep_dd_breach = (max_dd < -0.025).astype(float)
        losing_trade = (target_return <= 0.0).astype(float)
        underperforming_trade = (alpha <= 0.0).astype(float)
        crash_event = ((asset_ret < -0.02) | (max_dd < -0.025)).astype(float)
        rank_target = (
            1.25 * target_return.clip(lower=-0.08, upper=0.15)
            + 1.50 * alpha.clip(lower=-0.08, upper=0.15)
            + 0.20 * profit_label
            + 0.45 * beat_label
            + 1.75 * max_dd.clip(lower=-0.20, upper=0.0)
            - 0.35 * path_vol.clip(lower=0.0, upper=0.20)
            - 0.35 * target_frac
            - 0.60 * dd_breach
            - 1.10 * deep_dd_breach
            - 0.45 * losing_trade
            - 0.65 * underperforming_trade
            - 1.25 * crash_event
        )
    rank_pct = out.assign(_allocator_rank_target=rank_target).groupby(group_key)["_allocator_rank_target"].rank(pct=True, method="average")
    out["allocator_top_label"] = (rank_pct >= top_quantile).astype(np.float32)
    utility = (
        target_return.clip(-1.0, 3.0)
        + 0.35 * alpha.clip(-1.0, 3.0)
        + 0.15 * profit_label
        + 0.25 * beat_label
        + dd_penalty * out["max_drawdown"].clip(-1.0, 0.0)
        - vol_penalty * out["path_vol"].clip(0.0, 0.20)
    )
    if utility_mode == "stress_convex":
        utility = utility + 0.35 * raw_return.clip(lower=0.0, upper=0.12) + 0.25 * raw_alpha.clip(lower=0.0, upper=0.12)
    elif utility_mode == "tradable_stress":
        target_frac = out["target_position_frac"].astype(float).clip(lower=0.0, upper=1.0)
        state_vol = out["state_vol_1d"].astype(float).clip(lower=0.0, upper=0.20) if "state_vol_1d" in out.columns else 0.0
        utility = (
            utility
            + 0.20 * target_return.clip(lower=0.0, upper=0.12)
            + 0.25 * alpha.clip(lower=0.0, upper=0.12)
            - 0.08 * target_frac
            - 0.15 * state_vol
        )
    elif utility_mode == "crash_averse":
        max_dd = out["max_drawdown"].astype(float)
        path_vol = out["path_vol"].astype(float)
        asset_ret = out["future_asset_return"].astype(float) if "future_asset_return" in out.columns else target_return
        target_frac = out["target_position_frac"].astype(float).clip(lower=0.0, upper=1.0)
        state_vol = out["state_vol_1d"].astype(float).clip(lower=0.0, upper=0.20) if "state_vol_1d" in out.columns else 0.0
        state_ret_1d = out["state_ret_1d"].astype(float) if "state_ret_1d" in out.columns else 0.0
        state_drawdown = out["state_drawdown_5d"].astype(float) if "state_drawdown_5d" in out.columns else 0.0
        dd_breach = (max_dd < -0.0125).astype(float)
        deep_dd_breach = (max_dd < -0.025).astype(float)
        crash_event = ((asset_ret < -0.02) | (max_dd < -0.025)).astype(float)
        weak_context = ((state_ret_1d < -0.02) | (state_drawdown < -0.06)).astype(float)
        utility = (
            1.30 * target_return.clip(lower=-0.08, upper=0.15)
            + 1.60 * alpha.clip(lower=-0.08, upper=0.15)
            + 0.30 * profit_label
            + 0.55 * beat_label
            + 2.25 * max_dd.clip(lower=-0.20, upper=0.0)
            - 0.45 * path_vol.clip(lower=0.0, upper=0.20)
            - 0.40 * target_frac
            - 0.20 * state_vol
            - 0.75 * dd_breach
            - 1.35 * deep_dd_breach
            - 1.50 * crash_event
            - 0.35 * weak_context
        )
    out["allocator_utility"] = utility.astype(np.float32)
    return out


def make_matrices(df: pd.DataFrame, train_mask: np.ndarray, feature_mode: str = "compact") -> dict:
    wanted_features = list(NUMERIC_FEATURES)
    if feature_mode == "market":
        wanted_features += CONTEXT_FEATURES
    elif feature_mode != "compact":
        raise ValueError(f"unknown allocator feature mode: {feature_mode}")
    feature_cols = [c for c in wanted_features if c in df.columns]
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    if "horizon_bars" in feature_cols:
        x[:, feature_cols.index("horizon_bars")] = np.log1p(x[:, feature_cols.index("horizon_bars")])
    if "trade_notional" in feature_cols:
        x[:, feature_cols.index("trade_notional")] /= 50_000.0
    if "price" in feature_cols:
        x[:, feature_cols.index("price")] = np.log1p(np.maximum(x[:, feature_cols.index("price")], 0.0))
    x_mean = x[train_mask].mean(axis=0)
    x_std = np.where(x[train_mask].std(axis=0) < 1e-8, 1.0, x[train_mask].std(axis=0))
    x = np.clip((x - x_mean) / x_std, -20.0, 20.0).astype(np.float32)

    utility = df["allocator_utility"].to_numpy(np.float32)
    utility_mean = float(utility[train_mask].mean())
    utility_std = float(max(utility[train_mask].std(), 1e-8))
    utility_scaled = ((utility - utility_mean) / utility_std).astype(np.float32)
    top = df["allocator_top_label"].to_numpy(np.float32)

    symbols = sorted(df["symbol"].astype(str).unique().tolist())
    actions = sorted(action_keys(df).unique().tolist())
    sym_to_id = {s: i for i, s in enumerate(symbols)}
    action_to_id = {a: i for i, a in enumerate(actions)}
    symbol_id = df["symbol"].astype(str).map(sym_to_id).fillna(len(symbols)).to_numpy(np.int64)
    action_id = action_keys(df).map(action_to_id).fillna(len(actions)).to_numpy(np.int64)

    return {
        "x": x,
        "symbol_id": symbol_id,
        "action_id": action_id,
        "utility": utility_scaled,
        "utility_raw": utility,
        "top": top,
        "feature_cols": feature_cols,
        "symbols": symbols,
        "actions": actions,
        "x_mean": x_mean,
        "x_std": x_std,
        "utility_mean": utility_mean,
        "utility_std": utility_std,
    }


def dataset(mats: dict, mask: np.ndarray) -> TensorDataset:
    idx = np.where(mask)[0]
    return TensorDataset(
        torch.from_numpy(mats["x"][idx]),
        torch.from_numpy(mats["symbol_id"][idx]),
        torch.from_numpy(mats["action_id"][idx]),
        torch.from_numpy(mats["utility"][idx]),
        torch.from_numpy(mats["top"][idx]),
    )


def evaluate_model(model: AllocatorModel, loader: DataLoader, device: str, top_pos_weight: float = 1.0) -> dict:
    model.eval()
    mse = nn.MSELoss()
    pos_weight = torch.tensor([max(float(top_pos_weight), 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    losses, utility_losses, top_losses = [], [], []
    correct = total = 0
    with torch.no_grad():
        for xb, sid, aid, yutil, ytop in loader:
            xb, sid, aid = xb.to(device), sid.to(device), aid.to(device)
            yutil, ytop = yutil.to(device), ytop.to(device)
            pred_util, pred_top = model(xb, sid, aid)
            util_loss = mse(pred_util, yutil)
            top_loss = bce(pred_top, ytop)
            loss = util_loss + 0.75 * top_loss
            losses.append(float(loss.item()))
            utility_losses.append(float(util_loss.item()))
            top_losses.append(float(top_loss.item()))
            correct += int(((torch.sigmoid(pred_top) >= 0.5) == (ytop >= 0.5)).sum().item())
            total += int(ytop.numel())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "utility_mse": float(np.mean(utility_losses)) if utility_losses else 0.0,
        "top_bce": float(np.mean(top_losses)) if top_losses else 0.0,
        "top_accuracy": float(correct / max(total, 1)),
    }


def apply_allocator(scored: pd.DataFrame, ckpt: dict, device: str, batch_size: int) -> pd.DataFrame:
    feature_cols = ckpt["feature_cols"]
    x = scored[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    if "horizon_bars" in feature_cols:
        x[:, feature_cols.index("horizon_bars")] = np.log1p(x[:, feature_cols.index("horizon_bars")])
    if "trade_notional" in feature_cols:
        x[:, feature_cols.index("trade_notional")] /= 50_000.0
    if "price" in feature_cols:
        x[:, feature_cols.index("price")] = np.log1p(np.maximum(x[:, feature_cols.index("price")], 0.0))
    x = np.clip((x - ckpt["x_mean"]) / ckpt["x_std"], -20.0, 20.0).astype(np.float32)

    sym_to_id = {s: i for i, s in enumerate(ckpt["symbols"])}
    action_to_id = {a: i for i, a in enumerate(ckpt["actions"])}
    symbol_id = scored["symbol"].astype(str).map(sym_to_id).fillna(len(sym_to_id)).to_numpy(np.int64)
    action_id = action_keys(scored).map(action_to_id).fillna(len(action_to_id)).to_numpy(np.int64)

    model = AllocatorModel(
        n_features=len(feature_cols),
        n_symbols=len(ckpt["symbols"]),
        n_actions=len(ckpt["actions"]),
        hidden_dim=ckpt["config"]["hidden_dim"],
        n_layers=ckpt["config"]["n_layers"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scores = []
    with torch.no_grad():
        for i in range(0, len(scored), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(device)
            sid = torch.from_numpy(symbol_id[i : i + batch_size].copy()).to(device)
            aid = torch.from_numpy(action_id[i : i + batch_size].copy()).to(device)
            pred_util, pred_top = model(xb, sid, aid)
            score = pred_util * ckpt["utility_std"] + ckpt["utility_mean"] + 0.15 * torch.sigmoid(pred_top)
            scores.append(score.detach().cpu().numpy())
    out = scored.copy()
    out["pred_score"] = np.concatenate(scores, axis=0)
    return out


def train(config: AllocatorConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = pick_device(config.device)
    world_ckpt = torch.load(config.world_checkpoint, map_location="cpu", weights_only=False)

    train_df = filter_frame(load_frame(Path(config.train_data), config.limit_rows, seed=config.seed), config.min_horizon_bars, config.max_horizon_bars)
    train_scored = predict(train_df, world_ckpt, device=device, batch_size=config.batch_size)
    if config.train_entry_only:
        train_scored = entry_only_frame(train_scored)
    train_scored = add_targets(
        train_scored,
        config.top_quantile,
        utility_mode=config.utility_mode,
        extra_roundtrip_bps=config.extra_roundtrip_bps,
        drawdown_penalty=config.drawdown_penalty,
        volatility_penalty=config.volatility_penalty,
    )
    train_mask, val_mask = split_masks(train_scored, config.val_fraction, config.val_gap_days)
    mats = make_matrices(train_scored, train_mask, feature_mode=config.feature_mode)
    top_positive_rate = float(mats["top"][train_mask].mean())
    top_pos_weight = float((1.0 - top_positive_rate) / max(top_positive_rate, 1e-6))

    train_loader = DataLoader(dataset(mats, train_mask), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset(mats, val_mask), batch_size=config.batch_size * 2, shuffle=False)
    model = AllocatorModel(
        n_features=mats["x"].shape[1],
        n_symbols=len(mats["symbols"]),
        n_actions=len(mats["actions"]),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    top_pos_weight_tensor = torch.tensor([max(top_pos_weight, 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=top_pos_weight_tensor)
    history = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    print(
        f"[allocator] rows={len(train_scored):,} train={int(train_mask.sum()):,} val={int(val_mask.sum()):,} "
        f"features={mats['x'].shape[1]} symbols={len(mats['symbols'])} actions={len(mats['actions'])} "
        f"top_pos_rate={top_positive_rate:.3f} top_pos_weight={top_pos_weight:.2f} device={device}",
        flush=True,
    )
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, sid, aid, yutil, ytop in train_loader:
            xb, sid, aid = xb.to(device), sid.to(device), aid.to(device)
            yutil, ytop = yutil.to(device), ytop.to(device)
            opt.zero_grad(set_to_none=True)
            pred_util, pred_top = model(xb, sid, aid)
            loss = mse(pred_util, yutil) + 0.75 * bce(pred_top, ytop)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate_model(model, val_loader, device, top_pos_weight=top_pos_weight)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)) if losses else 0.0, **val}
        history.append(row)
        print(
            f"[allocator] epoch {epoch+1}/{config.epochs} train={row['train_loss']:.4f} "
            f"val={row['loss']:.4f} top_acc={row['top_accuracy']:.3f}",
            flush=True,
        )
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_cols": mats["feature_cols"],
        "symbols": mats["symbols"],
        "actions": mats["actions"],
        "x_mean": mats["x_mean"],
        "x_std": mats["x_std"],
        "utility_mean": mats["utility_mean"],
        "utility_std": mats["utility_std"],
        "top_positive_rate": top_positive_rate,
        "top_pos_weight": top_pos_weight,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output)

    val_scored = apply_allocator(train_scored.loc[val_mask].reset_index(drop=True), ckpt, device=device, batch_size=config.batch_size)
    val_result = evaluate_planner(val_scored)
    test_df = filter_frame(load_frame(Path(config.test_data), config.limit_rows, seed=config.seed), config.min_horizon_bars, config.max_horizon_bars)
    test_scored = predict(test_df, world_ckpt, device=device, batch_size=config.batch_size)
    if config.train_entry_only:
        test_scored = entry_only_frame(test_scored)
    test_scored = apply_allocator(test_scored, ckpt, device=device, batch_size=config.batch_size)
    test_result = evaluate_planner(test_scored)

    metrics = {
        "checkpoint": str(output),
        "world_checkpoint": config.world_checkpoint,
        "rows_scored_train": int(len(train_scored)),
        "rows_scored_test": int(len(test_scored)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "min_horizon_bars": int(config.min_horizon_bars),
        "max_horizon_bars": int(config.max_horizon_bars),
        "feature_mode": config.feature_mode,
        "utility_mode": config.utility_mode,
        "train_entry_only": bool(config.train_entry_only),
        "extra_roundtrip_bps": float(config.extra_roundtrip_bps),
        "drawdown_penalty": float(config.drawdown_penalty),
        "volatility_penalty": float(config.volatility_penalty),
        "device": device,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "last": history[-1] if history else {},
        "validation_planner": val_result,
        "test_planner": test_result,
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--val-gap-days", type=float, default=7.0)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=0)
    parser.add_argument("--top-quantile", type=float, default=0.80)
    parser.add_argument("--feature-mode", choices=["compact", "market"], default="compact")
    parser.add_argument("--utility-mode", choices=["default", "stress_adjusted", "stress_convex", "tradable_stress", "crash_averse"], default="default")
    parser.add_argument("--train-entry-only", action="store_true", help="train and report allocator only on cash-to-buy entry rows")
    parser.add_argument("--extra-roundtrip-bps", type=float, default=0.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.25)
    parser.add_argument("--volatility-penalty", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(AllocatorConfig(**vars(args)))


if __name__ == "__main__":
    main()
