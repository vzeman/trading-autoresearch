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
    "pred_profit_label",
    "pred_beat_spy_label",
    "pred_rank_top_quartile",
    "price",
    "horizon_bars",
    "current_position_frac",
    "target_position_frac",
    "trade_notional",
    "fees",
    "slippage",
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


def add_targets(df: pd.DataFrame, top_quantile: float) -> pd.DataFrame:
    out = df.copy()
    time_col = "decision_timestamp" if "decision_timestamp" in out.columns else "timestamp"
    group_key = out[time_col].astype(str) + "|" + out["horizon_bars"].astype(str)
    rank_pct = out.groupby(group_key)["portfolio_return"].rank(pct=True, method="average")
    out["allocator_top_label"] = (rank_pct >= top_quantile).astype(np.float32)
    out["allocator_utility"] = (
        out["portfolio_return"].clip(-1.0, 3.0)
        + 0.35 * out["future_alpha_vs_spy"].clip(-1.0, 3.0)
        + 0.15 * out["profit_label"]
        + 0.25 * out["beat_spy_label"]
        + 0.25 * out["max_drawdown"].clip(-1.0, 0.0)
        - 0.10 * out["path_vol"].clip(0.0, 0.20)
    ).astype(np.float32)
    return out


def make_matrices(df: pd.DataFrame, train_mask: np.ndarray) -> dict:
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
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


def evaluate_model(model: AllocatorModel, loader: DataLoader, device: str) -> dict:
    model.eval()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
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
    train_scored = add_targets(train_scored, config.top_quantile)
    train_mask, val_mask = split_masks(train_scored, config.val_fraction, config.val_gap_days)
    mats = make_matrices(train_scored, train_mask)

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
    bce = nn.BCEWithLogitsLoss()
    history = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    print(
        f"[allocator] rows={len(train_scored):,} train={int(train_mask.sum()):,} val={int(val_mask.sum()):,} "
        f"features={mats['x'].shape[1]} symbols={len(mats['symbols'])} actions={len(mats['actions'])} device={device}",
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
        val = evaluate_model(model, val_loader, device)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(AllocatorConfig(**vars(args)))


if __name__ == "__main__":
    main()
