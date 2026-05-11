"""Train the first action-conditioned portfolio world model.

The model consumes rows from world_model_dataset.py:

    market/portfolio state + action + horizon -> future portfolio outcome

This is a standalone trainer, deliberately separate from experiment.py. It is
the first baseline for the new world-model direction: a compact tabular latent
model with categorical embeddings and multi-task outcome heads.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CHECKPOINT_DIR = Path("checkpoints/world_model")
DATA_DIR = Path("data/world_model/top100_train_counterfactual")
TARGET_REGRESSION = [
    "portfolio_return",
    "max_drawdown",
    "path_vol",
    "future_alpha_vs_spy",
]
TARGET_CLASSIFICATION = ["profit_label", "beat_spy_label"]
TARGET_CLIPS = {
    "portfolio_return": (-1.0, 3.0),
    "max_drawdown": (-1.0, 0.0),
    "path_vol": (0.0, 0.20),
    "future_alpha_vs_spy": (-1.0, 3.0),
}


@dataclass(frozen=True)
class TrainConfig:
    data: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    n_layers: int
    dropout: float
    val_fraction: float
    val_gap_days: float
    seed: int
    device: str
    limit_rows: int
    min_horizon_bars: int
    max_horizon_bars: int
    symbol_dropout: float
    rank_loss_coef: float
    patience: int
    min_delta: float
    init_checkpoint: str
    output: str


class PortfolioWorldModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_symbols: int,
        n_actions: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        sym_dim = min(32, max(8, int(math.ceil(math.sqrt(max(n_symbols, 1))) * 2)))
        action_dim = 8
        self.unk_symbol_id = n_symbols
        self.symbol_emb = nn.Embedding(n_symbols + 1, sym_dim)
        self.action_emb = nn.Embedding(n_actions, action_dim)
        input_dim = n_features + sym_dim + action_dim
        blocks: list[nn.Module] = []
        dim = input_dim
        for _ in range(max(1, n_layers)):
            blocks.extend([
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            dim = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.reg_head = nn.Linear(hidden_dim, len(TARGET_REGRESSION))
        self.cls_head = nn.Linear(hidden_dim, len(TARGET_CLASSIFICATION))
        self.rank_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, symbol_id: torch.Tensor, action_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.cat([x, self.symbol_emb(symbol_id), self.action_emb(action_id)], dim=-1)
        h = self.trunk(z)
        return self.reg_head(h), self.cls_head(h), self.rank_head(h).squeeze(-1)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def action_keys(df: pd.DataFrame) -> pd.Series:
    if {"action", "current_position_frac", "target_position_frac"}.issubset(df.columns):
        current = df["current_position_frac"].astype(float).map(lambda x: f"{x:.2f}")
        target = df["target_position_frac"].astype(float).map(lambda x: f"{x:.2f}")
        return df["action"].astype(str) + "|" + current + "->" + target
    return df["action"].astype(str)


def load_frame(path: Path, limit_rows: int = 0, seed: int = 0) -> pd.DataFrame:
    def sort_cols(frame: pd.DataFrame) -> list[str]:
        cols = ["decision_timestamp" if "decision_timestamp" in frame.columns else "timestamp"]
        if "symbol" in frame.columns:
            cols.append("symbol")
        return cols

    if path.is_dir():
        frames = []
        shards = sorted(path.glob("*.parquet"))
        if limit_rows > 0 and shards:
            per_shard = max(1, int(math.ceil(limit_rows / len(shards))))
        else:
            per_shard = 0
        for shard_idx, shard in enumerate(shards):
            df = pd.read_parquet(shard)
            if limit_rows > 0:
                n = min(per_shard, len(df))
                if n < len(df):
                    df = df.sample(n=n, random_state=seed + shard_idx).sort_values(sort_cols(df))
            frames.append(df)
        if not frames:
            raise RuntimeError(f"no parquet shards found under {path}")
        out = pd.concat(frames, ignore_index=True)
        if limit_rows > 0 and len(out) > limit_rows:
            out = out.sample(n=limit_rows, random_state=seed).sort_values(sort_cols(out))
        return out.reset_index(drop=True)
    df = pd.read_parquet(path)
    return df.head(limit_rows) if limit_rows > 0 else df


def make_matrices(df: pd.DataFrame, val_fraction: float, val_gap_days: float) -> dict:
    feature_cols = [
        c for c in df.columns
        if c.startswith("feat_") or c.startswith("state_") or c.startswith("xsec_")
    ]
    feature_cols += [
        "price",
        "horizon_bars",
        "current_position_frac",
        "target_position_frac",
        "trade_notional",
        "fees",
        "slippage",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    symbols = sorted(df["symbol"].astype(str).unique().tolist())
    action_key = action_keys(df)
    actions = sorted(action_key.unique().tolist())
    sym_to_id = {s: i for i, s in enumerate(symbols)}
    action_to_id = {a: i for i, a in enumerate(actions)}

    time_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    timestamps = pd.to_datetime(df[time_col], utc=True)
    cutoff = timestamps.quantile(max(0.0, min(1.0, 1.0 - val_fraction)))
    train_cutoff = cutoff - pd.Timedelta(days=max(0.0, val_gap_days))
    train_mask = (timestamps <= train_cutoff).to_numpy()
    val_mask = (timestamps > cutoff).to_numpy()
    if val_mask.sum() == 0:
        rng = np.random.default_rng(0)
        val_mask = rng.random(len(df)) < val_fraction
        train_mask = ~val_mask

    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    x[:, feature_cols.index("horizon_bars")] = np.log1p(x[:, feature_cols.index("horizon_bars")])
    if "trade_notional" in feature_cols:
        x[:, feature_cols.index("trade_notional")] /= 50_000.0
    if "price" in feature_cols:
        x[:, feature_cols.index("price")] = np.log1p(np.maximum(x[:, feature_cols.index("price")], 0.0))

    y_reg_df = df[TARGET_REGRESSION].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col, (lo, hi) in TARGET_CLIPS.items():
        if col in y_reg_df.columns:
            y_reg_df[col] = y_reg_df[col].clip(lo, hi)
    y_reg = y_reg_df.to_numpy(np.float32)
    y_cls = df[TARGET_CLASSIFICATION].astype(np.float32).to_numpy(np.float32)
    group_key = (
        df[time_col].astype(str)
        + "|"
        + df["horizon_bars"].astype(str)
    )
    rank_pct = df.groupby(group_key)["portfolio_return"].rank(pct=True, method="average").to_numpy(np.float32)
    y_rank = (rank_pct >= 0.75).astype(np.float32)
    group_id = pd.factorize(group_key, sort=False)[0].astype(np.int64)
    symbol_id = df["symbol"].astype(str).map(sym_to_id).to_numpy(np.int64)
    action_id = action_key.map(action_to_id).to_numpy(np.int64)

    x_mean = x[train_mask].mean(axis=0)
    x_std = x[train_mask].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    x = (x - x_mean) / x_std
    x = np.clip(x, -20.0, 20.0).astype(np.float32)

    y_mean = y_reg[train_mask].mean(axis=0)
    y_std = y_reg[train_mask].std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)
    y_reg_scaled = ((y_reg - y_mean) / y_std).astype(np.float32)

    return {
        "x": x,
        "symbol_id": symbol_id,
        "action_id": action_id,
        "y_reg": y_reg_scaled,
        "y_reg_raw": y_reg,
        "y_cls": y_cls,
        "y_rank": y_rank,
        "group_id": group_id,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "feature_cols": feature_cols,
        "time_col": time_col,
        "symbols": symbols,
        "actions": actions,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def tensor_dataset(mats: dict, mask: np.ndarray) -> TensorDataset:
    idx = np.where(mask)[0]
    return TensorDataset(
        torch.from_numpy(mats["x"][idx]),
        torch.from_numpy(mats["symbol_id"][idx]),
        torch.from_numpy(mats["action_id"][idx]),
        torch.from_numpy(mats["y_reg"][idx]),
        torch.from_numpy(mats["y_cls"][idx]),
        torch.from_numpy(mats["y_rank"][idx]),
        torch.from_numpy(mats["group_id"][idx]),
        torch.from_numpy(mats["y_reg_raw"][idx]),
    )


def evaluate(
    model: PortfolioWorldModel,
    loader: DataLoader,
    device: str,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> dict[str, float]:
    model.eval()
    losses, mae_raw, bce_losses, rank_losses = [], [], [], []
    correct_profit = correct_spy = total = 0
    mse = nn.MSELoss(reduction="mean")
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    y_mean_t = torch.from_numpy(y_mean).to(device)
    y_std_t = torch.from_numpy(y_std).to(device)
    with torch.no_grad():
        for xb, sid, aid, yreg, ycls, yrank, _gid, yraw in loader:
            xb = xb.to(device)
            sid = sid.to(device)
            aid = aid.to(device)
            yreg = yreg.to(device)
            ycls = ycls.to(device)
            yraw = yraw.to(device)
            yrank = yrank.to(device)
            pred_reg, pred_cls, pred_rank = model(xb, sid, aid)
            reg_loss = mse(pred_reg, yreg)
            cls_loss = bce(pred_cls, ycls)
            rank_loss = bce(pred_rank, yrank)
            losses.append(float((reg_loss + 0.5 * cls_loss + rank_loss).item()))
            bce_losses.append(float(cls_loss.item()))
            rank_losses.append(float(rank_loss.item()))
            pred_raw = pred_reg * y_std_t + y_mean_t
            mae_raw.append(torch.mean(torch.abs(pred_raw - yraw), dim=0).detach().cpu().numpy())
            probs = torch.sigmoid(pred_cls)
            correct_profit += int(((probs[:, 0] >= 0.5) == (ycls[:, 0] >= 0.5)).sum().item())
            correct_spy += int(((probs[:, 1] >= 0.5) == (ycls[:, 1] >= 0.5)).sum().item())
            total += int(ycls.size(0))
    mae = np.mean(np.stack(mae_raw), axis=0) if mae_raw else np.zeros(len(TARGET_REGRESSION))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "bce": float(np.mean(bce_losses)) if bce_losses else 0.0,
        "rank_bce": float(np.mean(rank_losses)) if rank_losses else 0.0,
        "mae_portfolio_return": float(mae[0]),
        "mae_max_drawdown": float(mae[1]),
        "mae_path_vol": float(mae[2]),
        "mae_future_alpha_vs_spy": float(mae[3]),
        "profit_accuracy": float(correct_profit / max(total, 1)),
        "beat_spy_accuracy": float(correct_spy / max(total, 1)),
    }


def train(config: TrainConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = pick_device(config.device)
    started = time.time()
    df = load_frame(Path(config.data), config.limit_rows, seed=config.seed)
    if config.min_horizon_bars > 0:
        df = df[df["horizon_bars"] >= config.min_horizon_bars].reset_index(drop=True)
    if config.max_horizon_bars > 0:
        df = df[df["horizon_bars"] <= config.max_horizon_bars].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("no rows left after horizon filtering")
    mats = make_matrices(df, config.val_fraction, config.val_gap_days)
    train_ds = tensor_dataset(mats, mats["train_mask"])
    val_ds = tensor_dataset(mats, mats["val_mask"])
    pin_memory = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size * 2, shuffle=False, num_workers=0, pin_memory=pin_memory)
    model = PortfolioWorldModel(
        n_features=mats["x"].shape[1],
        n_symbols=len(mats["symbols"]),
        n_actions=len(mats["actions"]),
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    if config.init_checkpoint:
        ckpt = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
        expected = {
            "feature_cols": mats["feature_cols"],
            "symbols": mats["symbols"],
            "actions": mats["actions"],
        }
        for key, value in expected.items():
            if list(ckpt.get(key, [])) != list(value):
                raise RuntimeError(f"init checkpoint {key} does not match current training data")
        model.load_state_dict(ckpt["state_dict"])
        print(f"[world-train] initialized from {config.init_checkpoint}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss(reduction="mean")
    bce = nn.BCEWithLogitsLoss(reduction="mean")

    history = []
    print(
        f"[world-train] rows={len(df):,} train={len(train_ds):,} val={len(val_ds):,} "
        f"features={mats['x'].shape[1]} symbols={len(mats['symbols'])} actions={len(mats['actions'])} device={device}",
        flush=True,
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for xb, sid, aid, yreg, ycls, yrank, _gid, _yraw in train_loader:
            xb = xb.to(device)
            sid = sid.to(device)
            aid = aid.to(device)
            yreg = yreg.to(device)
            ycls = ycls.to(device)
            yrank = yrank.to(device)
            if config.symbol_dropout > 0:
                mask = torch.rand_like(sid.float()) < config.symbol_dropout
                sid = sid.masked_fill(mask, model.unk_symbol_id)
            opt.zero_grad(set_to_none=True)
            pred_reg, pred_cls, pred_rank = model(xb, sid, aid)
            loss = (
                mse(pred_reg, yreg)
                + 0.5 * bce(pred_cls, ycls)
                + config.rank_loss_coef * bce(pred_rank, yrank)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))
        val_metrics = evaluate(model, val_loader, device, mats["y_mean"], mats["y_std"])
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            **val_metrics,
        }
        history.append(row)
        print(
            f"[world-train] epoch {epoch+1}/{config.epochs} "
            f"train={row['train_loss']:.4f} val={row['loss']:.4f} "
            f"mae_ret={row['mae_portfolio_return']:.6f} "
            f"profit_acc={row['profit_accuracy']:.3f} beat_spy_acc={row['beat_spy_accuracy']:.3f}",
            flush=True,
        )
        if row["loss"] < best_loss - config.min_delta:
            best_loss = row["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
            if config.patience > 0 and stale_epochs >= config.patience:
                print(f"[world-train] early stop after {epoch+1} epochs (best={best_epoch})", flush=True)
                break

    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_cols": mats["feature_cols"],
        "symbols": mats["symbols"],
        "actions": mats["actions"],
        "target_regression": TARGET_REGRESSION,
        "target_classification": TARGET_CLASSIFICATION,
        "x_mean": mats["x_mean"],
        "x_std": mats["x_std"],
        "y_mean": mats["y_mean"],
        "y_std": mats["y_std"],
        "history": history,
        "elapsed_seconds": time.time() - started,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "init_checkpoint": config.init_checkpoint,
    }
    torch.save(payload, output)
    metrics = {
        "checkpoint": str(output),
        "rows": int(len(df)),
        "min_horizon_bars": int(config.min_horizon_bars),
        "max_horizon_bars": int(config.max_horizon_bars),
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "device": device,
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "best_val_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "last": history[-1] if history else {},
        "best": history[best_epoch - 1] if best_epoch > 0 else {},
    }
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[world-train] saved {output}", flush=True)
    print(f"[world-train] metrics {metrics_path}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--val-gap-days", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=os.environ.get("WORLD_MODEL_DEVICE", "auto"))
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--min-horizon-bars", type=int, default=0)
    parser.add_argument("--max-horizon-bars", type=int, default=0)
    parser.add_argument("--symbol-dropout", type=float, default=0.10)
    parser.add_argument("--rank-loss-coef", type=float, default=0.50)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--output", default=str(CHECKPOINT_DIR / "world_model_v1.pt"))
    args = parser.parse_args()
    train(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
