"""Train a date-listwise cross-sectional daily ranker.

Unlike the row-wise daily ranker, this model sees one decision date at a time
and optimizes a listwise objective: put probability mass on the best stocks in
that date's cross-section.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_daily_ranker import Config as RuleConfig
from train_daily_ranker import FEATURE_COLS, choose_rule, pick_device, simulate


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    train_end: str
    test_start: str
    test_end: str
    validation_fraction: float
    epochs: int
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    list_temperature: float
    utility_mode: str
    downside_penalty: float
    profit_loss_weight: float
    crash_loss_weight: float
    score_profit_weight: float
    score_top_weight: float
    score_crash_weight: float
    batch_size: int
    top_k: int
    max_positions: int
    horizon_days: int
    min_validation_trades: int
    min_validation_return: float
    min_validation_active_alpha: float
    min_validation_profit_rate: float
    min_validation_beat_spy_rate: float
    max_validation_drawdown: float
    rule_validation_fraction: float
    min_rule_validation_trades: int
    device: str
    seed: int


class ListwiseRanker(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(hidden_dim, 1)
        self.profit = nn.Linear(hidden_dim, 1)
        self.crash = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.score(h).squeeze(-1), self.profit(h).squeeze(-1), self.crash(h).squeeze(-1)


def split_masks(df: pd.DataFrame, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(df["date"], utc=True)
    train_end = pd.Timestamp(config.train_end, tz="UTC")
    test_start = pd.Timestamp(config.test_start, tz="UTC")
    test_end = pd.Timestamp(config.test_end, tz="UTC")
    train_all = dates < train_end
    train_dates = dates[train_all]
    val_cutoff = train_dates.quantile(1.0 - max(0.0, min(config.validation_fraction, 0.8)))
    train_mask = (dates < val_cutoff).to_numpy()
    val_mask = ((dates >= val_cutoff) & (dates < train_end)).to_numpy()
    test_mask = ((dates >= test_start) & (dates < test_end)).to_numpy()
    return train_mask, val_mask, test_mask


def make_arrays(df: pd.DataFrame, train_mask: np.ndarray, config: Config) -> dict:
    x = df[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    mean = x[train_mask].mean(axis=0)
    std = np.where(x[train_mask].std(axis=0) < 1e-8, 1.0, x[train_mask].std(axis=0))
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    if config.utility_mode == "utility":
        y = df["target_utility"].astype(float).to_numpy(np.float32)
    elif config.utility_mode == "alpha":
        y = df["future_alpha"].astype(float).to_numpy(np.float32)
    elif config.utility_mode == "risk_adjusted_alpha":
        alpha = df["future_alpha"].astype(float).to_numpy(np.float32)
        downside = np.maximum(-df["future_min_return"].astype(float).to_numpy(np.float32), 0.0)
        y = alpha - float(config.downside_penalty) * downside
    else:
        raise ValueError(f"unsupported utility_mode: {config.utility_mode}")
    return {
        "x": x,
        "target": y,
        "profit": df["profit_label"].astype(float).to_numpy(np.float32),
        "crash": df["crash_label"].astype(float).to_numpy(np.float32),
        "x_mean": mean,
        "x_std": std,
    }


def date_groups(df: pd.DataFrame, mask: np.ndarray) -> list[np.ndarray]:
    work = pd.DataFrame({"idx": np.where(mask)[0], "date": pd.to_datetime(df.loc[mask, "date"], utc=True).to_numpy()})
    return [g["idx"].to_numpy(np.int64) for _, g in work.groupby("date", sort=True)]


def train_model(df: pd.DataFrame, config: Config, train_mask: np.ndarray, val_mask: np.ndarray) -> tuple[ListwiseRanker, dict, dict]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = pick_device(config.device)
    arrays = make_arrays(df, train_mask, config)
    model = ListwiseRanker(len(FEATURE_COLS), config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    train_groups = date_groups(df, train_mask)
    val_groups = date_groups(df, val_mask)
    best = math.inf
    best_state = None
    history = []
    print(f"[listwise-ranker] train_dates={len(train_groups):,} val_dates={len(val_groups):,} device={device}", flush=True)
    for epoch in range(config.epochs):
        model.train()
        np.random.shuffle(train_groups)
        losses = []
        for idx in train_groups:
            xb = torch.from_numpy(arrays["x"][idx]).to(device)
            target = torch.from_numpy(arrays["target"][idx]).to(device)
            profit = torch.from_numpy(arrays["profit"][idx]).to(device)
            crash = torch.from_numpy(arrays["crash"][idx]).to(device)
            opt.zero_grad(set_to_none=True)
            score, pred_profit, pred_crash = model(xb)
            target_prob = torch.softmax(target / max(config.list_temperature, 1e-6), dim=0)
            rank_loss = -(target_prob * torch.log_softmax(score, dim=0)).sum()
            aux_loss = (
                float(config.profit_loss_weight) * bce(pred_profit, profit)
                + float(config.crash_loss_weight) * bce(pred_crash, crash)
            )
            loss = rank_loss + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        val_loss = evaluate_loss(model, arrays, val_groups, device, config.list_temperature)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(row)
        print(f"[listwise-ranker] epoch {epoch+1}/{config.epochs} train={row['train_loss']:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, arrays, {"history": history, "best_val_loss": best, "device": device}


def evaluate_loss(model: ListwiseRanker, arrays: dict, groups: list[np.ndarray], device: str, temperature: float) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for idx in groups:
            xb = torch.from_numpy(arrays["x"][idx]).to(device)
            target = torch.from_numpy(arrays["target"][idx]).to(device)
            score, _, _ = model(xb)
            target_prob = torch.softmax(target / max(temperature, 1e-6), dim=0)
            losses.append(float((-(target_prob * torch.log_softmax(score, dim=0)).sum()).item()))
    return float(np.mean(losses)) if losses else 0.0


def score_frame(
    model: ListwiseRanker,
    df: pd.DataFrame,
    arrays: dict,
    mask: np.ndarray,
    device: str,
    config: Config,
) -> pd.DataFrame:
    idx = np.where(mask)[0]
    x = torch.from_numpy(arrays["x"][idx])
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), config.batch_size):
            xb = x[start:start + config.batch_size].to(device)
            score, profit, crash = model(xb)
            preds.append(torch.stack([score, torch.sigmoid(profit), torch.sigmoid(crash)], dim=1).cpu().numpy())
    arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 3), dtype=np.float32)
    out = df.iloc[idx].copy().reset_index(drop=True)
    out["pred_utility"] = arr[:, 0]
    out["pred_profit"] = arr[:, 1]
    out["pred_crash"] = arr[:, 2]
    out["pred_top"] = out.groupby("date")["pred_utility"].rank(pct=True).astype(float)
    out["pred_score"] = (
        out["pred_utility"]
        + float(config.score_profit_weight) * out["pred_profit"]
        + float(config.score_top_weight) * out["pred_top"]
        - float(config.score_crash_weight) * out["pred_crash"]
    )
    return out


def rule_config(config: Config) -> RuleConfig:
    return RuleConfig(
        output_dir=config.output_dir,
        start_date="",
        end_date="",
        train_end=config.train_end,
        test_start=config.test_start,
        test_end=config.test_end,
        horizon_days=config.horizon_days,
        top500=True,
        cached_all=True,
        symbol_limit=0,
        min_rows=300,
        epochs=config.epochs,
        batch_size=config.batch_size,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        lr=config.lr,
        weight_decay=config.weight_decay,
        validation_fraction=config.validation_fraction,
        min_validation_trades=config.min_validation_trades,
        min_validation_return=config.min_validation_return,
        min_validation_active_alpha=config.min_validation_active_alpha,
        min_validation_profit_rate=config.min_validation_profit_rate,
        min_validation_beat_spy_rate=config.min_validation_beat_spy_rate,
        max_validation_drawdown=config.max_validation_drawdown,
        rule_validation_fraction=config.rule_validation_fraction,
        min_rule_validation_trades=config.min_rule_validation_trades,
        top_k=config.top_k,
        max_positions=config.max_positions,
        observed_score_weight=0.0,
        utility_mode="alpha",
        device=config.device,
        seed=config.seed,
    )


def run(config: Config) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(config.dataset)
    train_mask, val_mask, test_mask = split_masks(df, config)
    model, arrays, meta = train_model(df, config, train_mask, val_mask)
    device = meta["device"]
    val_scored = score_frame(model, df, arrays, val_mask, device, config)
    test_scored = score_frame(model, df, arrays, test_mask, device, config)
    choice = choose_rule(val_scored, rule_config(config))
    rule = choice["rule"]
    test_result = simulate(
        test_scored,
        config.top_k,
        config.max_positions,
        config.horizon_days,
        rule.get("score_threshold", float("inf")),
        rule.get("min_profit", 1.0),
        rule.get("max_crash", 0.0),
        min_spy_ret_20d=rule.get("min_spy_ret_20d"),
        min_ret_20d=rule.get("min_ret_20d"),
        min_rel_spy_20d=rule.get("min_rel_spy_20d"),
        min_drawdown_60d=rule.get("min_drawdown_60d"),
        max_vol_20d_rank=rule.get("max_vol_20d_rank"),
        min_mkt_pct_positive_20d=rule.get("min_mkt_pct_positive_20d"),
        min_mkt_pct_above_ma20=rule.get("min_mkt_pct_above_ma20"),
        min_mkt_ret_20d_mean=rule.get("min_mkt_ret_20d_mean"),
        max_mkt_ret_20d_dispersion=rule.get("max_mkt_ret_20d_dispersion"),
    )
    ckpt = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "feature_cols": FEATURE_COLS,
        "arrays": {"x_mean": arrays["x_mean"], "x_std": arrays["x_std"]},
        "rule": rule,
        "train_meta": meta,
    }
    torch.save(ckpt, output_dir / "daily_listwise_ranker.pt")
    payload = {
        "config": asdict(config),
        "rows": int(len(df)),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_meta": meta,
        "rule_selection": choice,
        "test_result": test_result,
        "checkpoint": str(output_dir / "daily_listwise_ranker.pt"),
    }
    (output_dir / "daily_listwise_ranker_result.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/daily_ranker/exp11_latest_dataset_h5_2026/daily_ranker_dataset.parquet")
    parser.add_argument("--output-dir", default="checkpoints/daily_listwise_ranker/exp1_train2025_2026")
    parser.add_argument("--train-end", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="2026-05-10")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--list-temperature", type=float, default=0.05)
    parser.add_argument("--utility-mode", choices=("utility", "alpha", "risk_adjusted_alpha"), default="utility")
    parser.add_argument("--downside-penalty", type=float, default=2.0)
    parser.add_argument("--profit-loss-weight", type=float, default=0.15)
    parser.add_argument("--crash-loss-weight", type=float, default=0.30)
    parser.add_argument("--score-profit-weight", type=float, default=0.04)
    parser.add_argument("--score-top-weight", type=float, default=0.08)
    parser.add_argument("--score-crash-weight", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--min-validation-trades", type=int, default=20)
    parser.add_argument("--min-validation-return", type=float, default=-0.05)
    parser.add_argument("--min-validation-active-alpha", type=float, default=-0.05)
    parser.add_argument("--min-validation-profit-rate", type=float, default=0.45)
    parser.add_argument("--min-validation-beat-spy-rate", type=float, default=0.45)
    parser.add_argument("--max-validation-drawdown", type=float, default=0.20)
    parser.add_argument("--rule-validation-fraction", type=float, default=0.25)
    parser.add_argument("--min-rule-validation-trades", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
