"""True walk-forward retraining for the second-stage allocator.

For each forward test fold, this script:

1. trains a fresh allocator using only calibration data plus earlier folds,
2. chooses a trade/cash threshold on the same past data,
3. applies the allocator and threshold to the next fold.

This is stricter than training one allocator once and only moving the threshold.
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
from torch.utils.data import DataLoader

from evaluate_world_model import load_frame, predict
from train_allocator import (
    AllocatorModel,
    add_targets,
    apply_allocator,
    dataset,
    evaluate_model,
    filter_frame,
    make_matrices,
)
from train_world_model import pick_device
from walk_forward_allocator import choose_threshold, planner_rows, summarize_with_cash


@dataclass(frozen=True)
class RetrainConfig:
    calibration_data: str
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
    folds: int
    min_coverage: float
    objective_mode: str
    seed: int
    device: str


def split_candidates_by_time(scored: pd.DataFrame, folds: int) -> list[pd.DataFrame]:
    time_col = "decision_timestamp" if "decision_timestamp" in scored.columns else "timestamp"
    unique_times = pd.Series(pd.to_datetime(scored[time_col], utc=True).sort_values().unique())
    chunks = np.array_split(unique_times.to_numpy(), folds)
    ts = pd.to_datetime(scored[time_col], utc=True)
    out = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        mask = ts.isin(pd.to_datetime(chunk, utc=True))
        fold = scored[mask.to_numpy()].copy().sort_values([time_col, "symbol", "horizon_bars"]).reset_index(drop=True)
        if len(fold):
            out.append(fold)
    return out


def split_masks(df: pd.DataFrame, val_fraction: float, val_gap_days: float) -> tuple[np.ndarray, np.ndarray]:
    time_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    timestamps = pd.to_datetime(df[time_col], utc=True)
    cutoff = timestamps.quantile(max(0.0, min(1.0, 1.0 - val_fraction)))
    train_cutoff = cutoff - pd.Timedelta(days=max(0.0, val_gap_days))
    train_mask = (timestamps <= train_cutoff).to_numpy()
    val_mask = (timestamps > cutoff).to_numpy()
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        rng = np.random.default_rng(0)
        val_mask = rng.random(len(df)) < val_fraction
        train_mask = ~val_mask
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError("empty walk-forward allocator train/validation split")
    return train_mask, val_mask


def train_fold_allocator(scored_train: pd.DataFrame, config: RetrainConfig, device: str, fold_idx: int) -> tuple[dict, dict]:
    torch.manual_seed(config.seed + fold_idx)
    np.random.seed(config.seed + fold_idx)

    train_scored = add_targets(scored_train, config.top_quantile)
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
        f"[wf-retrain] fold={fold_idx} rows={len(train_scored):,} "
        f"train={int(train_mask.sum()):,} val={int(val_mask.sum()):,} device={device}",
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
            f"[wf-retrain] fold={fold_idx} epoch={epoch+1}/{config.epochs} "
            f"train={row['train_loss']:.4f} val={row['loss']:.4f} top_acc={row['top_accuracy']:.3f}",
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
        "config": {
            "hidden_dim": config.hidden_dim,
            "n_layers": config.n_layers,
        },
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
    metrics = {
        "fold": fold_idx,
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_loss),
        "last": history[-1] if history else {},
    }
    return ckpt, metrics


def score_raw_dataset(
    path: Path,
    world_ckpt: dict,
    config: RetrainConfig,
    device: str,
) -> pd.DataFrame:
    df = filter_frame(
        load_frame(path, config.limit_rows, seed=config.seed),
        config.min_horizon_bars,
        config.max_horizon_bars,
    )
    return predict(df, world_ckpt, device=device, batch_size=config.batch_size)


def run(config: RetrainConfig) -> dict:
    device = pick_device(config.device)
    world_ckpt = torch.load(config.world_checkpoint, map_location="cpu", weights_only=False)
    calibration_base = score_raw_dataset(Path(config.calibration_data), world_ckpt, config, device)
    test_base = score_raw_dataset(Path(config.test_data), world_ckpt, config, device)
    test_folds = split_candidates_by_time(test_base, config.folds)

    fold_results = []
    active_frames = []
    observed_past = calibration_base.copy()
    for fold_idx, fold_base in enumerate(test_folds, start=1):
        allocator_ckpt, train_metrics = train_fold_allocator(observed_past, config, device, fold_idx)

        past_allocated = apply_allocator(observed_past, allocator_ckpt, device=device, batch_size=config.batch_size)
        past_planner = planner_rows(past_allocated)
        threshold_choice = choose_threshold(past_planner, config.min_coverage, config.objective_mode)
        selected = threshold_choice["best"]

        fold_allocated = apply_allocator(fold_base, allocator_ckpt, device=device, batch_size=config.batch_size)
        fold_planner = planner_rows(fold_allocated)
        active = fold_planner[fold_planner["pred_score"] >= selected["threshold"]].copy()
        applied = summarize_with_cash(
            f"retrained_walk_forward_fold_{fold_idx}",
            active,
            len(fold_planner),
            selected["threshold"],
            selected["quantile"],
        )
        fold_results.append({
            "fold": fold_idx,
            "candidate_rows": int(len(fold_base)),
            "groups": int(len(fold_planner)),
            "train_metrics": train_metrics,
            "selected_quantile": selected["quantile"],
            "selected_threshold": selected["threshold"],
            "calibration_objective": selected["objective"],
            "applied": applied,
        })
        active_frames.append(active)
        observed_past = pd.concat([observed_past, fold_base], ignore_index=True)

    active_all = pd.concat(active_frames, ignore_index=True) if active_frames else test_base.iloc[:0].copy()
    total_groups = int(sum(row["groups"] for row in fold_results))
    aggregate = summarize_with_cash("retrained_walk_forward_aggregate", active_all, total_groups, float("nan"), float("nan"))
    payload = {
        "config": asdict(config),
        "device": device,
        "calibration_candidate_rows": int(len(calibration_base)),
        "test_candidate_rows": int(len(test_base)),
        "test_groups": total_groups,
        "folds": fold_results,
        "aggregate": aggregate,
    }
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
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
    parser.add_argument("--max-horizon-bars", type=int, default=120)
    parser.add_argument("--top-quantile", type=float, default=0.80)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--objective-mode", choices=["cash_return", "active_return", "hybrid"], default="hybrid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(RetrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
