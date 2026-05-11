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
from torch.utils.data import DataLoader, TensorDataset

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
    regime_mode: str
    sizing_mode: str
    gate_mode: str
    feature_mode: str
    seed: int
    device: str


GATE_FEATURES = [
    "pred_score",
    "pred_portfolio_return",
    "pred_max_drawdown",
    "pred_path_vol",
    "pred_future_alpha_vs_spy",
    "pred_profit_label",
    "pred_beat_spy_label",
    "pred_rank_top_quartile",
    "feat_spy_logret_60",
    "feat_spy_logret_390",
    "feat_spy_logret_2730",
    "state_ret_30m",
    "state_ret_2h",
    "state_ret_1d",
    "state_ret_5d",
    "state_vol_30m",
    "state_vol_2h",
    "state_vol_1d",
    "state_vol_5d",
    "state_drawdown_5d",
    "horizon_bars",
    "current_position_frac",
    "target_position_frac",
]


class TradeGate(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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
    mats = make_matrices(train_scored, train_mask, feature_mode=config.feature_mode)
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


def regime_candidates(past_active: pd.DataFrame) -> list[dict]:
    candidates = [{"name": "none", "conditions": []}]
    specs = [
        ("feat_spy_logret_60", ">=", (0.20, 0.30, 0.40)),
        ("feat_spy_logret_390", ">=", (0.20, 0.30, 0.40)),
        ("feat_spy_logret_2730", ">=", (0.20, 0.30, 0.40)),
        ("state_ret_1d", ">=", (0.20, 0.30, 0.40)),
        ("state_ret_5d", ">=", (0.20, 0.30, 0.40)),
        ("state_vol_1d", "<=", (0.60, 0.70, 0.80)),
        ("state_vol_5d", "<=", (0.60, 0.70, 0.80)),
        ("state_drawdown_5d", ">=", (0.20, 0.30, 0.40)),
    ]
    for col, op, qs in specs:
        if col not in past_active.columns:
            continue
        values = past_active[col].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        for q in qs:
            threshold = float(values.quantile(q))
            candidates.append({
                "name": f"{col}_{op}_q{q:.2f}",
                "conditions": [{"column": col, "op": op, "threshold": threshold}],
            })

    combos = [
        ("feat_spy_logret_390", ">=", 0.30, "state_vol_1d", "<=", 0.70),
        ("feat_spy_logret_60", ">=", 0.30, "state_vol_1d", "<=", 0.70),
        ("state_ret_1d", ">=", 0.30, "state_vol_1d", "<=", 0.70),
        ("state_ret_5d", ">=", 0.30, "state_drawdown_5d", ">=", 0.30),
    ]
    for col_a, op_a, q_a, col_b, op_b, q_b in combos:
        if col_a not in past_active.columns or col_b not in past_active.columns:
            continue
        vals_a = past_active[col_a].replace([np.inf, -np.inf], np.nan).dropna()
        vals_b = past_active[col_b].replace([np.inf, -np.inf], np.nan).dropna()
        if vals_a.empty or vals_b.empty:
            continue
        candidates.append({
            "name": f"{col_a}_{op_a}_q{q_a:.2f}__{col_b}_{op_b}_q{q_b:.2f}",
            "conditions": [
                {"column": col_a, "op": op_a, "threshold": float(vals_a.quantile(q_a))},
                {"column": col_b, "op": op_b, "threshold": float(vals_b.quantile(q_b))},
            ],
        })
    return candidates


def apply_regime_rule(df: pd.DataFrame, rule: dict) -> pd.DataFrame:
    if not rule.get("conditions"):
        return df.copy()
    mask = pd.Series(True, index=df.index)
    for condition in rule["conditions"]:
        col = condition["column"]
        if col not in df.columns:
            mask &= False
            continue
        values = df[col].replace([np.inf, -np.inf], np.nan)
        if condition["op"] == ">=":
            mask &= values >= condition["threshold"]
        elif condition["op"] == "<=":
            mask &= values <= condition["threshold"]
        else:
            raise ValueError(f"unknown regime op: {condition['op']}")
    return df[mask].copy()


def choose_regime_rule(
    past_active: pd.DataFrame,
    total_groups: int,
    min_coverage: float,
    objective_mode: str,
    enabled: bool,
) -> dict:
    from walk_forward_allocator import objective

    if not enabled or past_active.empty:
        return {
            "selected": {"name": "none", "conditions": [], "objective": 0.0},
            "candidates": [],
        }
    best: dict | None = None
    rows = []
    for rule in regime_candidates(past_active):
        filtered = apply_regime_rule(past_active, rule)
        summary = summarize_with_cash(f"regime_{rule['name']}", filtered, total_groups, float("nan"), float("nan"))
        score = float(objective(summary, min_coverage, objective_mode))
        candidate = {**rule, "summary": summary, "objective": score}
        rows.append(candidate)
        if best is None or score > best["objective"]:
            best = candidate
    if best is None:
        best = {"name": "none", "conditions": [], "objective": 0.0}
    return {"selected": best, "candidates": rows}


def fit_sizing_rule(past_active: pd.DataFrame, enabled: bool) -> dict:
    if not enabled or past_active.empty:
        return {"name": "none", "score_q50": None, "score_q80": None}
    return {
        "name": "score_quantile",
        "score_q50": float(past_active["pred_score"].quantile(0.50)),
        "score_q80": float(past_active["pred_score"].quantile(0.80)),
    }


def apply_sizing_rule(active: pd.DataFrame, rule: dict) -> pd.DataFrame:
    if active.empty or rule.get("name") != "score_quantile":
        out = active.copy()
        out["position_size_multiplier"] = 1.0
        return out
    q50 = float(rule["score_q50"])
    q80 = float(rule["score_q80"])
    score = active["pred_score"].astype(float)
    size = np.where(score >= q80, 1.0, np.where(score >= q50, 0.75, 0.50))
    out = active.copy()
    out["position_size_multiplier"] = size.astype(float)
    for col in ("portfolio_return", "portfolio_pnl", "future_alpha_vs_spy", "max_drawdown", "path_vol"):
        if col in out.columns:
            out[col] = out[col].astype(float) * out["position_size_multiplier"]
    if "portfolio_return" in out.columns:
        out["profit_label"] = (out["portfolio_return"] > 0.0).astype(float)
    if "future_alpha_vs_spy" in out.columns:
        out["beat_spy_label"] = (out["future_alpha_vs_spy"] > 0.0).astype(float)
    return out


def make_gate_matrix(df: pd.DataFrame, feature_cols: list[str], mean: np.ndarray | None = None, std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    if "horizon_bars" in feature_cols:
        x[:, feature_cols.index("horizon_bars")] = np.log1p(x[:, feature_cols.index("horizon_bars")])
    if mean is None:
        mean = x.mean(axis=0)
    if std is None:
        std = x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x = np.clip((x - mean) / std, -20.0, 20.0).astype(np.float32)
    return x, mean, std


def train_trade_gate(past_planner: pd.DataFrame, config: RetrainConfig, device: str, fold_idx: int) -> tuple[dict, dict]:
    feature_cols = [c for c in GATE_FEATURES if c in past_planner.columns]
    if not feature_cols:
        return {"mode": "none"}, {"enabled": False}
    time_col = "decision_timestamp" if "decision_timestamp" in past_planner.columns else "timestamp"
    timestamps = pd.to_datetime(past_planner[time_col], utc=True)
    cutoff = timestamps.quantile(0.80)
    train_mask = (timestamps <= cutoff).to_numpy()
    val_mask = (timestamps > cutoff).to_numpy()
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        train_mask = np.ones(len(past_planner), dtype=bool)
        val_mask = np.ones(len(past_planner), dtype=bool)

    y = (
        (past_planner["portfolio_return"].astype(float) > 0.0)
        & (past_planner["future_alpha_vs_spy"].astype(float) > 0.0)
    ).astype(np.float32).to_numpy()
    x, mean, std = make_gate_matrix(past_planner, feature_cols)
    train_ds = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    val_ds = TensorDataset(torch.from_numpy(x[val_mask]), torch.from_numpy(y[val_mask]))
    train_loader = DataLoader(train_ds, batch_size=min(4096, max(128, len(train_ds))), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(8192, max(128, len(val_ds))), shuffle=False)
    model = TradeGate(len(feature_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    pos = float(y[train_mask].sum())
    neg = float(train_mask.sum() - pos)
    pos_weight = torch.tensor([min(10.0, max(1.0, neg / max(pos, 1.0)))], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(8):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_losses.append(float(bce(logits, yb).item()))
                correct += int(((torch.sigmoid(logits) >= 0.5) == (yb >= 0.5)).sum().item())
                total += int(yb.numel())
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_loss": val_loss,
            "val_accuracy": float(correct / max(total, 1)),
        }
        history.append(row)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = {
        "mode": "learned",
        "state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "x_mean": mean,
        "x_std": std,
    }
    metrics = {
        "enabled": True,
        "fold": fold_idx,
        "rows": int(len(past_planner)),
        "positive_rate": float(y.mean()),
        "best_val_loss": float(best_loss),
        "last": history[-1] if history else {},
    }
    return ckpt, metrics


def apply_trade_gate(df: pd.DataFrame, gate_ckpt: dict, device: str) -> pd.DataFrame:
    if gate_ckpt.get("mode") != "learned" or df.empty:
        out = df.copy()
        out["trade_gate_score"] = 1.0
        return out
    x, _, _ = make_gate_matrix(df, gate_ckpt["feature_cols"], gate_ckpt["x_mean"], gate_ckpt["x_std"])
    model = TradeGate(len(gate_ckpt["feature_cols"]), dropout=0.0).to(device)
    model.load_state_dict(gate_ckpt["state_dict"])
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(df), 8192):
            xb = torch.from_numpy(x[i : i + 8192]).to(device)
            scores.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    out = df.copy()
    out["trade_gate_score"] = np.concatenate(scores, axis=0)
    return out


def choose_gate_threshold(past_active: pd.DataFrame, total_groups: int, min_coverage: float, objective_mode: str, enabled: bool) -> dict:
    from walk_forward_allocator import objective

    if not enabled or past_active.empty or "trade_gate_score" not in past_active.columns:
        return {"threshold": 0.0, "quantile": 0.0, "objective": 0.0, "selected": "none"}
    best: dict | None = None
    for q in (0.0, 0.20, 0.40, 0.60, 0.80):
        threshold = float(past_active["trade_gate_score"].quantile(q))
        gated = past_active[past_active["trade_gate_score"] >= threshold].copy()
        summary = summarize_with_cash(f"gate_q{q:.2f}", gated, total_groups, threshold, q)
        score = float(objective(summary, min_coverage, objective_mode))
        row = {"threshold": threshold, "quantile": q, "objective": score, "summary": summary, "selected": "learned"}
        if best is None or score > best["objective"]:
            best = row
    return best or {"threshold": 0.0, "quantile": 0.0, "objective": 0.0, "selected": "none"}


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
        past_active = past_planner[past_planner["pred_score"] >= selected["threshold"]].copy()
        regime_choice = choose_regime_rule(
            past_active,
            len(past_planner),
            config.min_coverage,
            config.objective_mode,
            enabled=config.regime_mode == "select",
        )
        sizing_rule = fit_sizing_rule(past_active, enabled=config.sizing_mode == "score_quantile")
        gate_ckpt, gate_metrics = train_trade_gate(past_planner, config, device, fold_idx) if config.gate_mode == "learned" else ({"mode": "none"}, {"enabled": False})
        past_active_gated = apply_trade_gate(past_active, gate_ckpt, device)
        gate_rule = choose_gate_threshold(
            past_active_gated,
            len(past_planner),
            config.min_coverage,
            config.objective_mode,
            enabled=config.gate_mode == "learned",
        )

        fold_allocated = apply_allocator(fold_base, allocator_ckpt, device=device, batch_size=config.batch_size)
        fold_planner = planner_rows(fold_allocated)
        active = fold_planner[fold_planner["pred_score"] >= selected["threshold"]].copy()
        active = apply_regime_rule(active, regime_choice["selected"])
        active = apply_trade_gate(active, gate_ckpt, device)
        if config.gate_mode == "learned":
            active = active[active["trade_gate_score"] >= gate_rule["threshold"]].copy()
        active = apply_sizing_rule(active, sizing_rule)
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
            "regime_rule": regime_choice["selected"],
            "sizing_rule": sizing_rule,
            "gate_metrics": gate_metrics,
            "gate_rule": gate_rule,
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
    parser.add_argument("--regime-mode", choices=["none", "select"], default="none")
    parser.add_argument("--sizing-mode", choices=["none", "score_quantile"], default="none")
    parser.add_argument("--gate-mode", choices=["none", "learned"], default="none")
    parser.add_argument("--feature-mode", choices=["compact", "market"], default="compact")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(RetrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
