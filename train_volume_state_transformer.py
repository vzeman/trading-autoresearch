"""Train a volume-shape market-state transformer and test it as a risk gate.

The model is intentionally separate from the stock-picking strategies. It reads
recent market-wide volume-shape sequences, predicts whether the next few days
look like a risk/state-change window, and shifts that signal forward one trading
day before applying it to portfolio simulations.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluate_liquid_holding_multiyear import config_args
from evaluate_markov_holding_periods import (
    HoldingStrategy,
    add_cross_sectional_scores,
    default_markov_config,
    simulate_holding,
)
from evaluate_markov_regime_quant_strategy import load_daily_eval_frame


START_EQUITY = 50_000.0


MARKET_FEATURES = [
    "ew_return",
    "breadth_positive",
    "ret_dispersion",
    "volume_ratio_mean",
    "volume_ratio_std",
    "volume_ratio_p90",
    "dollar_volume_ratio_mean",
    "dollar_volume_ratio_std",
    "dollar_volume_ratio_p90",
    "volume_spike_mean",
    "volume_spike_p90",
    "up_dollar_volume_share",
    "down_dollar_volume_share",
    "price_volume_corr",
    "gap_mean",
    "range_mean",
    "spy_ret_1",
    "spy_ret_5",
    "spy_ret_20",
    "spy_vol_10",
]


class VolumeStateITransformer(nn.Module):
    """Small iTransformer-style encoder: variables are tokens, history is embedded."""

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.temporal_projection = nn.Linear(seq_len, d_model)
        self.feature_embedding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(d_model, 1)
        self.change_head = nn.Linear(d_model, 1)
        self.return_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch, time, features -> batch, features, time
        tokens = self.temporal_projection(x.transpose(1, 2)) + self.feature_embedding
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        hidden = self.head(pooled)
        return (
            self.risk_head(hidden).squeeze(-1),
            self.change_head(hidden).squeeze(-1),
            self.return_head(hidden).squeeze(-1),
        )


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def aggregate_market_volume_state(daily: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    src = daily.sort_values(["symbol", "date"]).copy()
    if "dollar_volume" not in src.columns:
        src["dollar_volume"] = src["close"].astype(float) * src["volume"].astype(float)
    by_symbol = src.groupby("symbol", sort=False)
    src["avg_vol_5"] = by_symbol["volume"].rolling(5).mean().reset_index(level=0, drop=True)
    src["avg_vol_20"] = by_symbol["volume"].rolling(20).mean().reset_index(level=0, drop=True)
    src["avg_dollar_vol_5"] = by_symbol["dollar_volume"].rolling(5).mean().reset_index(level=0, drop=True)
    src["avg_dollar_vol_20"] = by_symbol["dollar_volume"].rolling(20).mean().reset_index(level=0, drop=True)
    src["volume_ratio"] = src["volume"].astype(float) / src["avg_vol_20"].replace(0.0, np.nan)
    src["volume_ratio_5_20"] = src["avg_vol_5"] / src["avg_vol_20"].replace(0.0, np.nan)
    src["dollar_volume_ratio_5_20"] = src["avg_dollar_vol_5"] / src["avg_dollar_vol_20"].replace(0.0, np.nan)
    src["volume_spike"] = (src["volume_ratio"] - 2.5).clip(lower=0.0)
    src["signed_dollar_volume"] = np.sign(src["daily_return"].fillna(0.0).astype(float)) * src["dollar_volume"].astype(float)

    rows = []
    for date_value, group in src.groupby("date", sort=True):
        g = group.replace([np.inf, -np.inf], np.nan)
        total_dv = float(g["dollar_volume"].sum())
        up_dv = float(g.loc[g["daily_return"].astype(float) > 0.0, "dollar_volume"].sum())
        down_dv = float(g.loc[g["daily_return"].astype(float) < 0.0, "dollar_volume"].sum())
        corr_cols = g[["daily_return", "volume_ratio_5_20"]].dropna()
        corr = float(corr_cols.corr().iloc[0, 1]) if len(corr_cols) >= 3 else 0.0
        rows.append(
            {
                "date": date_value,
                "ew_return": float(g["daily_return"].astype(float).mean()),
                "breadth_positive": float(g["daily_return"].astype(float).gt(0.0).mean()),
                "ret_dispersion": float(g["daily_return"].astype(float).std(ddof=0)),
                "volume_ratio_mean": float(g["volume_ratio_5_20"].mean()),
                "volume_ratio_std": float(g["volume_ratio_5_20"].std(ddof=0)),
                "volume_ratio_p90": float(g["volume_ratio_5_20"].quantile(0.90)),
                "dollar_volume_ratio_mean": float(g["dollar_volume_ratio_5_20"].mean()),
                "dollar_volume_ratio_std": float(g["dollar_volume_ratio_5_20"].std(ddof=0)),
                "dollar_volume_ratio_p90": float(g["dollar_volume_ratio_5_20"].quantile(0.90)),
                "volume_spike_mean": float(g["volume_spike"].mean()),
                "volume_spike_p90": float(g["volume_spike"].quantile(0.90)),
                "up_dollar_volume_share": up_dv / total_dv if total_dv > 0 else 0.0,
                "down_dollar_volume_share": down_dv / total_dv if total_dv > 0 else 0.0,
                "price_volume_corr": corr if np.isfinite(corr) else 0.0,
                "gap_mean": float(g.get("gap_return", pd.Series(index=g.index, dtype=float)).astype(float).mean()),
                "range_mean": float(g.get("intraday_range", pd.Series(index=g.index, dtype=float)).astype(float).mean()),
            }
        )

    market = pd.DataFrame(rows)
    spy = frame[["date", "spy_daily_return"]].drop_duplicates("date").sort_values("date").copy()
    market = market.merge(spy, on="date", how="left")
    market["spy_ret_1"] = market["spy_daily_return"].fillna(0.0).astype(float)
    spy_ret = market["spy_ret_1"].astype(float)
    market["spy_ret_5"] = (1.0 + spy_ret).rolling(5).apply(np.prod, raw=True) - 1.0
    market["spy_ret_20"] = (1.0 + spy_ret).rolling(20).apply(np.prod, raw=True) - 1.0
    market["spy_vol_10"] = spy_ret.rolling(10).std()
    market["future_spy_ret_5"] = (1.0 + spy_ret.shift(-1)).rolling(5).apply(np.prod, raw=True).shift(-4) - 1.0

    future_path = pd.concat([(1.0 + spy_ret.shift(-i)).cumprod() for i in range(1, 6)], axis=1)
    market["future_spy_min_path_5"] = future_path.min(axis=1) - 1.0
    current_state = np.select(
        [
            (market["spy_ret_20"] < -0.03) | ((market["spy_ret_5"] < -0.015) & (market["spy_vol_10"] > 0.015)),
            (market["spy_ret_20"] > 0.03) & (market["spy_ret_5"] > 0.0),
        ],
        [0, 2],
        default=1,
    )
    future_state = pd.Series(current_state, index=market.index).shift(-5)
    market["state_change_label"] = (future_state != current_state).astype(float)
    market["risk_off_label"] = (
        (market["future_spy_ret_5"] < -0.02)
        | (market["future_spy_min_path_5"] < -0.025)
        | (future_state == 0)
    ).astype(float)
    market["date"] = pd.to_datetime(market["date"].astype(str), utc=True)
    market = market.replace([np.inf, -np.inf], np.nan).dropna(subset=MARKET_FEATURES + ["future_spy_ret_5"]).reset_index(drop=True)
    return market


def build_samples(market: pd.DataFrame, seq_len: int) -> dict[str, np.ndarray]:
    feature_values = market[MARKET_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    dates = pd.to_datetime(market["date"], utc=True).to_numpy()
    x, risk, change, future_ret, signal_dates, gate_dates = [], [], [], [], [], []
    for end_idx in range(seq_len - 1, len(market) - 1):
        if not np.isfinite(feature_values[end_idx - seq_len + 1 : end_idx + 1]).all():
            continue
        x.append(feature_values[end_idx - seq_len + 1 : end_idx + 1])
        risk.append(float(market["risk_off_label"].iloc[end_idx]))
        change.append(float(market["state_change_label"].iloc[end_idx]))
        future_ret.append(float(market["future_spy_ret_5"].iloc[end_idx]))
        signal_dates.append(dates[end_idx])
        gate_dates.append(dates[end_idx + 1])
    return {
        "x": np.asarray(x, dtype=np.float32),
        "risk": np.asarray(risk, dtype=np.float32),
        "change": np.asarray(change, dtype=np.float32),
        "future_ret": np.asarray(future_ret, dtype=np.float32),
        "signal_dates": np.asarray(signal_dates),
        "gate_dates": np.asarray(gate_dates),
    }


def standardize_samples(samples: dict[str, np.ndarray], train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = samples["x"].copy()
    train_x = x[train_mask]
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x = np.clip((x - mean) / std, -8.0, 8.0).astype(np.float32)
    return x, mean.astype(np.float32), std.astype(np.float32)


def train_one_fold(
    samples: dict[str, np.ndarray],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> tuple[pd.DataFrame, dict]:
    x, mean, std = standardize_samples(samples, train_mask)
    model = VolumeStateITransformer(
        seq_len=args.seq_len,
        n_features=x.shape[-1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    train_idx = np.flatnonzero(train_mask)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), args.batch_size):
            idx = train_idx[start : start + args.batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            risk = torch.from_numpy(samples["risk"][idx]).to(device)
            change = torch.from_numpy(samples["change"][idx]).to(device)
            future_ret = torch.from_numpy(samples["future_ret"][idx]).to(device)
            model.train()
            opt.zero_grad(set_to_none=True)
            risk_logit, change_logit, pred_ret = model(xb)
            loss = (
                bce(risk_logit, risk)
                + 0.45 * bce(change_logit, change)
                + 8.0 * mse(pred_ret, future_ret)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0}
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(args.epochs // 4, 1) == 0:
            print(f"[volume-state] epoch {epoch+1}/{args.epochs} loss={row['loss']:.4f}", flush=True)

    eval_idx = np.flatnonzero(eval_mask)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(eval_idx), args.batch_size):
            idx = eval_idx[start : start + args.batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            risk_logit, change_logit, pred_ret = model(xb)
            preds.append(
                pd.DataFrame(
                    {
                        "signal_date": pd.to_datetime(samples["signal_dates"][idx], utc=True),
                        "date": pd.to_datetime(samples["gate_dates"][idx], utc=True).date,
                        "risk_off_prob": torch.sigmoid(risk_logit).detach().cpu().numpy(),
                        "state_change_prob": torch.sigmoid(change_logit).detach().cpu().numpy(),
                        "pred_future_spy_ret_5": pred_ret.detach().cpu().numpy(),
                        "risk_off_label": samples["risk"][idx],
                        "state_change_label": samples["change"][idx],
                        "future_spy_ret_5": samples["future_ret"][idx],
                    }
                )
            )
    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    meta = {
        "history": history,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "train_samples": int(train_mask.sum()),
        "eval_samples": int(eval_mask.sum()),
    }
    return pred_df, meta


def attach_market_context(predictions: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    pred = predictions.copy()
    context_cols = [
        "date",
        "breadth_positive",
        "ret_dispersion",
        "volume_ratio_mean",
        "volume_ratio_p90",
        "dollar_volume_ratio_mean",
        "volume_spike_p90",
        "up_dollar_volume_share",
        "down_dollar_volume_share",
        "price_volume_corr",
        "spy_ret_5",
        "spy_ret_20",
        "spy_vol_10",
    ]
    context = market[context_cols].copy()
    context["signal_date"] = pd.to_datetime(context["date"], utc=True)
    context = context.drop(columns=["date"])
    return pred.merge(context, on="signal_date", how="left")


def attach_leader_context(predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    rows = []
    for date_value, group in frame.groupby("date", sort=True):
        candidates = group[group["symbol"].astype(str).str.upper() != "SPY"].copy()
        top = candidates.sort_values("trend_quality_score", ascending=False).head(3)
        rows.append(
            {
                "date": date_value,
                "leader_tq_mean": float(top["trend_quality_score"].mean()) if not top.empty else np.nan,
                "leader_rel_mean": float(top["relative_momentum_score"].mean()) if not top.empty else np.nan,
                "leader_ret20_mean": float(top["ret_20_prev"].mean()) if not top.empty else np.nan,
                "leader_volume_shape_mean": float(top["volume_shape_score"].mean()) if "volume_shape_score" in top and not top.empty else np.nan,
            }
        )
    leaders = pd.DataFrame(rows)
    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    return pred.merge(leaders, on="date", how="left")


def apply_volume_state_gate(frame: pd.DataFrame, predictions: pd.DataFrame, args: argparse.Namespace, mode: str = "hard") -> pd.DataFrame:
    out = frame.copy()
    pred = add_risk_gate_columns(predictions, args)
    pred["date"] = pd.to_datetime(pred["date"]).dt.date
    merge_cols = [
        "date",
        "risk_off_prob",
        "state_change_prob",
        "pred_future_spy_ret_5",
        "volume_state_risk_score",
        "volume_state_risk_off",
        "volume_state_liquidation_risk",
        "volume_state_rotation_risk",
        "volume_state_leader_risk",
    ]
    out = out.merge(
        pred[[col for col in merge_cols if col in pred.columns]],
        on="date",
        how="left",
    )
    out["volume_state_risk_off"] = out["volume_state_risk_off"].fillna(False).astype(bool)
    out["volume_state_liquidation_risk"] = out["volume_state_liquidation_risk"].fillna(False).astype(bool)
    out["volume_state_rotation_risk"] = out["volume_state_rotation_risk"].fillna(False).astype(bool)
    out["volume_state_leader_risk"] = out["volume_state_leader_risk"].fillna(False).astype(bool)
    out["volume_state_exposure_scale"] = 1.0
    out.loc[out["volume_state_risk_off"].astype(bool), "volume_state_exposure_scale"] = float(args.rotation_exposure)
    out.loc[out["volume_state_liquidation_risk"].astype(bool), "volume_state_exposure_scale"] = float(args.liquidation_exposure)
    if mode == "soft":
        return out
    score_cols = [
        "relative_momentum_score",
        "relative_momentum_volume_shape_score",
        "trend_quality_score",
        "trend_quality_volume_shape_score",
        "trend_quality_avoid_failed_gap_score",
        "trend_quality_avoid_failed_gap_volume_shape_score",
        "hybrid_markov_trend_score",
    ]
    if mode == "rotation":
        rotation_mask = out["volume_state_rotation_risk"].astype(bool)
        if args.rotation_relative_weight > 0.0 or args.rotation_volume_weight > 0.0:
            trend = out["trend_quality_score"].astype(float) if "trend_quality_score" in out else 0.0
            rel = out["relative_momentum_score"].astype(float) if "relative_momentum_score" in out else 0.0
            vol = out["volume_shape_score"].astype(float) if "volume_shape_score" in out else 0.0
            trend_weight = max(1.0 - args.rotation_relative_weight - args.rotation_volume_weight, 0.0)
            if "trend_quality_score" in out.columns:
                out.loc[rotation_mask, "trend_quality_score"] = (
                    trend_weight * trend.loc[rotation_mask].fillna(0.0)
                    + args.rotation_relative_weight * rel.loc[rotation_mask].fillna(0.0)
                    + args.rotation_volume_weight * vol.loc[rotation_mask].fillna(0.0)
                )
            if "trend_quality_volume_shape_score" in out.columns:
                out.loc[rotation_mask, "trend_quality_volume_shape_score"] = out.loc[rotation_mask, "trend_quality_score"]
        risk_mask = out["volume_state_liquidation_risk"].astype(bool)
    elif mode == "leader":
        risk_mask = out["volume_state_leader_risk"].astype(bool)
    else:
        risk_mask = out["volume_state_risk_off"].astype(bool)
    for col in score_cols:
        if col in out.columns:
            out.loc[risk_mask, col] = -999.0
    if "markov_signal" in out.columns:
        out.loc[risk_mask, "markov_signal"] = -999.0
    if "spy_markov_signal" in out.columns:
        out.loc[risk_mask, "spy_markov_signal"] = -999.0
    return out


def add_risk_gate_columns(predictions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    pred = predictions.copy()
    if pred.empty:
        pred["volume_state_risk_score"] = []
        pred["volume_state_risk_off"] = []
        pred["volume_state_liquidation_risk"] = []
        pred["volume_state_rotation_risk"] = []
        pred["volume_state_leader_risk"] = []
        return pred
    pred["volume_state_risk_score"] = (
        0.70 * pred["risk_off_prob"].astype(float)
        + 0.30 * pred["state_change_prob"].astype(float)
        - 0.20 * np.tanh(pred["pred_future_spy_ret_5"].astype(float) * 20.0)
    )
    pred["volume_state_risk_off"] = (
        (pred["volume_state_risk_score"] >= args.gate_threshold)
        | (pred["risk_off_prob"].astype(float) >= args.risk_threshold)
    )
    for col in [
        "breadth_positive",
        "down_dollar_volume_share",
        "ret_dispersion",
        "volume_spike_p90",
        "spy_ret_5",
        "spy_ret_20",
        "spy_vol_10",
    ]:
        if col not in pred.columns:
            pred[col] = np.nan
    breadth = pred["breadth_positive"].astype(float)
    down_share = pred["down_dollar_volume_share"].astype(float)
    ret_dispersion = pred["ret_dispersion"].astype(float)
    spike_p90 = pred["volume_spike_p90"].astype(float)
    spy_ret_5 = pred["spy_ret_5"].astype(float)
    spy_ret_20 = pred["spy_ret_20"].astype(float)
    spy_vol_10 = pred["spy_vol_10"].astype(float)
    observable_liquidation = (
        (
            spy_ret_5.le(args.liquidation_max_spy_ret_5)
            | spy_ret_20.le(args.liquidation_max_spy_ret_20)
        )
        & breadth.le(args.liquidation_max_breadth)
        & down_share.ge(args.liquidation_min_down_dollar_share)
    )
    liquidation_context = (
        breadth.le(args.liquidation_max_breadth)
        & down_share.ge(args.liquidation_min_down_dollar_share)
        & (
            ret_dispersion.ge(args.liquidation_min_ret_dispersion)
            | spike_p90.ge(args.liquidation_min_volume_spike_p90)
            | spy_vol_10.ge(args.liquidation_min_spy_vol_10)
        )
    )
    pred["volume_state_liquidation_risk"] = observable_liquidation | (
        pred["volume_state_risk_off"].astype(bool) & liquidation_context
    )
    pred["volume_state_rotation_risk"] = pred["volume_state_risk_off"].astype(bool) & ~pred["volume_state_liquidation_risk"].astype(bool)
    if "leader_rel_mean" not in pred.columns:
        pred["leader_rel_mean"] = np.nan
    pred["volume_state_leader_risk"] = (
        pred["volume_state_risk_off"].astype(bool)
        & pred["leader_rel_mean"].fillna(-999.0).astype(float).lt(args.leader_min_relative_strength)
    )
    return pred


def plot_results(results: pd.DataFrame, curves: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    curves = curves.copy()
    curves["timestamp"] = pd.to_datetime(curves["timestamp"], utc=True)
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"])

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.2, 1.4])
    ax_eq = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_dd = fig.add_subplot(gs[1, 1])
    ax_signal = fig.add_subplot(gs[2, :])

    for (year, strategy), group in curves.groupby(["year", "strategy"], sort=True):
        ax_eq.plot(group["timestamp"], group["equity"], linewidth=1.15, label=f"{year} {strategy}")
    spy = curves.groupby("timestamp", sort=True)["spy_equity"].first().reset_index()
    ax_eq.plot(spy["timestamp"], spy["spy_equity"], "--", color="black", linewidth=1.8, label="SPY")
    ax_eq.set_title("Volume-State Transformer Gate: Equity By Fold")
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.grid(alpha=0.25)
    ax_eq.legend(frameon=False, fontsize=7, ncol=3)

    pivot = results.pivot(index="year", columns="strategy", values="total_return")
    pivot.plot(kind="bar", ax=ax_bar)
    ax_bar.set_title("Fold Returns")
    ax_bar.set_ylabel("Return")
    ax_bar.axhline(0.0, color="#777", linewidth=0.8)
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.legend(frameon=False, fontsize=7)

    dd = results.pivot(index="year", columns="strategy", values="max_drawdown")
    dd.plot(kind="bar", ax=ax_dd, color=plt.cm.Reds_r(np.linspace(0.25, 0.75, max(len(dd.columns), 2))[: len(dd.columns)]))
    ax_dd.set_title("Fold Max Drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.axhline(0.0, color="#777", linewidth=0.8)
    ax_dd.grid(axis="y", alpha=0.25)
    ax_dd.legend(frameon=False, fontsize=7)

    if not predictions.empty:
        pred_line = predictions.groupby("date", as_index=False)[["risk_off_prob", "state_change_prob", "volume_state_risk_score"]].mean()
        ax_signal.plot(pred_line["date"], pred_line["risk_off_prob"], label="risk_off_prob", linewidth=1.4)
        ax_signal.plot(pred_line["date"], pred_line["state_change_prob"], label="state_change_prob", linewidth=1.4)
        ax_signal.plot(pred_line["date"], pred_line["volume_state_risk_score"], label="risk_score", linewidth=1.5)
        ax_signal.axhline(float(args.gate_threshold), color="#777", linestyle="--", linewidth=0.9, label="gate_threshold")
    ax_signal.set_title("Predicted Volume-State Risk")
    ax_signal.set_ylabel("Probability / score")
    ax_signal.grid(alpha=0.25)
    ax_signal.legend(frameon=False, fontsize=8)

    for path in [
        output_dir / "volume_state_transformer_gate.png",
        docs / "volume_state_transformer_gate.png",
    ]:
        fig.savefig(path, dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    print(f"[volume-state] device={device}", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = SimpleNamespace(**vars(args))
    base_args.output_dir = str(output_dir / "base_frame")
    base_args.eval_start = args.data_start
    base_args.eval_end = args.data_end
    config = default_markov_config(base_args)
    full_frame, daily, _signals = load_daily_eval_frame(config)
    full_frame = add_cross_sectional_scores(full_frame, daily)
    if args.min_median_dollar_volume > 0:
        full_frame = full_frame[
            full_frame["median_dollar_volume_20_prev"].fillna(0.0).astype(float).ge(args.min_median_dollar_volume)
        ].copy()
    market = aggregate_market_volume_state(daily, full_frame)
    samples = build_samples(market, args.seq_len)
    if len(samples["x"]) == 0:
        raise RuntimeError("No transformer samples were built")
    print(
        f"[volume-state] market_days={len(market):,} samples={len(samples['x']):,} "
        f"features={len(MARKET_FEATURES)}",
        flush=True,
    )

    base_strategy = HoldingStrategy(args.strategy, args.selector, max_hold_days=args.max_hold_days)
    gated_strategy = HoldingStrategy(
        f"{args.strategy}_volume_state_gate",
        args.selector,
        max_hold_days=args.max_hold_days,
        min_hold_days=1,
        exit_on_spy_gate=True,
    )
    rotation_strategy = HoldingStrategy(
        f"{args.strategy}_volume_state_rotation_gate",
        args.selector,
        max_hold_days=args.max_hold_days,
        min_hold_days=1,
        exit_on_spy_gate=True,
    )
    soft_strategy = HoldingStrategy(
        f"{args.strategy}_volume_state_soft_rotation",
        args.selector,
        max_hold_days=args.max_hold_days,
        min_hold_days=1,
        exit_on_spy_gate=False,
        exposure_column="volume_state_exposure_scale",
    )
    leader_strategy = HoldingStrategy(
        f"{args.strategy}_volume_state_leader_gate",
        args.selector,
        max_hold_days=args.max_hold_days,
        min_hold_days=1,
        exit_on_spy_gate=True,
    )

    all_results = []
    all_curves = []
    all_trades = []
    all_predictions = []
    fold_meta = {}
    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    for year in years:
        eval_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        eval_end = pd.Timestamp(args.final_eval_end if year == args.final_year else f"{year}-12-31", tz="UTC")
        gate_dates = pd.to_datetime(samples["gate_dates"], utc=True)
        train_mask = gate_dates < eval_start
        eval_mask = (gate_dates >= eval_start) & (gate_dates <= eval_end)
        if int(train_mask.sum()) < args.min_train_samples or int(eval_mask.sum()) == 0:
            print(
                f"[volume-state] skip {year}: train={int(train_mask.sum())} eval={int(eval_mask.sum())}",
                flush=True,
            )
            continue
        print(f"[volume-state] fold={year} train={int(train_mask.sum())} eval={int(eval_mask.sum())}", flush=True)
        pred_df, meta = train_one_fold(samples, train_mask, eval_mask, args, device)
        pred_df["year"] = int(year)
        pred_df = attach_market_context(pred_df, market)
        pred_df = attach_leader_context(pred_df, full_frame)
        pred_with_actions = add_risk_gate_columns(pred_df, args)
        all_predictions.append(pred_with_actions)
        fold_meta[str(year)] = meta

        cargs = config_args(args, year, output_dir)
        fold_config = default_markov_config(cargs)
        fold_frame = full_frame[
            (pd.to_datetime(full_frame["date"].astype(str), utc=True) >= eval_start)
            & (pd.to_datetime(full_frame["date"].astype(str), utc=True) <= eval_end)
        ].copy()
        if fold_frame.empty:
            continue
        gated_frame = apply_volume_state_gate(fold_frame, pred_df, args, mode="hard")
        rotation_frame = apply_volume_state_gate(fold_frame, pred_df, args, mode="rotation")
        soft_frame = apply_volume_state_gate(fold_frame, pred_df, args, mode="soft")
        leader_frame = apply_volume_state_gate(fold_frame, pred_df, args, mode="leader")
        simulation_frames = [
            (base_strategy, fold_frame, 0, 0, 0),
            (
                gated_strategy,
                gated_frame,
                int(gated_frame[["date", "volume_state_risk_off"]].drop_duplicates()["volume_state_risk_off"].sum()),
                int(gated_frame[["date", "volume_state_liquidation_risk"]].drop_duplicates()["volume_state_liquidation_risk"].sum()),
                int(gated_frame[["date", "volume_state_rotation_risk"]].drop_duplicates()["volume_state_rotation_risk"].sum()),
            ),
            (
                rotation_strategy,
                rotation_frame,
                int(rotation_frame[["date", "volume_state_risk_off"]].drop_duplicates()["volume_state_risk_off"].sum()),
                int(rotation_frame[["date", "volume_state_liquidation_risk"]].drop_duplicates()["volume_state_liquidation_risk"].sum()),
                int(rotation_frame[["date", "volume_state_rotation_risk"]].drop_duplicates()["volume_state_rotation_risk"].sum()),
            ),
            (
                soft_strategy,
                soft_frame,
                int(soft_frame[["date", "volume_state_risk_off"]].drop_duplicates()["volume_state_risk_off"].sum()),
                int(soft_frame[["date", "volume_state_liquidation_risk"]].drop_duplicates()["volume_state_liquidation_risk"].sum()),
                int(soft_frame[["date", "volume_state_rotation_risk"]].drop_duplicates()["volume_state_rotation_risk"].sum()),
            ),
            (
                leader_strategy,
                leader_frame,
                int(leader_frame[["date", "volume_state_risk_off"]].drop_duplicates()["volume_state_risk_off"].sum()),
                int(leader_frame[["date", "volume_state_leader_risk"]].drop_duplicates()["volume_state_leader_risk"].sum()),
                int(leader_frame[["date", "volume_state_rotation_risk"]].drop_duplicates()["volume_state_rotation_risk"].sum()),
            ),
        ]
        for strategy, frame, risk_days, liquidation_days, rotation_days in simulation_frames:
            summary, curve, trades, _ = simulate_holding(frame, fold_config, strategy)
            summary.update(
                {
                    "year": int(year),
                    "train_samples": int(train_mask.sum()),
                    "eval_samples": int(eval_mask.sum()),
                    "risk_gate_days": risk_days,
                    "liquidation_gate_days": liquidation_days,
                    "rotation_risk_days": rotation_days,
                    "leader_gate_days": int(frame[["date", "volume_state_leader_risk"]].drop_duplicates()["volume_state_leader_risk"].sum())
                    if "volume_state_leader_risk" in frame.columns
                    else 0,
                }
            )
            all_results.append(summary)
            curve = curve.copy()
            curve["year"] = int(year)
            all_curves.append(curve)
            if not trades.empty:
                trades = trades.copy()
                trades["year"] = int(year)
                all_trades.append(trades)
            print(
                f"  {strategy.name}: return={summary['total_return']:.2%} "
                f"spy={summary['spy_total_return']:.2%} alpha={summary['alpha_return']:.2%} "
                f"dd={summary['max_drawdown']:.2%}",
                flush=True,
            )

    results = pd.DataFrame(all_results)
    curves = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    results.to_csv(output_dir / "volume_state_transformer_results.csv", index=False)
    curves.to_csv(output_dir / "volume_state_transformer_curves.csv", index=False)
    trades.to_csv(output_dir / "volume_state_transformer_trades.csv", index=False)
    predictions.to_csv(output_dir / "volume_state_transformer_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "market_features": MARKET_FEATURES,
                "fold_meta": fold_meta,
                "warning": "research_only_volume_state_transformer_not_financial_advice",
            },
            indent=2,
            default=str,
        )
    )
    if not results.empty and not curves.empty:
        plot_results(results, curves, predictions, output_dir, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/transformer_15m/shared_15m_top10_volume_valuation_algo.parquet")
    parser.add_argument("--output-dir", default="checkpoints/transformer_15m/volume_state_transformer_gate")
    parser.add_argument("--years", default="2022,2023,2024,2025,2026")
    parser.add_argument("--final-year", type=int, default=2026)
    parser.add_argument("--final-eval-end", default="2026-05-14")
    parser.add_argument("--data-start", default="2020-11-01")
    parser.add_argument("--data-end", default="2026-05-14")
    parser.add_argument("--strategy", default="trend_quality_hold_3d")
    parser.add_argument("--selector", default="trend_quality")
    parser.add_argument("--max-hold-days", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gate-threshold", type=float, default=0.55)
    parser.add_argument("--risk-threshold", type=float, default=0.60)
    parser.add_argument("--liquidation-max-breadth", type=float, default=0.48)
    parser.add_argument("--liquidation-min-down-dollar-share", type=float, default=0.52)
    parser.add_argument("--liquidation-max-spy-ret-5", type=float, default=-0.015)
    parser.add_argument("--liquidation-max-spy-ret-20", type=float, default=-0.030)
    parser.add_argument("--liquidation-min-ret-dispersion", type=float, default=0.012)
    parser.add_argument("--liquidation-min-volume-spike-p90", type=float, default=0.020)
    parser.add_argument("--liquidation-min-spy-vol-10", type=float, default=0.010)
    parser.add_argument("--rotation-relative-weight", type=float, default=0.0)
    parser.add_argument("--rotation-volume-weight", type=float, default=0.0)
    parser.add_argument("--liquidation-exposure", type=float, default=0.35)
    parser.add_argument("--rotation-exposure", type=float, default=0.70)
    parser.add_argument("--leader-min-relative-strength", type=float, default=0.24)
    parser.add_argument("--min-train-samples", type=int, default=180)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--symbols", default="")
    parser.add_argument("--top-symbols-limit", type=int, default=0)
    parser.add_argument("--universe-rank-cache", default="checkpoints/transformer_15m/top_volume_valuation_universe.csv")
    parser.add_argument("--regime-window-days", type=int, default=20)
    parser.add_argument("--bull-threshold", type=float, default=0.05)
    parser.add_argument("--bear-threshold", type=float, default=-0.05)
    parser.add_argument("--min-transition-days", type=int, default=80)
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--forecast-horizon-days", type=int, default=1)
    parser.add_argument("--min-signal", type=float, default=0.05)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--portfolio-exposure", type=float, default=1.0)
    parser.add_argument("--signal-full-exposure", type=float, default=0.35)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0008)
    parser.add_argument("--min-close", type=float, default=5.0)
    parser.add_argument("--max-abs-return", type=float, default=0.08)
    parser.add_argument("--adaptive-min-history-days", type=int, default=80)
    parser.add_argument("--regime-source", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--transition-lookback-days", type=int, default=126)
    parser.add_argument("--transition-halflife-days", type=float, default=42.0)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
