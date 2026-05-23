"""Train/evaluate a daily cross-sectional stock ranker.

This is intentionally simpler than the action-conditioned world model. It
tests whether basic cross-sectional daily features contain enough tradable edge
before adding more deep architecture.
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
from torch.utils.data import DataLoader, TensorDataset

from prepare import CACHE_DIR
from top500_universe import load_top500_symbols


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_20d", "ret_60d",
    "vol_20d", "vol_60d", "volume_z_20d",
    "ma20_dist", "ma60_dist", "drawdown_60d",
    "spy_ret_1d", "spy_ret_5d", "spy_ret_20d", "spy_ret_60d",
    "rel_spy_1d", "rel_spy_5d", "rel_spy_20d", "rel_spy_60d",
    "rel_spy_speed_1d", "rel_spy_speed_5d", "rel_spy_speed_20d", "rel_spy_speed_60d",
    "spy_same_dir_1d", "spy_same_dir_5d", "spy_same_dir_20d", "spy_same_dir_60d",
    "spy_opposite_dir_1d", "spy_opposite_dir_5d", "spy_opposite_dir_20d", "spy_opposite_dir_60d",
    "spy_lagging_same_dir_5d", "spy_lagging_same_dir_20d", "spy_lagging_same_dir_60d",
    "spy_leading_same_dir_5d", "spy_leading_same_dir_20d", "spy_leading_same_dir_60d",
    "mkt_ret_1d_mean", "mkt_ret_5d_mean", "mkt_ret_20d_mean", "mkt_ret_60d_mean",
    "mkt_ret_20d_median", "mkt_rel_spy_20d_mean", "mkt_vol_20d_mean",
    "mkt_ret_20d_dispersion", "mkt_pct_positive_20d", "mkt_pct_above_ma20",
    "mkt_pct_drawdown_gt_10", "mkt_pct_low_vol",
    "rel_mkt_20d", "rel_mkt_60d",
    "xsec_ret_5d_rank", "xsec_ret_20d_rank", "xsec_vol_20d_rank",
    "xsec_drawdown_60d_rank", "xsec_volume_z_20d_rank",
]


@dataclass(frozen=True)
class Config:
    output_dir: str
    start_date: str
    end_date: str
    train_end: str
    test_start: str
    test_end: str
    horizon_days: int
    top500: bool
    cached_all: bool
    symbol_limit: int
    min_rows: int
    epochs: int
    batch_size: int
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    validation_fraction: float
    min_validation_trades: int
    min_validation_return: float
    min_validation_active_alpha: float
    min_validation_profit_rate: float
    min_validation_beat_spy_rate: float
    max_validation_drawdown: float
    rule_validation_fraction: float
    min_rule_validation_trades: int
    top_k: int
    max_positions: int
    observed_score_weight: float
    utility_mode: str
    device: str
    seed: int


class DailyRanker(nn.Module):
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
        self.utility = nn.Linear(hidden_dim, 1)
        self.profit = nn.Linear(hidden_dim, 1)
        self.crash = nn.Linear(hidden_dim, 1)
        self.top = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.utility(h).squeeze(-1), self.profit(h).squeeze(-1), self.crash(h).squeeze(-1), self.top(h).squeeze(-1)


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_1m.parquet"


def load_symbols(config: Config) -> list[str]:
    if config.cached_all:
        symbols = sorted(p.name.removesuffix("_1m.parquet") for p in CACHE_DIR.glob("*_1m.parquet"))
    elif config.top500:
        symbols = load_top500_symbols()
    else:
        symbols = sorted(p.name.removesuffix("_1m.parquet") for p in CACHE_DIR.glob("*_1m.parquet"))
    out = []
    seen = set()
    for sym in symbols:
        if sym in seen or not _cache_path(sym).exists() or sym.startswith("^"):
            continue
        seen.add(sym)
        out.append(sym)
        if config.symbol_limit > 0 and len(out) >= config.symbol_limit:
            break
    return out


def daily_bars(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(_cache_path(symbol), columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    for col in ("open", "high", "low", "close"):
        df = df[df[col].astype(float) > 0]
    df["date"] = df["timestamp"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("date", sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    daily["symbol"] = symbol
    return daily


def add_market_context(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("date", group_keys=False)
    market = grouped.agg(
        mkt_ret_1d_mean=("ret_1d", "mean"),
        mkt_ret_5d_mean=("ret_5d", "mean"),
        mkt_ret_20d_mean=("ret_20d", "mean"),
        mkt_ret_60d_mean=("ret_60d", "mean"),
        mkt_ret_20d_median=("ret_20d", "median"),
        mkt_rel_spy_20d_mean=("rel_spy_20d", "mean"),
        mkt_vol_20d_mean=("vol_20d", "mean"),
        mkt_ret_20d_dispersion=("ret_20d", "std"),
        mkt_pct_positive_20d=("ret_20d", lambda s: float((s > 0).mean())),
        mkt_pct_above_ma20=("ma20_dist", lambda s: float((s > 0).mean())),
        mkt_pct_drawdown_gt_10=("drawdown_60d", lambda s: float((s > -0.10).mean())),
        mkt_pct_low_vol=("xsec_vol_20d_rank", lambda s: float((s < 0.50).mean())),
    ).reset_index()
    out = df.merge(market, on="date", how="left")
    out["rel_mkt_20d"] = out["ret_20d"] - out["mkt_ret_20d_mean"]
    out["rel_mkt_60d"] = out["ret_60d"] - out["mkt_ret_60d_mean"]
    return out


def add_features(df: pd.DataFrame, spy: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    for n in (1, 5, 20, 60):
        df[f"ret_{n}d"] = np.log(close / close.shift(n))
    for n in (20, 60):
        df[f"vol_{n}d"] = df["ret_1d"].rolling(n, min_periods=max(5, n // 4)).std()
    vol_mean = np.log1p(volume).rolling(20, min_periods=5).mean()
    vol_std = np.log1p(volume).rolling(20, min_periods=5).std()
    df["volume_z_20d"] = (np.log1p(volume) - vol_mean) / vol_std.replace(0.0, np.nan)
    df["ma20_dist"] = close / close.rolling(20, min_periods=5).mean() - 1.0
    df["ma60_dist"] = close / close.rolling(60, min_periods=10).mean() - 1.0
    df["drawdown_60d"] = close / close.rolling(60, min_periods=10).max() - 1.0
    df["future_return"] = close.shift(-horizon) / close - 1.0
    fwd_lows = pd.concat([close.shift(-i) / close - 1.0 for i in range(1, horizon + 1)], axis=1)
    df["future_min_return"] = fwd_lows.min(axis=1)

    spy_cols = spy[["date", "close"]].rename(columns={"close": "spy_close"})
    df = df.merge(spy_cols, on="date", how="left")
    spy_close = df["spy_close"].ffill().bfill()
    for n in (1, 5, 20, 60):
        df[f"spy_ret_{n}d"] = np.log(spy_close / spy_close.shift(n))
    df["future_spy_return"] = spy_close.shift(-horizon) / spy_close - 1.0
    df["future_alpha"] = df["future_return"] - df["future_spy_return"]
    for n in (1, 5, 20, 60):
        df[f"rel_spy_{n}d"] = df[f"ret_{n}d"] - df[f"spy_ret_{n}d"]
        spy_abs = df[f"spy_ret_{n}d"].abs()
        ret_abs = df[f"ret_{n}d"].abs()
        df[f"rel_spy_speed_{n}d"] = (df[f"rel_spy_{n}d"] / (spy_abs + 0.01)).clip(-5.0, 5.0)
        same_dir = np.sign(df[f"ret_{n}d"]) == np.sign(df[f"spy_ret_{n}d"])
        active_dir = (ret_abs > 1e-6) & (spy_abs > 1e-6)
        df[f"spy_same_dir_{n}d"] = (same_dir & active_dir).astype(float)
        df[f"spy_opposite_dir_{n}d"] = (~same_dir & active_dir).astype(float)
        if n > 1:
            df[f"spy_lagging_same_dir_{n}d"] = (same_dir & active_dir & (ret_abs < spy_abs)).astype(float)
            df[f"spy_leading_same_dir_{n}d"] = (same_dir & active_dir & (ret_abs > spy_abs)).astype(float)
    df["profit_label"] = (df["future_return"] > 0).astype(float)
    df["crash_label"] = ((df["future_return"] < -0.04) | (df["future_min_return"] < -0.06)).astype(float)
    df["utility"] = (
        df["future_return"].clip(-0.20, 0.30)
        + 0.50 * df["future_alpha"].clip(-0.20, 0.30)
        - 1.50 * df["future_min_return"].clip(upper=0.0).abs()
        - 0.50 * df["crash_label"]
    )
    df["alpha_utility"] = (
        1.50 * df["future_alpha"].clip(-0.20, 0.30)
        - 1.25 * df["future_min_return"].clip(upper=0.0).abs()
        - 0.50 * df["crash_label"]
    )
    return df


def build_dataset(config: Config) -> pd.DataFrame:
    symbols = load_symbols(config)
    if "SPY" not in symbols and _cache_path("SPY").exists():
        symbols = ["SPY"] + symbols
    spy_daily = daily_bars("SPY")
    frames = []
    for idx, sym in enumerate(symbols, start=1):
        try:
            daily = daily_bars(sym)
            if len(daily) < config.min_rows:
                continue
            feat = add_features(daily, spy_daily, config.horizon_days)
            frames.append(feat)
        except Exception as exc:
            print(f"[daily-ranker] skip {sym}: {exc}", flush=True)
        if idx % 50 == 0:
            print(f"[daily-ranker] featurized {idx}/{len(symbols)}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    if config.start_date:
        date = pd.to_datetime(df["date"], utc=True)
        df = df[date >= pd.Timestamp(config.start_date, tz="UTC")]
    if config.end_date:
        date = pd.to_datetime(df["date"], utc=True)
        df = df[date < pd.Timestamp(config.end_date, tz="UTC")]
    for col in ("ret_5d", "ret_20d", "vol_20d", "drawdown_60d", "volume_z_20d"):
        rank_col = {
            "ret_5d": "xsec_ret_5d_rank",
            "ret_20d": "xsec_ret_20d_rank",
            "vol_20d": "xsec_vol_20d_rank",
            "drawdown_60d": "xsec_drawdown_60d_rank",
            "volume_z_20d": "xsec_volume_z_20d_rank",
        }[col]
        df[rank_col] = df.groupby("date")[col].rank(pct=True)
    df = add_market_context(df)
    if config.utility_mode == "alpha":
        df["target_utility"] = df["alpha_utility"]
    elif config.utility_mode == "return":
        df["target_utility"] = df["utility"]
    else:
        raise ValueError(f"unsupported utility_mode={config.utility_mode!r}")
    df["top_utility_label"] = (df.groupby("date")["target_utility"].rank(pct=True) >= 0.90).astype(float)
    keep = [
        "date", "symbol", "close", "future_return", "future_spy_return", "future_alpha",
        "future_min_return", "profit_label", "crash_label", "top_utility_label",
        "utility", "alpha_utility", "target_utility",
    ] + FEATURE_COLS
    df = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


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


def make_arrays(df: pd.DataFrame, train_mask: np.ndarray) -> dict:
    x = df[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    mean = x[train_mask].mean(axis=0)
    std = np.where(x[train_mask].std(axis=0) < 1e-8, 1.0, x[train_mask].std(axis=0))
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    target_col = "target_utility" if "target_utility" in df.columns else "utility"
    y = df[target_col].astype(float).to_numpy(np.float32)
    y_mean = float(y[train_mask].mean())
    y_std = float(max(y[train_mask].std(), 1e-8))
    return {
        "x": x,
        "utility": ((y - y_mean) / y_std).astype(np.float32),
        "profit": df["profit_label"].astype(float).to_numpy(np.float32),
        "crash": df["crash_label"].astype(float).to_numpy(np.float32),
        "top": df["top_utility_label"].astype(float).to_numpy(np.float32),
        "x_mean": mean,
        "x_std": std,
        "utility_mean": y_mean,
        "utility_std": y_std,
    }


def loader(arrays: dict, mask: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    idx = np.where(mask)[0]
    ds = TensorDataset(
        torch.from_numpy(arrays["x"][idx]),
        torch.from_numpy(arrays["utility"][idx]),
        torch.from_numpy(arrays["profit"][idx]),
        torch.from_numpy(arrays["crash"][idx]),
        torch.from_numpy(arrays["top"][idx]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(df: pd.DataFrame, config: Config, train_mask: np.ndarray, val_mask: np.ndarray) -> tuple[DailyRanker, dict, dict]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = pick_device(config.device)
    arrays = make_arrays(df, train_mask)
    model = DailyRanker(len(FEATURE_COLS), config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    top_rate = float(max(arrays["top"][train_mask].mean(), 1e-4))
    top_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1.0 - top_rate) / top_rate, device=device))
    train_loader = loader(arrays, train_mask, config.batch_size, True)
    val_loader = loader(arrays, val_mask, config.batch_size * 2, False)
    best = math.inf
    best_state = None
    history = []
    print(f"[daily-ranker] train={int(train_mask.sum()):,} val={int(val_mask.sum()):,} device={device}", flush=True)
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for xb, yutil, yprofit, ycrash, ytop in train_loader:
            xb = xb.to(device)
            yutil = yutil.to(device)
            yprofit = yprofit.to(device)
            ycrash = ycrash.to(device)
            ytop = ytop.to(device)
            opt.zero_grad(set_to_none=True)
            pred_util, pred_profit, pred_crash, pred_top = model(xb)
            loss = (
                mse(pred_util, yutil)
                + 0.20 * bce(pred_profit, yprofit)
                + 0.40 * bce(pred_crash, ycrash)
                + 0.25 * top_bce(pred_top, ytop)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        val = evaluate_loss(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val}
        history.append(row)
        print(f"[daily-ranker] epoch {epoch+1}/{config.epochs} train={row['train_loss']:.4f} val={row['loss']:.4f}", flush=True)
        if row["loss"] < best:
            best = row["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, arrays, {"history": history, "best_val_loss": best, "device": device}


def evaluate_loss(model: DailyRanker, data_loader: DataLoader, device: str) -> dict:
    model.eval()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    losses = []
    with torch.no_grad():
        for xb, yutil, yprofit, ycrash, ytop in data_loader:
            xb = xb.to(device)
            yutil = yutil.to(device)
            yprofit = yprofit.to(device)
            ycrash = ycrash.to(device)
            ytop = ytop.to(device)
            pred_util, pred_profit, pred_crash, pred_top = model(xb)
            loss = mse(pred_util, yutil) + 0.20 * bce(pred_profit, yprofit) + 0.40 * bce(pred_crash, ycrash) + 0.25 * bce(pred_top, ytop)
            losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


def score_frame(
    model: DailyRanker,
    df: pd.DataFrame,
    arrays: dict,
    mask: np.ndarray,
    device: str,
    batch_size: int,
    observed_score_weight: float = 0.0,
) -> pd.DataFrame:
    idx = np.where(mask)[0]
    x = torch.from_numpy(arrays["x"][idx])
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            xb = x[start:start + batch_size].to(device)
            util, profit, crash, top = model(xb)
            preds.append(torch.stack([util, torch.sigmoid(profit), torch.sigmoid(crash), torch.sigmoid(top)], dim=1).cpu().numpy())
    arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 4), dtype=np.float32)
    out = df.iloc[idx].copy().reset_index(drop=True)
    out["pred_utility"] = arr[:, 0] * arrays["utility_std"] + arrays["utility_mean"]
    out["pred_profit"] = arr[:, 1]
    out["pred_crash"] = arr[:, 2]
    out["pred_top"] = arr[:, 3]
    out["pred_score"] = out["pred_utility"] + 0.04 * out["pred_profit"] + 0.08 * out["pred_top"] - 0.10 * out["pred_crash"]
    if observed_score_weight:
        observed_edge = (
            0.40 * out["xsec_ret_20d_rank"].astype(float)
            + 0.30 * out["xsec_drawdown_60d_rank"].astype(float)
            - 0.30 * out["xsec_vol_20d_rank"].astype(float)
            + 0.10 * out["rel_spy_20d"].astype(float).clip(-0.10, 0.10) / 0.10
            - 0.20
        )
        out["observed_edge_score"] = observed_edge
        out["pred_score"] = out["pred_score"] + observed_score_weight * observed_edge
    return out


def simulate(
    scored: pd.DataFrame,
    top_k: int,
    max_positions: int,
    horizon_days: int,
    score_threshold: float,
    min_profit: float,
    max_crash: float,
    min_spy_ret_20d: float | None = None,
    min_ret_20d: float | None = None,
    min_rel_spy_20d: float | None = None,
    min_drawdown_60d: float | None = None,
    max_vol_20d_rank: float | None = None,
    min_mkt_pct_positive_20d: float | None = None,
    min_mkt_pct_above_ma20: float | None = None,
    min_mkt_ret_20d_mean: float | None = None,
    max_mkt_ret_20d_dispersion: float | None = None,
    roundtrip_cost: float = 0.0015,
) -> dict:
    active = scored[
        (scored["pred_score"] >= score_threshold)
        & (scored["pred_profit"] >= min_profit)
        & (scored["pred_crash"] <= max_crash)
    ].copy()
    if min_spy_ret_20d is not None:
        active = active[active["spy_ret_20d"] >= min_spy_ret_20d]
    if min_ret_20d is not None:
        active = active[active["ret_20d"] >= min_ret_20d]
    if min_rel_spy_20d is not None:
        active = active[active["rel_spy_20d"] >= min_rel_spy_20d]
    if min_drawdown_60d is not None:
        active = active[active["drawdown_60d"] >= min_drawdown_60d]
    if max_vol_20d_rank is not None:
        active = active[active["xsec_vol_20d_rank"] <= max_vol_20d_rank]
    if min_mkt_pct_positive_20d is not None and "mkt_pct_positive_20d" in active.columns:
        active = active[active["mkt_pct_positive_20d"] >= min_mkt_pct_positive_20d]
    if min_mkt_pct_above_ma20 is not None and "mkt_pct_above_ma20" in active.columns:
        active = active[active["mkt_pct_above_ma20"] >= min_mkt_pct_above_ma20]
    if min_mkt_ret_20d_mean is not None and "mkt_ret_20d_mean" in active.columns:
        active = active[active["mkt_ret_20d_mean"] >= min_mkt_ret_20d_mean]
    if max_mkt_ret_20d_dispersion is not None and "mkt_ret_20d_dispersion" in active.columns:
        active = active[active["mkt_ret_20d_dispersion"] <= max_mkt_ret_20d_dispersion]
    selected = []
    next_date = pd.Timestamp.min.tz_localize("UTC")
    for date, group in active.sort_values(["date", "pred_score"], ascending=[True, False]).groupby("date", sort=True):
        date = pd.Timestamp(date)
        if date < next_date:
            continue
        selected.append(group.head(top_k))
        next_date = date + pd.Timedelta(days=max(1, horizon_days))
    trades = pd.concat(selected, ignore_index=True) if selected else active.iloc[:0].copy()
    if max_positions > 0 and not trades.empty:
        # For daily horizon labels this approximates equal exposure per selected date.
        trades = trades.groupby("date", group_keys=False).head(max_positions).reset_index(drop=True)
    returns = trades["future_return"].astype(float).to_numpy() - roundtrip_cost
    if len(returns) == 0:
        return {
            "trades": 0,
            "periods": 0,
            "total_return": 0.0,
            "spy_active_return": 0.0,
            "active_alpha_return": 0.0,
            "profit_rate": 0.0,
            "mean_return": 0.0,
            "max_drawdown": 0.0,
            "beat_spy_rate": 0.0,
        }
    trades = trades.copy()
    trades["_net_return"] = returns
    period = trades.groupby("date")["_net_return"].mean().sort_index()
    if "future_spy_return" in trades.columns:
        spy_returns = trades["future_spy_return"].astype(float)
    else:
        spy_returns = trades["future_return"].astype(float) - trades["future_alpha"].astype(float)
    spy_period = spy_returns.groupby(trades["date"]).mean().reindex(period.index).fillna(0.0)
    eq = (1.0 + period).cumprod().to_numpy()
    spy_eq = (1.0 + spy_period.to_numpy()).cumprod()
    peaks = np.maximum.accumulate(np.r_[1.0, eq])
    curve = np.r_[1.0, eq]
    dd = (curve - peaks) / np.maximum(peaks, 1e-12)
    return {
        "trades": int(len(trades)),
        "periods": int(period.shape[0]),
        "total_return": float(eq[-1] - 1.0),
        "spy_active_return": float(spy_eq[-1] - 1.0) if len(spy_eq) else 0.0,
        "active_alpha_return": float((eq[-1] - 1.0) - (spy_eq[-1] - 1.0)) if len(spy_eq) else float(eq[-1] - 1.0),
        "profit_rate": float((returns > 0).mean()),
        "beat_spy_rate": float((trades["future_alpha"].astype(float) > 0).mean()),
        "mean_return": float(np.mean(returns)),
        "max_drawdown": float(dd.min()),
        "symbols": trades["symbol"].value_counts().head(20).to_dict(),
    }


def _rule_payload(row: dict) -> dict:
    rule_keys = (
        "score_threshold", "min_profit", "max_crash", "min_spy_ret_20d",
        "min_ret_20d", "min_rel_spy_20d", "min_drawdown_60d", "max_vol_20d_rank",
        "min_mkt_pct_positive_20d", "min_mkt_pct_above_ma20",
        "min_mkt_ret_20d_mean", "max_mkt_ret_20d_dispersion",
    )
    return {k: row[k] for k in rule_keys}


def _simulate_rule(scored: pd.DataFrame, config: Config, rule: dict) -> dict:
    return simulate(
        scored,
        config.top_k,
        config.max_positions,
        config.horizon_days,
        rule["score_threshold"],
        rule["min_profit"],
        rule["max_crash"],
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


def _passes_validation(result: dict, config: Config, min_trades: int) -> bool:
    if result["trades"] < min_trades:
        return False
    if result["total_return"] < config.min_validation_return:
        return False
    if result["active_alpha_return"] < config.min_validation_active_alpha:
        return False
    if result["profit_rate"] < config.min_validation_profit_rate:
        return False
    if result["beat_spy_rate"] < config.min_validation_beat_spy_rate:
        return False
    if abs(result["max_drawdown"]) > config.max_validation_drawdown:
        return False
    return True


def _rule_objective(result: dict) -> float:
    return (
        1.50 * result["active_alpha_return"]
        + 0.50 * result["total_return"]
        - 3.0 * abs(result["max_drawdown"])
        + 0.05 * result["profit_rate"]
        + 0.02 * result["beat_spy_rate"]
    )


def choose_rule(val_scored: pd.DataFrame, config: Config) -> dict:
    scored_for_search = val_scored
    scored_for_holdout = None
    if config.rule_validation_fraction > 0:
        dates = pd.to_datetime(val_scored["date"], utc=True)
        cutoff = dates.quantile(1.0 - max(0.0, min(config.rule_validation_fraction, 0.8)))
        scored_for_search = val_scored[dates < cutoff].copy()
        scored_for_holdout = val_scored[dates >= cutoff].copy()

    thresholds = [float(scored_for_search["pred_score"].quantile(q)) for q in (0.80, 0.90, 0.95)]
    min_profits = (0.55, 0.60)
    max_crashes = (0.40, 0.30)
    regime_filters = (0.0,)
    stock_trend_filters = (None, 0.03)
    relative_filters = (None, 0.0)
    drawdown_filters = (-0.15, -0.05)
    vol_rank_filters = (None, 0.80)
    breadth_filters = (None, 0.55)
    ma_breadth_filters = (None, 0.55)
    market_return_filters = (None, 0.0)
    dispersion_filters = (None, 0.08)
    best = None
    candidates = []
    for threshold in thresholds:
        for min_profit in min_profits:
            for max_crash in max_crashes:
                for min_spy_ret_20d in regime_filters:
                    for min_ret_20d in stock_trend_filters:
                        for min_rel_spy_20d in relative_filters:
                            for min_drawdown_60d in drawdown_filters:
                                for max_vol_20d_rank in vol_rank_filters:
                                    for min_mkt_pct_positive_20d in breadth_filters:
                                        for min_mkt_pct_above_ma20 in ma_breadth_filters:
                                            for min_mkt_ret_20d_mean in market_return_filters:
                                                for max_mkt_ret_20d_dispersion in dispersion_filters:
                                                    row = {
                                                        "score_threshold": threshold,
                                                        "min_profit": min_profit,
                                                        "max_crash": max_crash,
                                                        "min_spy_ret_20d": min_spy_ret_20d,
                                                        "min_ret_20d": min_ret_20d,
                                                        "min_rel_spy_20d": min_rel_spy_20d,
                                                        "min_drawdown_60d": min_drawdown_60d,
                                                        "max_vol_20d_rank": max_vol_20d_rank,
                                                        "min_mkt_pct_positive_20d": min_mkt_pct_positive_20d,
                                                        "min_mkt_pct_above_ma20": min_mkt_pct_above_ma20,
                                                        "min_mkt_ret_20d_mean": min_mkt_ret_20d_mean,
                                                        "max_mkt_ret_20d_dispersion": max_mkt_ret_20d_dispersion,
                                                    }
                                                    rule = _rule_payload(row)
                                                    result = _simulate_rule(scored_for_search, config, rule)
                                                    if not _passes_validation(result, config, config.min_validation_trades):
                                                        continue
                                                    row["search_validation"] = result
                                                    objective_result = result
                                                    if scored_for_holdout is not None:
                                                        holdout = _simulate_rule(scored_for_holdout, config, rule)
                                                        if not _passes_validation(holdout, config, config.min_rule_validation_trades):
                                                            continue
                                                        row["rule_holdout_validation"] = holdout
                                                        objective_result = holdout
                                                    score = _rule_objective(objective_result)
                                                    row["objective"] = score
                                                    row["validation"] = objective_result
                                                    candidates.append(row)
                                                    if best is None or score > best["objective"]:
                                                        best = row
    if best is None:
        return {"rule": {"no_trade": True, "score_threshold": float("inf"), "min_profit": 1.0, "max_crash": 0.0}, "candidates": []}
    return {"rule": _rule_payload(best), "best": best, "candidates": sorted(candidates, key=lambda x: x["objective"], reverse=True)[:10]}


def run(config: Config) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "daily_ranker_dataset.parquet"
    if dataset_path.exists():
        df = pd.read_parquet(dataset_path)
    else:
        df = build_dataset(config)
        df.to_parquet(dataset_path, index=False)
    train_mask, val_mask, test_mask = split_masks(df, config)
    model, arrays, train_meta = train_model(df, config, train_mask, val_mask)
    device = train_meta["device"]
    val_scored = score_frame(model, df, arrays, val_mask, device, config.batch_size, config.observed_score_weight)
    test_scored = score_frame(model, df, arrays, test_mask, device, config.batch_size, config.observed_score_weight)
    choice = choose_rule(val_scored, config)
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
        "arrays": {k: v for k, v in arrays.items() if k in {"x_mean", "x_std", "utility_mean", "utility_std"}},
        "rule": rule,
        "train_meta": train_meta,
    }
    torch.save(ckpt, output_dir / "daily_ranker.pt")
    payload = {
        "config": asdict(config),
        "rows": int(len(df)),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_meta": train_meta,
        "rule_selection": choice,
        "test_result": test_result,
        "checkpoint": str(output_dir / "daily_ranker.pt"),
        "dataset": str(dataset_path),
    }
    (output_dir / "daily_ranker_result.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="checkpoints/daily_ranker/exp1_2025")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-01-01")
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2026-01-01")
    parser.add_argument("--horizon-days", type=int, default=5)
    parser.add_argument("--top500", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cached-all", action="store_true")
    parser.add_argument("--symbol-limit", type=int, default=503)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--min-validation-trades", type=int, default=50)
    parser.add_argument("--min-validation-return", type=float, default=0.0)
    parser.add_argument("--min-validation-active-alpha", type=float, default=0.0)
    parser.add_argument("--min-validation-profit-rate", type=float, default=0.52)
    parser.add_argument("--min-validation-beat-spy-rate", type=float, default=0.50)
    parser.add_argument("--max-validation-drawdown", type=float, default=0.10)
    parser.add_argument("--rule-validation-fraction", type=float, default=0.0)
    parser.add_argument("--min-rule-validation-trades", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--observed-score-weight", type=float, default=0.0)
    parser.add_argument("--utility-mode", choices=("return", "alpha"), default="return")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
