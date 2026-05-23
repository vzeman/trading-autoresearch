"""Train a causal day-by-day patched daily trading model.

Protocol:

1. Build or load a 1-day-horizon daily dataset.
2. Train on the first calendar year of feature-complete rows.
3. For each following decision date:
   - score that day's cross-section with the current model,
   - buy the top candidates for the next trading day or hold cash,
   - record portfolio and SPY benchmark returns,
   - patch-train the model on that date's now-realized labels.

This is intentionally separate from the fixed-fold rankers. It tests whether a
continuously patched model behaves better than retraining isolated folds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_daily_ranker import Config as DailyConfig
from train_daily_ranker import FEATURE_COLS, build_dataset, daily_bars, pick_device


@dataclass(frozen=True)
class Config:
    dataset: str
    output_dir: str
    build_dataset: bool
    start_date: str
    end_date: str
    cached_all: bool
    symbol_limit: int
    min_rows: int
    initial_years: float
    initial_epochs: int
    patch_epochs: int
    hidden_dim: int
    dropout: float
    lr: float
    weight_decay: float
    list_temperature: float
    downside_penalty: float
    profit_loss_weight: float
    crash_loss_weight: float
    score_profit_weight: float
    score_top_weight: float
    score_crash_weight: float
    min_pred_profit: float
    max_pred_crash: float
    min_pred_top: float
    min_spy_ret_20d: float
    min_mkt_pct_above_ma20: float
    benchmark_symbol: str
    require_benchmark_history: bool
    benchmark_warmup_days: int
    min_close: float
    max_abs_daily_return: float
    max_abs_spy_daily_return: float
    max_positions: int
    roundtrip_cost: float
    starting_equity: float
    device: str
    seed: int


class OnlineListwiseRanker(nn.Module):
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


def maybe_build_dataset(config: Config) -> None:
    path = Path(config.dataset)
    if path.exists() and not config.build_dataset:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_config = DailyConfig(
        output_dir=str(path.parent),
        start_date=config.start_date,
        end_date=config.end_date,
        train_end="",
        test_start="",
        test_end="",
        horizon_days=1,
        top500=True,
        cached_all=config.cached_all,
        symbol_limit=config.symbol_limit,
        min_rows=config.min_rows,
        epochs=1,
        batch_size=8192,
        hidden_dim=128,
        dropout=0.20,
        lr=3e-4,
        weight_decay=1e-3,
        validation_fraction=0.25,
        min_validation_trades=20,
        min_validation_return=0.0,
        min_validation_active_alpha=0.0,
        min_validation_profit_rate=0.52,
        min_validation_beat_spy_rate=0.50,
        max_validation_drawdown=0.10,
        rule_validation_fraction=0.25,
        min_rule_validation_trades=5,
        top_k=3,
        max_positions=3,
        observed_score_weight=0.0,
        utility_mode="alpha",
        device=config.device,
        seed=config.seed,
    )
    df = build_dataset(daily_config)
    df.to_parquet(path, index=False)
    print(f"[online-patch] wrote dataset rows={len(df):,} path={path}", flush=True)


def load_frame(config: Config) -> pd.DataFrame:
    maybe_build_dataset(config)
    df = pd.read_parquet(config.dataset)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(f"dataset missing {len(missing)} required feature columns, rebuild it: {missing[:5]}")
    before = len(df)
    if config.require_benchmark_history:
        benchmark = daily_bars(config.benchmark_symbol)
        benchmark_dates = sorted(pd.to_datetime(benchmark["date"], utc=True).unique())
        if len(benchmark_dates) <= config.benchmark_warmup_days + 1:
            raise RuntimeError(f"not enough {config.benchmark_symbol} bars for benchmark-aware online training")
        usable_dates = set(benchmark_dates[config.benchmark_warmup_days:-1])
        df = df[df["date"].isin(usable_dates)].copy()
        print(
            f"[online-patch] benchmark usable dates "
            f"{pd.Timestamp(benchmark_dates[config.benchmark_warmup_days]).date()} -> "
            f"{pd.Timestamp(benchmark_dates[-2]).date()}",
            flush=True,
        )
    clean = (
        df["close"].astype(float).ge(config.min_close)
        & df["future_return"].astype(float).between(-config.max_abs_daily_return, config.max_abs_daily_return)
        & df["future_min_return"].astype(float).between(-config.max_abs_daily_return, config.max_abs_daily_return)
        & df["ret_1d"].astype(float).between(-config.max_abs_daily_return, config.max_abs_daily_return)
        & df["future_spy_return"].astype(float).between(-config.max_abs_spy_daily_return, config.max_abs_spy_daily_return)
    )
    df = df[clean].copy().reset_index(drop=True)
    print(f"[online-patch] cleaned rows {before:,} -> {len(df):,}", flush=True)
    return df


def target_values(df: pd.DataFrame, config: Config) -> np.ndarray:
    alpha = df["future_alpha"].astype(float).to_numpy(np.float32)
    downside = np.maximum(-df["future_min_return"].astype(float).to_numpy(np.float32), 0.0)
    return alpha - float(config.downside_penalty) * downside


def make_arrays(df: pd.DataFrame, warm_mask: np.ndarray, config: Config) -> dict:
    x = df[FEATURE_COLS].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    mean = x[warm_mask].mean(axis=0)
    std = np.where(x[warm_mask].std(axis=0) < 1e-8, 1.0, x[warm_mask].std(axis=0))
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    return {
        "x": x,
        "target": target_values(df, config).astype(np.float32),
        "profit": df["profit_label"].astype(float).to_numpy(np.float32),
        "crash": df["crash_label"].astype(float).to_numpy(np.float32),
        "x_mean": mean,
        "x_std": std,
    }


def date_groups(df: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
    out = {}
    for date, group in df.groupby("date", sort=True):
        out[pd.Timestamp(date)] = group.index.to_numpy(np.int64)
    return out


def benchmark_return_by_date(config: Config) -> dict[pd.Timestamp, float]:
    benchmark = daily_bars(config.benchmark_symbol)
    benchmark["date"] = pd.to_datetime(benchmark["date"], utc=True)
    benchmark = benchmark.sort_values("date").reset_index(drop=True)
    close = benchmark["close"].astype(float)
    benchmark["next_return"] = close.shift(-1) / close - 1.0
    return {
        pd.Timestamp(row.date): float(row.next_return)
        for row in benchmark.dropna(subset=["next_return"]).itertuples(index=False)
    }


def train_group(
    model: OnlineListwiseRanker,
    opt: torch.optim.Optimizer,
    arrays: dict,
    idx: np.ndarray,
    device: str,
    config: Config,
) -> float:
    if len(idx) == 0:
        return 0.0
    xb = torch.from_numpy(arrays["x"][idx]).to(device)
    target = torch.from_numpy(arrays["target"][idx]).to(device)
    profit = torch.from_numpy(arrays["profit"][idx]).to(device)
    crash = torch.from_numpy(arrays["crash"][idx]).to(device)
    bce = nn.BCEWithLogitsLoss()
    model.train()
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
    return float(loss.detach().cpu().item())


def initial_train(
    model: OnlineListwiseRanker,
    opt: torch.optim.Optimizer,
    arrays: dict,
    warm_dates: list[pd.Timestamp],
    groups: dict[pd.Timestamp, np.ndarray],
    device: str,
    config: Config,
) -> list[dict]:
    rng = np.random.default_rng(config.seed)
    history = []
    for epoch in range(config.initial_epochs):
        dates = list(warm_dates)
        rng.shuffle(dates)
        losses = [train_group(model, opt, arrays, groups[d], device, config) for d in dates]
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)) if losses else 0.0}
        history.append(row)
        print(f"[online-patch] initial epoch {epoch+1}/{config.initial_epochs} loss={row['loss']:.4f}", flush=True)
    return history


def score_day(
    model: OnlineListwiseRanker,
    df: pd.DataFrame,
    arrays: dict,
    idx: np.ndarray,
    device: str,
    config: Config,
) -> pd.DataFrame:
    if len(idx) == 0:
        return df.iloc[:0].copy()
    xb = torch.from_numpy(arrays["x"][idx]).to(device)
    model.eval()
    with torch.no_grad():
        score, profit, crash = model(xb)
    out = df.iloc[idx].copy().reset_index(drop=True)
    out["pred_utility"] = score.detach().cpu().numpy()
    out["pred_profit"] = torch.sigmoid(profit).detach().cpu().numpy()
    out["pred_crash"] = torch.sigmoid(crash).detach().cpu().numpy()
    out["pred_top"] = out["pred_utility"].rank(pct=True).astype(float)
    out["pred_score"] = (
        out["pred_utility"]
        + float(config.score_profit_weight) * out["pred_profit"]
        + float(config.score_top_weight) * out["pred_top"]
        - float(config.score_crash_weight) * out["pred_crash"]
    )
    return out.sort_values("pred_score", ascending=False)


def select_positions(scored: pd.DataFrame, config: Config) -> pd.DataFrame:
    if scored.empty:
        return scored
    active = scored[
        (scored["pred_profit"] >= config.min_pred_profit)
        & (scored["pred_crash"] <= config.max_pred_crash)
        & (scored["pred_top"] >= config.min_pred_top)
    ].copy()
    if config.min_spy_ret_20d > -99 and "spy_ret_20d" in active.columns:
        active = active[active["spy_ret_20d"] >= config.min_spy_ret_20d]
    if config.min_mkt_pct_above_ma20 > 0 and "mkt_pct_above_ma20" in active.columns:
        active = active[active["mkt_pct_above_ma20"] >= config.min_mkt_pct_above_ma20]
    return active.sort_values("pred_score", ascending=False).head(config.max_positions)


def max_drawdown(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    dd = arr / np.maximum(peaks, 1e-12) - 1.0
    return float(dd.min())


def plot_equity(curve: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(curve["date"], curve["equity"], label="online patched model", linewidth=1.6)
    ax.plot(curve["date"], curve["spy_equity"], label="SPY buy-and-hold", linewidth=1.4, linestyle="--")
    ax.set_title("Online patched daily model vs SPY")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def run(config: Config) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(config.device)
    df = load_frame(config)
    groups = date_groups(df)
    benchmark_returns = benchmark_return_by_date(config)
    dates = sorted(groups)
    if not dates:
        raise RuntimeError("empty dataset")
    first_date = dates[0]
    warm_end = first_date + pd.DateOffset(days=int(round(365.25 * config.initial_years)))
    warm_dates = [d for d in dates if d < warm_end]
    trade_dates = [d for d in dates if d >= warm_end]
    if not warm_dates or not trade_dates:
        raise RuntimeError("not enough dates for warmup and online walk")

    warm_mask = df["date"].isin(warm_dates).to_numpy()
    arrays = make_arrays(df, warm_mask, config)
    model = OnlineListwiseRanker(len(FEATURE_COLS), config.hidden_dim, config.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    print(
        f"[online-patch] rows={len(df):,} dates={len(dates):,} "
        f"warmup={len(warm_dates):,} walk={len(trade_dates):,} device={device}",
        flush=True,
    )
    history = initial_train(model, opt, arrays, warm_dates, groups, device, config)

    equity = float(config.starting_equity)
    spy_equity = float(config.starting_equity)
    curve_rows = []
    trade_rows = []
    patch_losses = []
    previous_symbols: set[str] = set()
    for i, date in enumerate(trade_dates, start=1):
        scored = score_day(model, df, arrays, groups[date], device, config)
        selected = select_positions(scored, config)
        spy_return = float(benchmark_returns.get(date, 0.0))
        if selected.empty:
            portfolio_return = 0.0
            symbols: list[str] = []
            action = "hold_cash"
        else:
            symbols = selected["symbol"].astype(str).tolist()
            portfolio_return = float(selected["future_return"].astype(float).mean() - config.roundtrip_cost)
            action = "buy"
        sells = sorted(previous_symbols - set(symbols))
        buys = sorted(set(symbols) - previous_symbols)
        previous_symbols = set(symbols)
        equity *= 1.0 + portfolio_return
        spy_equity *= 1.0 + spy_return
        curve_rows.append({
            "date": str(date.date()),
            "equity": equity,
            "spy_equity": spy_equity,
            "portfolio_return": portfolio_return,
            "spy_return": spy_return,
            "action": action,
            "symbols": ",".join(symbols),
            "buys": ",".join(buys),
            "sells": ",".join(sells),
        })
        if not selected.empty:
            for _, row in selected.iterrows():
                trade_rows.append({
                    "date": str(date.date()),
                    "symbol": row["symbol"],
                    "pred_score": float(row["pred_score"]),
                    "pred_profit": float(row["pred_profit"]),
                    "pred_crash": float(row["pred_crash"]),
                    "future_return": float(row["future_return"]),
                    "future_spy_return": spy_return,
                    "future_alpha": float(row["future_return"]) - spy_return,
                })

        for _ in range(config.patch_epochs):
            patch_losses.append(train_group(model, opt, arrays, groups[date], device, config))
        if i % 250 == 0:
            print(
                f"[online-patch] day {i:,}/{len(trade_dates):,} "
                f"equity={equity:,.2f} spy={spy_equity:,.2f} trades={len(trade_rows):,}",
                flush=True,
            )

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    curve_path = output_dir / "online_equity_curve.csv"
    trades_path = output_dir / "online_trades.csv"
    chart_path = output_dir / "online_equity.png"
    docs_chart = Path("docs/online_daily_patch_equity.png")
    curve.to_csv(curve_path, index=False)
    trades.to_csv(trades_path, index=False)
    plot_equity(curve.assign(date=pd.to_datetime(curve["date"])), chart_path)
    plot_equity(curve.assign(date=pd.to_datetime(curve["date"])), docs_chart)
    trade_returns = trades["future_return"].astype(float).to_numpy() if not trades.empty else np.array([])
    summary = {
        "config": asdict(config),
        "dataset_rows": int(len(df)),
        "first_date": str(first_date.date()),
        "warmup_end": str(pd.Timestamp(warm_end).date()),
        "decision_start": str(trade_dates[0].date()),
        "decision_end": str(trade_dates[-1].date()),
        "decision_days": int(len(trade_dates)),
        "active_days": int((curve["action"] == "buy").sum()),
        "trades": int(len(trades)),
        "final_equity": float(equity),
        "spy_final_equity": float(spy_equity),
        "total_return": float(equity / config.starting_equity - 1.0),
        "spy_total_return": float(spy_equity / config.starting_equity - 1.0),
        "active_alpha_return": float(equity / config.starting_equity - spy_equity / config.starting_equity),
        "max_drawdown": max_drawdown(curve["equity"].astype(float).tolist()),
        "spy_max_drawdown": max_drawdown(curve["spy_equity"].astype(float).tolist()),
        "trade_profit_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
        "mean_patch_loss": float(np.mean(patch_losses)) if patch_losses else 0.0,
        "initial_train_history": history,
        "curve": str(curve_path),
        "trades_csv": str(trades_path),
        "chart": str(chart_path),
        "docs_chart": str(docs_chart),
        "warning": "research_only_daily_close_rebalanced_online_backtest",
    }
    (output_dir / "online_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_cols": FEATURE_COLS,
            "x_mean": arrays["x_mean"],
            "x_std": arrays["x_std"],
            "config": asdict(config),
            "summary": summary,
        },
        output_dir / "online_daily_patch_model.pt",
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="checkpoints/daily_ranker/exp14_online_h1_relspy/daily_ranker_dataset.parquet")
    parser.add_argument("--output-dir", default="checkpoints/online_daily_patch/exp1_h1_relspy")
    parser.add_argument("--build-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2026-05-16")
    parser.add_argument("--cached-all", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--symbol-limit", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--initial-years", type=float, default=1.0)
    parser.add_argument("--initial-epochs", type=int, default=8)
    parser.add_argument("--patch-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--list-temperature", type=float, default=0.05)
    parser.add_argument("--downside-penalty", type=float, default=2.0)
    parser.add_argument("--profit-loss-weight", type=float, default=0.15)
    parser.add_argument("--crash-loss-weight", type=float, default=0.80)
    parser.add_argument("--score-profit-weight", type=float, default=0.04)
    parser.add_argument("--score-top-weight", type=float, default=0.08)
    parser.add_argument("--score-crash-weight", type=float, default=0.35)
    parser.add_argument("--min-pred-profit", type=float, default=0.52)
    parser.add_argument("--max-pred-crash", type=float, default=0.55)
    parser.add_argument("--min-pred-top", type=float, default=0.98)
    parser.add_argument("--min-spy-ret-20d", type=float, default=-99.0)
    parser.add_argument("--min-mkt-pct-above-ma20", type=float, default=0.0)
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--require-benchmark-history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-warmup-days", type=int, default=70)
    parser.add_argument("--min-close", type=float, default=1.0)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.35)
    parser.add_argument("--max-abs-spy-daily-return", type=float, default=0.15)
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--roundtrip-cost", type=float, default=0.0015)
    parser.add_argument("--starting-equity", type=float, default=50_000.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
