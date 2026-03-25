"""
optimizer.py - Train/test strategy lab and parameter optimization helpers.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from analytics import PerformanceMetrics
from engine import BacktestEngine
from reporting import RegimeLens, compute_monthly_stats
from strategies import (
    DonchianBreakoutStrategy,
    MACDRSIStrategy,
    MACrossStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    Strategy,
)


@dataclass(frozen=True)
class StrategyLabSpec:
    family_key: str
    family_name: str
    factory: Callable[..., Strategy]
    grid: list[dict]
    label_builder: Callable[[dict], str]


def get_default_strategy_lab_specs() -> list[StrategyLabSpec]:
    return [
        StrategyLabSpec(
            family_key="ma_cross",
            family_name="MA Cross",
            factory=MACrossStrategy,
            grid=[
                {"fast": 10, "slow": 30, "ma_type": "ema", "trend_filter": False, "volume_confirm": False},
                {"fast": 15, "slow": 45, "ma_type": "ema", "trend_filter": False, "volume_confirm": False},
                {"fast": 20, "slow": 50, "ma_type": "ema", "trend_filter": False, "volume_confirm": False},
                {"fast": 20, "slow": 60, "ma_type": "ema", "trend_filter": True, "volume_confirm": False},
                {"fast": 30, "slow": 90, "ma_type": "ema", "trend_filter": True, "volume_confirm": False},
            ],
            label_builder=_format_ma_cross_params,
        ),
        StrategyLabSpec(
            family_key="momentum",
            family_name="Momentum",
            factory=MomentumStrategy,
            grid=[
                {"lookback": 84, "skip": 10, "rsi_entry_cap": 78, "trend_ma": 80},
                {"lookback": 126, "skip": 10, "rsi_entry_cap": 80, "trend_ma": 100},
                {"lookback": 126, "skip": 21, "rsi_entry_cap": 75, "trend_ma": 100},
                {"lookback": 189, "skip": 21, "rsi_entry_cap": 80, "trend_ma": 120},
            ],
            label_builder=_format_momentum_params,
        ),
        StrategyLabSpec(
            family_key="mean_reversion",
            family_name="Mean Reversion",
            factory=MeanReversionStrategy,
            grid=[
                {"bb_period": 20, "bb_std": 2.0, "zscore_period": 20, "zscore_entry": -1.6, "zscore_exit": -0.1, "rsi_oversold": 40},
                {"bb_period": 20, "bb_std": 2.0, "zscore_period": 20, "zscore_entry": -1.8, "zscore_exit": 0.0, "rsi_oversold": 38},
                {"bb_period": 20, "bb_std": 2.2, "zscore_period": 20, "zscore_entry": -2.0, "zscore_exit": 0.0, "rsi_oversold": 35},
                {"bb_period": 30, "bb_std": 2.2, "zscore_period": 30, "zscore_entry": -2.1, "zscore_exit": -0.2, "rsi_oversold": 35},
            ],
            label_builder=_format_mean_reversion_params,
        ),
        StrategyLabSpec(
            family_key="macd_rsi",
            family_name="MACD+RSI",
            factory=MACDRSIStrategy,
            grid=[
                {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "rsi_low": 30, "rsi_high": 72, "trend_filter": False},
                {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "rsi_low": 35, "rsi_high": 75, "trend_filter": False},
                {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "rsi_low": 40, "rsi_high": 75, "trend_filter": True},
                {"macd_fast": 8, "macd_slow": 21, "macd_signal": 5, "rsi_low": 38, "rsi_high": 72, "trend_filter": True},
            ],
            label_builder=_format_macd_params,
        ),
        StrategyLabSpec(
            family_key="donchian_breakout",
            family_name="Donchian Breakout",
            factory=DonchianBreakoutStrategy,
            grid=[
                {"entry_period": 20, "exit_period": 10, "vol_confirm": False},
                {"entry_period": 20, "exit_period": 10, "vol_confirm": True},
                {"entry_period": 30, "exit_period": 10, "vol_confirm": False},
                {"entry_period": 55, "exit_period": 20, "vol_confirm": False},
            ],
            label_builder=_format_donchian_params,
        ),
    ]


def split_train_test(
    df: pd.DataFrame,
    train_ratio: float,
    min_train_bars: int = 160,
    min_test_bars: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < (min_train_bars + min_test_bars):
        raise ValueError(
            "Optimization mode needs more history. "
            f"Need at least {min_train_bars + min_test_bars} bars, got {len(df)}."
        )

    split_idx = int(len(df) * train_ratio)
    split_idx = max(min_train_bars, min(split_idx, len(df) - min_test_bars))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Could not create a non-empty train/test split for optimization mode.")

    return train_df, test_df


def run_strategy_lab(
    df: pd.DataFrame,
    *,
    base_cost_config,
    base_risk_config,
    risk_free_rate: float = 0.065,
    train_ratio: float = 0.7,
    min_trades: int = 3,
    family_specs: list[StrategyLabSpec] | None = None,
) -> dict:
    specs = family_specs or get_default_strategy_lab_specs()
    train_df, test_df = split_train_test(df, train_ratio=train_ratio)

    regime_lens = RegimeLens()
    train_regimes = regime_lens.classify(train_df)
    test_regimes = regime_lens.classify(test_df)

    candidate_rows: list[dict] = []
    summary_rows: list[dict] = []
    selected_strategies: list[dict] = []

    for spec in specs:
        family_candidates: list[dict] = []

        for params in spec.grid:
            params_copy = copy.deepcopy(params)
            param_label = spec.label_builder(params_copy)
            candidate_name = f"{spec.family_name} [{param_label}]"
            row = {
                "family_key": spec.family_key,
                "family": spec.family_name,
                "candidate": candidate_name,
                "param_label": param_label,
                "params": params_copy,
                "params_json": json.dumps(params_copy, sort_keys=True),
                "selected": False,
                "rank_within_family": None,
                "status": "success",
                "error_type": "",
                "error_message": "",
            }

            try:
                train_eval = _evaluate_snapshot(
                    name=candidate_name,
                    strategy=spec.factory(**params_copy),
                    df=train_df,
                    benchmark=train_df["close"],
                    base_cost_config=base_cost_config,
                    base_risk_config=base_risk_config,
                    risk_free_rate=risk_free_rate,
                    regimes=train_regimes,
                    regime_lens=regime_lens,
                )
                row.update(_candidate_metrics_payload(train_eval, prefix="train"))
                row["train_score"] = _candidate_score(
                    summary=train_eval["summary"],
                    monthly_stats=train_eval["monthly_stats"],
                    regime_analysis=train_eval["regime_analysis"],
                    min_trades=min_trades,
                )
                family_candidates.append(row)
            except Exception as exc:
                row.update({
                    "status": "failed",
                    "train_score": 0.0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

            candidate_rows.append(row)

        successful = [row for row in family_candidates if row.get("status") == "success"]
        successful.sort(
            key=lambda item: (
                item.get("train_score", 0),
                item.get("train_sharpe_ratio", 0),
                item.get("train_total_return_pct", 0),
                -abs(item.get("train_max_drawdown_pct", 0)),
            ),
            reverse=True,
        )

        for rank, row in enumerate(successful, start=1):
            row["rank_within_family"] = rank

        if not successful:
            continue

        selected = successful[0]
        selected["selected"] = True
        selected_params = copy.deepcopy(selected["params"])
        selected_strategy = spec.factory(**selected_params)

        selected_name = f"{spec.family_name} [{selected['param_label']}]"
        test_error_type = ""
        test_error_message = ""
        test_score = 0.0
        test_payload = {}

        try:
            test_eval = _evaluate_snapshot(
                name=selected_name,
                strategy=selected_strategy,
                df=test_df,
                benchmark=test_df["close"],
                base_cost_config=base_cost_config,
                base_risk_config=base_risk_config,
                risk_free_rate=risk_free_rate,
                regimes=test_regimes,
                regime_lens=regime_lens,
            )
            test_payload = _candidate_metrics_payload(test_eval, prefix="test")
            test_score = _candidate_score(
                summary=test_eval["summary"],
                monthly_stats=test_eval["monthly_stats"],
                regime_analysis=test_eval["regime_analysis"],
                min_trades=min_trades,
            )
        except Exception as exc:
            test_error_type = type(exc).__name__
            test_error_message = str(exc)

        summary_rows.append({
            "family_key": spec.family_key,
            "family": spec.family_name,
            "selected_candidate": selected_name,
            "selected_param_label": selected["param_label"],
            "params": selected_params,
            "params_json": json.dumps(selected_params, sort_keys=True),
            "candidate_count": len(successful),
            "train_score": selected.get("train_score", 0.0),
            "train_total_return_pct": selected.get("train_total_return_pct", 0.0),
            "train_sharpe_ratio": selected.get("train_sharpe_ratio", 0.0),
            "train_max_drawdown_pct": selected.get("train_max_drawdown_pct", 0.0),
            "train_total_trades": selected.get("train_total_trades", 0),
            "test_score": test_score,
            "test_status": "success" if not test_error_type else "failed",
            "test_error_type": test_error_type,
            "test_error_message": test_error_message,
            **test_payload,
        })

        selected_strategies.append({
            "family_key": spec.family_key,
            "family": spec.family_name,
            "name": selected_name,
            "strategy": selected_strategy,
            "params": selected_params,
            "param_label": selected["param_label"],
            "train_score": selected.get("train_score", 0.0),
            "test_score": test_score,
            "candidate_count": len(successful),
            "test_status": "success" if not test_error_type else "failed",
            "test_error_type": test_error_type,
            "test_error_message": test_error_message,
            "test_metrics": test_payload,
        })

    if not selected_strategies:
        raise RuntimeError("Strategy lab could not find a successful candidate to promote.")

    candidate_df = pd.DataFrame(candidate_rows)
    if not candidate_df.empty:
        candidate_df = candidate_df.sort_values(
            by=["family", "rank_within_family", "train_score", "candidate"],
            ascending=[True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["test_score", "train_score", "family"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    return {
        "split": {
            "train_ratio": round(train_ratio, 4),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "train_start": str(train_df.index[0].date()),
            "train_end": str(train_df.index[-1].date()),
            "test_start": str(test_df.index[0].date()),
            "test_end": str(test_df.index[-1].date()),
        },
        "candidate_df": candidate_df,
        "summary_df": summary_df,
        "candidate_records": candidate_df.to_dict("records") if not candidate_df.empty else [],
        "summary_records": summary_df.to_dict("records") if not summary_df.empty else [],
        "selected_strategies": selected_strategies,
    }


def _evaluate_snapshot(
    *,
    name: str,
    strategy: Strategy,
    df: pd.DataFrame,
    benchmark: pd.Series,
    base_cost_config,
    base_risk_config,
    risk_free_rate: float,
    regimes: pd.Series,
    regime_lens: RegimeLens,
) -> dict:
    engine = BacktestEngine(
        strategy=strategy,
        cost_config=copy.deepcopy(base_cost_config),
        risk_config=copy.deepcopy(base_risk_config),
        symbol="ASSET",
        verbose=False,
    )
    result = engine.run(df.copy())
    equity = result.equity_curve["equity"]
    if len(equity) < 10:
        raise ValueError(f"Not enough equity observations to evaluate {name}.")

    metrics = PerformanceMetrics(
        equity,
        result.trade_df,
        benchmark=benchmark,
        rf_annual=risk_free_rate,
    )
    summary = metrics.summary()
    monthly_stats = compute_monthly_stats(equity)
    regime_analysis = regime_lens.evaluate(equity, regimes)

    return {
        "name": name,
        "strategy": strategy,
        "result": result,
        "summary": summary,
        "monthly_stats": monthly_stats,
        "regime_analysis": regime_analysis,
    }


def _candidate_metrics_payload(snapshot: dict, prefix: str) -> dict:
    summary = snapshot.get("summary", {})
    monthly = snapshot.get("monthly_stats", {})
    regime = snapshot.get("regime_analysis", {})
    return {
        f"{prefix}_total_return_pct": round(float(summary.get("total_return_pct", 0.0)), 4),
        f"{prefix}_cagr_pct": round(float(summary.get("cagr_pct", 0.0)), 4),
        f"{prefix}_sharpe_ratio": round(float(summary.get("sharpe_ratio", 0.0)), 4),
        f"{prefix}_max_drawdown_pct": round(float(summary.get("max_drawdown_pct", 0.0)), 4),
        f"{prefix}_profit_factor": round(float(summary.get("profit_factor", 0.0)), 4),
        f"{prefix}_win_rate_pct": round(float(summary.get("win_rate_pct", 0.0)), 4),
        f"{prefix}_total_trades": int(summary.get("total_trades", 0)),
        f"{prefix}_positive_month_pct": round(float(monthly.get("positive_month_pct", 0.0)), 4),
        f"{prefix}_regime_resilience_pct": round(float(regime.get("resilience_score", 0.0)), 4),
    }


def _candidate_score(
    *,
    summary: dict,
    monthly_stats: dict,
    regime_analysis: dict,
    min_trades: int,
) -> float:
    trades = max(int(summary.get("total_trades", 0)), 0)
    trade_factor = min(trades / max(min_trades, 1), 1.0)
    if trade_factor <= 0:
        return 0.0

    score = (
        26 * _scale(summary.get("sharpe_ratio", 0), 0, 2.5) +
        18 * _scale(summary.get("calmar_ratio", 0), 0, 2.0) +
        18 * _inverse_scale(abs(summary.get("max_drawdown_pct", 0)), 5, 30) +
        16 * _scale(summary.get("total_return_pct", 0), 0, 35) +
        10 * _scale(summary.get("profit_factor", 0), 1.0, 2.5) +
        7 * _scale(monthly_stats.get("positive_month_pct", 0), 40, 75) +
        5 * _scale(regime_analysis.get("resilience_score", 0), 45, 80)
    )
    return round(float(np.clip(score * trade_factor, 0, 100)), 2)


def _scale(value: float, floor: float, ceiling: float) -> float:
    if ceiling <= floor:
        return 0.0
    return float(np.clip((value - floor) / (ceiling - floor), 0, 1))


def _inverse_scale(value: float, floor: float, ceiling: float) -> float:
    return 1 - _scale(value, floor, ceiling)


def _flag_label(enabled: bool) -> str:
    return "on" if enabled else "off"


def _format_ma_cross_params(params: dict) -> str:
    return (
        f"{params['ma_type'].upper()} {params['fast']}/{params['slow']}, "
        f"trend {_flag_label(params.get('trend_filter', False))}, "
        f"vol {_flag_label(params.get('volume_confirm', False))}"
    )


def _format_momentum_params(params: dict) -> str:
    return (
        f"{params['lookback']}/{params['skip']}, "
        f"RSI<{params['rsi_entry_cap']}, "
        f"trend {params['trend_ma']}"
    )


def _format_mean_reversion_params(params: dict) -> str:
    return (
        f"BB{params['bb_period']} z<{params['zscore_entry']}, "
        f"exit {params['zscore_exit']}, "
        f"RSI<{params['rsi_oversold']}"
    )


def _format_macd_params(params: dict) -> str:
    return (
        f"{params['macd_fast']}/{params['macd_slow']}/{params['macd_signal']}, "
        f"RSI {params['rsi_low']}-{params['rsi_high']}, "
        f"trend {_flag_label(params.get('trend_filter', False))}"
    )


def _format_donchian_params(params: dict) -> str:
    return (
        f"{params['entry_period']}/{params['exit_period']}, "
        f"vol {_flag_label(params.get('vol_confirm', False))}"
    )
