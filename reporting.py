"""
reporting.py — Strategy scorecards, robustness ranking, and narrative exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_monthly_stats(equity_curve: pd.Series) -> dict:
    """Summarize monthly consistency from an equity curve."""
    daily_ret = equity_curve.pct_change().dropna()
    if daily_ret.empty:
        return {
            "positive_month_pct": 0.0,
            "median_monthly_return_pct": 0.0,
            "best_month_pct": 0.0,
            "worst_month_pct": 0.0,
            "monthly_observations": 0,
        }

    monthly = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    return {
        "positive_month_pct": round(float((monthly > 0).mean() * 100), 2),
        "median_monthly_return_pct": round(float(monthly.median()), 2),
        "best_month_pct": round(float(monthly.max()), 2),
        "worst_month_pct": round(float(monthly.min()), 2),
        "monthly_observations": int(len(monthly)),
    }


def summarize_walk_forward(wf_df: pd.DataFrame) -> dict:
    """Convert fold-level walk-forward output into ranking features."""
    if wf_df is None or wf_df.empty:
        return {
            "wf_folds": 0,
            "wf_positive_pct": 0.0,
            "wf_avg_return_pct": 0.0,
            "wf_avg_sharpe": 0.0,
            "wf_avg_profit_factor": 0.0,
        }

    return {
        "wf_folds": int(len(wf_df)),
        "wf_positive_pct": round(float((wf_df["total_return_pct"] > 0).mean() * 100), 2),
        "wf_avg_return_pct": round(float(wf_df["total_return_pct"].mean()), 2),
        "wf_avg_sharpe": round(float(wf_df["sharpe"].mean()), 3),
        "wf_avg_profit_factor": round(float(wf_df["profit_factor"].mean()), 3),
    }


class RegimeLens:
    """
    Lightweight market-regime classifier used to evaluate strategy resilience.
    """

    def __init__(
        self,
        trend_window: int = 60,
        vol_window: int = 20,
        anchor_window: int = 126,
        trend_threshold: float = 0.04,
    ):
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.anchor_window = anchor_window
        self.trend_threshold = trend_threshold

    def classify(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend = close.pct_change(self.trend_window)
        daily_ret = close.pct_change()
        vol = daily_ret.rolling(self.vol_window).std() * np.sqrt(252)
        baseline_vol = vol.rolling(self.anchor_window, min_periods=self.vol_window).median()
        baseline_vol = baseline_vol.fillna(vol.expanding(min_periods=1).median())

        bull_quiet = (trend >= self.trend_threshold) & (vol <= baseline_vol)
        bull_volatile = (trend >= self.trend_threshold) & (vol > baseline_vol)
        bear_quiet = (trend <= -self.trend_threshold) & (vol <= baseline_vol)
        bear_volatile = (trend <= -self.trend_threshold) & (vol > baseline_vol)

        labels = np.select(
            [bull_quiet, bull_volatile, bear_quiet, bear_volatile],
            ["Bull Calm", "Bull Volatile", "Bear Calm", "Bear Volatile"],
            default="Range Bound",
        )
        regimes = pd.Series(labels, index=df.index, name="regime")
        warmup = max(self.trend_window, self.vol_window)
        if len(regimes) > warmup:
            regimes.iloc[:warmup] = "Warmup"
        return regimes

    def evaluate(self, equity_curve: pd.Series, regimes: pd.Series) -> dict:
        strat_ret = equity_curve.pct_change().rename("strategy_ret")
        joined = pd.concat([strat_ret, regimes.rename("regime")], axis=1, join="inner").dropna()
        joined = joined[joined["regime"] != "Warmup"]
        if joined.empty:
            return {
                "resilience_score": 0.0,
                "dominant_regime": "N/A",
                "positive_regime_pct": 0.0,
                "regime_details": [],
            }

        grouped = joined.groupby("regime")
        stats = grouped["strategy_ret"].agg(["mean", "std", "count"]).rename(
            columns={"mean": "avg_daily_return", "std": "daily_vol", "count": "bars"}
        )
        stats["hit_rate_pct"] = grouped.apply(lambda x: (x["strategy_ret"] > 0).mean() * 100)
        stats["avg_daily_return_pct"] = stats["avg_daily_return"] * 100
        stats["annualized_vol_pct"] = stats["daily_vol"].fillna(0) * np.sqrt(252) * 100
        stats["share_pct"] = stats["bars"] / stats["bars"].sum() * 100

        weights = stats["bars"] / stats["bars"].sum()
        positive_regime_pct = float(np.average((stats["avg_daily_return"] > 0).astype(float), weights=weights) * 100)
        edge_score = float(np.average(np.clip(stats["avg_daily_return_pct"] * 35, -20, 20), weights=weights))
        coverage_score = min(len(stats) / 5, 1.0) * 100
        resilience_score = np.clip(
            0.45 * positive_regime_pct + 0.35 * (50 + edge_score) + 0.20 * coverage_score,
            0,
            100,
        )

        stats = stats.reset_index()
        stats = stats[[
            "regime", "bars", "share_pct", "avg_daily_return_pct",
            "annualized_vol_pct", "hit_rate_pct"
        ]].sort_values("share_pct", ascending=False)

        for col in ["share_pct", "avg_daily_return_pct", "annualized_vol_pct", "hit_rate_pct"]:
            stats[col] = stats[col].round(3)

        return {
            "resilience_score": round(float(resilience_score), 2),
            "dominant_regime": str(stats.iloc[0]["regime"]) if not stats.empty else "N/A",
            "positive_regime_pct": round(positive_regime_pct, 2),
            "regime_details": stats.to_dict("records"),
        }


class StrategyScorecard:
    """Rank strategies using a multi-factor resilience score."""

    @staticmethod
    def _scale(value: float, floor: float, ceiling: float) -> float:
        if ceiling <= floor:
            return 0.0
        return float(np.clip((value - floor) / (ceiling - floor), 0, 1))

    @staticmethod
    def _inverse_scale(value: float, floor: float, ceiling: float) -> float:
        return 1 - StrategyScorecard._scale(value, floor, ceiling)

    def _score_components(
        self,
        summary: dict,
        monthly: dict,
        walk_forward: dict,
        regime: dict,
    ) -> dict:
        return {
            "score_sharpe": round(18 * self._scale(summary.get("sharpe_ratio", 0), 0, 2.5), 2),
            "score_calmar": round(14 * self._scale(summary.get("calmar_ratio", 0), 0, 2.0), 2),
            "score_drawdown": round(16 * self._inverse_scale(abs(summary.get("max_drawdown_pct", 0)), 5, 30), 2),
            "score_profit_factor": round(10 * self._scale(summary.get("profit_factor", 0), 1.0, 2.5), 2),
            "score_total_return": round(10 * self._scale(summary.get("total_return_pct", 0), 0, 40), 2),
            "score_monthly_consistency": round(12 * self._scale(monthly.get("positive_month_pct", 0), 40, 75), 2),
            "score_walk_forward": round(12 * self._scale(walk_forward.get("wf_positive_pct", 0), 35, 80), 2),
            "score_regime_resilience": round(8 * self._scale(regime.get("resilience_score", 0), 45, 80), 2),
        }

    def build(self, results: list[dict]) -> pd.DataFrame:
        rows = []
        for item in results:
            summary = item.get("summary", {})
            monthly = item.get("monthly_stats", {})
            walk_forward = item.get("walk_forward", {})
            regime = item.get("regime_analysis", {})
            components = self._score_components(summary, monthly, walk_forward, regime)
            resilience_score = round(sum(components.values()), 2)

            rows.append({
                "strategy": item["name"],
                "resilience_score": resilience_score,
                "total_return_pct": round(summary.get("total_return_pct", 0), 2),
                "cagr_pct": round(summary.get("cagr_pct", 0), 2),
                "sharpe_ratio": round(summary.get("sharpe_ratio", 0), 3),
                "calmar_ratio": round(summary.get("calmar_ratio", 0), 3),
                "max_drawdown_pct": round(summary.get("max_drawdown_pct", 0), 2),
                "profit_factor": round(summary.get("profit_factor", 0), 3),
                "win_rate_pct": round(summary.get("win_rate_pct", 0), 2),
                "total_trades": int(summary.get("total_trades", 0)),
                "positive_month_pct": round(monthly.get("positive_month_pct", 0), 2),
                "wf_positive_pct": round(walk_forward.get("wf_positive_pct", 0), 2),
                "wf_avg_sharpe": round(walk_forward.get("wf_avg_sharpe", 0), 3),
                "regime_resilience_pct": round(regime.get("resilience_score", 0), 2),
                "dominant_regime": regime.get("dominant_regime", "N/A"),
                **components,
            })

        scorecard = pd.DataFrame(rows)
        if scorecard.empty:
            return scorecard

        scorecard = scorecard.sort_values(
            by=["resilience_score", "sharpe_ratio", "total_return_pct"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        scorecard.insert(0, "rank", np.arange(1, len(scorecard) + 1))
        return scorecard


def build_regime_frame(results: list[dict]) -> pd.DataFrame:
    rows = []
    for item in results:
        for detail in item.get("regime_analysis", {}).get("regime_details", []):
            rows.append({"strategy": item["name"], **detail})
    return pd.DataFrame(rows)


def export_research_report(
    output_dir: str | Path,
    ranking_df: pd.DataFrame,
    all_results: list[dict],
    dataset_info: dict,
) -> dict:
    """Write markdown + JSON reports for the whole strategy run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_name = ranking_df.iloc[0]["strategy"] if not ranking_df.empty else None
    best_result = next((item for item in all_results if item["name"] == best_name), None)

    payload = {
        "dataset": dataset_info,
        "leaderboard": ranking_df.to_dict("records"),
        "best_strategy": _build_best_payload(best_result) if best_result else None,
    }

    json_path = output_path / "research_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

    markdown_path = output_path / "research_report.md"
    markdown_path.write_text(_build_markdown_report(dataset_info, ranking_df, best_result), encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }


def _build_markdown_report(
    dataset_info: dict,
    ranking_df: pd.DataFrame,
    best_result: dict | None,
) -> str:
    lines = [
        "# Quant Strategy Research Report",
        "",
        "## Dataset Snapshot",
        f"- Source: {dataset_info.get('source', 'unknown')}",
        f"- Bars: {dataset_info.get('bars', 0)}",
        f"- Range: {dataset_info.get('start', 'n/a')} to {dataset_info.get('end', 'n/a')}",
        f"- Selection mode: {dataset_info.get('selection_metric', 'resilience_score')}",
        "",
        "## Resilience Leaderboard",
        _markdown_table(
            ranking_df,
            ["rank", "strategy", "resilience_score", "sharpe_ratio", "max_drawdown_pct", "wf_positive_pct"],
        ) if not ranking_df.empty else "No strategies ranked.",
        "",
    ]

    if best_result:
        summary = best_result.get("summary", {})
        monthly = best_result.get("monthly_stats", {})
        walk_forward = best_result.get("walk_forward", {})
        regime = best_result.get("regime_analysis", {})
        monte_carlo = best_result.get("monte_carlo", {})

        lines.extend([
            "## Best Strategy",
            f"**{best_result['name']}** led the pack on the composite resilience score.",
            "",
            f"- Total return: {summary.get('total_return_pct', 0):.2f}%",
            f"- Sharpe ratio: {summary.get('sharpe_ratio', 0):.3f}",
            f"- Max drawdown: {summary.get('max_drawdown_pct', 0):.2f}%",
            f"- Positive months: {monthly.get('positive_month_pct', 0):.2f}%",
            f"- Positive walk-forward folds: {walk_forward.get('wf_positive_pct', 0):.2f}%",
            f"- Regime resilience: {regime.get('resilience_score', 0):.2f}%",
        ])

        if monte_carlo:
            lines.extend([
                f"- Monte Carlo profit probability: {monte_carlo.get('prob_profit', 0):.2f}%",
                f"- Monte Carlo median final equity: ₹{monte_carlo.get('median_final', 0):,.0f}",
            ])

        regime_details = best_result.get("regime_analysis", {}).get("regime_details", [])
        if regime_details:
            lines.extend([
                "",
                "## Regime Breakdown",
                _markdown_table(pd.DataFrame(regime_details), [
                    "regime", "share_pct", "avg_daily_return_pct", "annualized_vol_pct", "hit_rate_pct"
                ]),
            ])

    lines.append("")
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table = df.loc[:, columns].copy()
    headers = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(table.iloc[i][col]) for col in columns) + " |" for i in range(len(table))]
    return "\n".join([headers, divider, *rows])


def _json_default(value):
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    return str(value)


def _make_serializable(value):
    if value is None:
        return None
    return json.loads(json.dumps(value, default=_json_default))


def _build_best_payload(best_result: dict) -> dict:
    monte_carlo = best_result.get("monte_carlo", {}).copy()
    for key in ["all_final_values", "all_max_dds", "sample_paths", "percentile_paths"]:
        monte_carlo.pop(key, None)

    payload = {
        "name": best_result.get("name"),
        "summary": best_result.get("summary", {}),
        "monthly_stats": best_result.get("monthly_stats", {}),
        "walk_forward": best_result.get("walk_forward", {}),
        "regime_analysis": best_result.get("regime_analysis", {}),
        "resilience_score": best_result.get("resilience_score", 0),
        "monte_carlo": monte_carlo,
    }
    return _make_serializable(payload)
