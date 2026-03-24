"""
backend.py — Runtime orchestration helpers for a stronger pipeline backend.
"""

from __future__ import annotations

import copy
import json
import logging
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd

from analytics import PerformanceMetrics
from engine import BacktestEngine
from reporting import compute_monthly_stats, summarize_walk_forward


SUPPORTED_SOURCES = {"synthetic", "csv", "yfinance"}
SUPPORTED_SIZING_METHODS = {"kelly_fractional", "fixed_pct", "atr_based", "equal"}


@dataclass
class PipelineConfig:
    data_source: str = "synthetic"
    filepath: str | None = None
    symbol: str = "RELIANCE.NS"
    start: str = "2021-01-01"
    end: str = "2024-01-01"
    output_dir: str | Path = "./output"
    initial_capital: float = 1_000_000.0
    sizing_method: str = "kelly_fractional"
    risk_free_rate: float = 0.065
    strict: bool = False

    def validate(self) -> "PipelineConfig":
        if self.data_source not in SUPPORTED_SOURCES:
            raise ValueError(
                f"Unsupported data source '{self.data_source}'. Supported: {sorted(SUPPORTED_SOURCES)}"
            )
        if self.sizing_method not in SUPPORTED_SIZING_METHODS:
            raise ValueError(
                f"Unsupported sizing method '{self.sizing_method}'. Supported: {sorted(SUPPORTED_SIZING_METHODS)}"
            )
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        if self.risk_free_rate < -1:
            raise ValueError("Risk-free rate must be greater than -100%.")
        if self.data_source == "csv" and not self.filepath:
            raise ValueError("A CSV filepath is required when source='csv'.")
        if self.data_source == "yfinance" and not self.symbol:
            raise ValueError("A ticker symbol is required when source='yfinance'.")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("Start date must be earlier than end date.")

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass
class StrategyAuditRecord:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    trades: int = 0
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    resilience_score: float = 0.0
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_runtime_risk_config(base_risk_config, config: PipelineConfig):
    runtime_risk = copy.deepcopy(base_risk_config)
    runtime_risk.initial_capital = config.initial_capital
    runtime_risk.sizing_method = config.sizing_method
    return runtime_risk


def build_runtime_cost_config(base_cost_config):
    return copy.deepcopy(base_cost_config)


def setup_run_logger(output_dir: str | Path, run_id: str) -> tuple[logging.Logger, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "pipeline.log"

    logger = logging.getLogger(f"backtest_pipeline_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def evaluate_strategy(
    *,
    name: str,
    strategy,
    df: pd.DataFrame,
    benchmark: pd.Series,
    base_cost_config,
    base_risk_config,
    wf_validator,
    regime_lens,
    regimes: pd.Series,
    risk_free_rate: float,
    logger: logging.Logger | None = None,
) -> tuple[dict | None, StrategyAuditRecord]:
    started_at = utc_now_iso()
    started_clock = perf_counter()

    try:
        engine = BacktestEngine(
            strategy=strategy,
            cost_config=build_runtime_cost_config(base_cost_config),
            risk_config=copy.deepcopy(base_risk_config),
            symbol="ASSET",
            verbose=False,
        )
        result = engine.run(df.copy())
        equity = result.equity_curve["equity"]
        if len(equity) < 10:
            raise ValueError("Not enough equity observations to compute metrics.")

        metrics = PerformanceMetrics(
            equity,
            result.trade_df,
            benchmark=benchmark,
            rf_annual=risk_free_rate,
        )
        summary = metrics.summary()
        monthly_stats = compute_monthly_stats(equity)
        wf_df = wf_validator.run(
            df,
            strategy,
            cost_config=build_runtime_cost_config(base_cost_config),
            risk_config=copy.deepcopy(base_risk_config),
            verbose=False,
        )
        walk_forward = summarize_walk_forward(wf_df)
        regime_analysis = regime_lens.evaluate(equity, regimes)

        finished_at = utc_now_iso()
        duration_seconds = round(perf_counter() - started_clock, 4)
        payload = {
            "name": name,
            "strategy": strategy,
            "result": result,
            "metrics": metrics,
            "summary": summary,
            "monthly_stats": monthly_stats,
            "walk_forward": walk_forward,
            "walk_forward_df": wf_df,
            "regime_analysis": regime_analysis,
            "monte_carlo": {},
        }
        audit = StrategyAuditRecord(
            name=name,
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            trades=int(summary.get("total_trades", 0)),
            total_return_pct=float(summary.get("total_return_pct", 0)),
            sharpe_ratio=float(summary.get("sharpe_ratio", 0)),
            resilience_score=float(regime_analysis.get("resilience_score", 0)),
        )
        if logger:
            logger.info(
                "Strategy completed | name=%s duration=%.4fs trades=%s return_pct=%.2f sharpe=%.3f",
                name,
                duration_seconds,
                audit.trades,
                audit.total_return_pct,
                audit.sharpe_ratio,
            )
        return payload, audit

    except Exception as exc:
        finished_at = utc_now_iso()
        duration_seconds = round(perf_counter() - started_clock, 4)
        if logger:
            logger.error("Strategy failed | name=%s error=%s", name, exc)
            logger.error(traceback.format_exc())
        audit = StrategyAuditRecord(
            name=name,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return None, audit


def write_run_manifest(output_dir: str | Path, manifest: dict) -> Path:
    manifest_path = Path(output_dir) / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest_path


def _json_default(value):
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return str(value)
