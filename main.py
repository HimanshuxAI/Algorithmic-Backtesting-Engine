"""
main.py — Full backtesting pipeline
Run: python main.py

Runs all 5 strategies, walk-forward validation, Monte Carlo,
outputs performance report + trade log CSV + charts.
"""

import sys
import warnings
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from time import perf_counter

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from data        import generate_synthetic, load_csv, load_yfinance
from strategies  import (
    MACrossStrategy, MomentumStrategy, MeanReversionStrategy,
    MACDRSIStrategy, DonchianBreakoutStrategy
)
from execution   import CostConfig, RiskConfig
from analytics   import WalkForwardValidator, MonteCarloAnalysis
from charts      import (
    plot_dashboard, plot_monte_carlo, plot_walk_forward, plot_strategy_comparison,
    plot_strategy_lab,
)
from reporting   import (
    RegimeLens, StrategyScorecard, build_regime_frame, export_research_report
)
from backend     import (
    PipelineConfig, build_runtime_cost_config, build_runtime_risk_config,
    evaluate_strategy, setup_run_logger, utc_now_iso, write_run_manifest
)
from optimizer   import run_strategy_lab


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR      = Path("./output")
INITIAL_CAPITAL = 1_000_000.0    # ₹10 Lakh
RISK_FREE_RATE  = 0.065          # India 10yr yield

COST_CFG = CostConfig(
    brokerage_flat  = 20.0,
    brokerage_pct   = 0.0003,
    stt_sell        = 0.001,
    slippage_eta    = 0.1,
    slippage_alpha  = 0.5,
    slippage_cap    = 0.005,
    spread_bps      = 5.0,
)

RISK_CFG = RiskConfig(
    sizing_method    = "kelly_fractional",
    kelly_fraction   = 0.25,
    max_position_pct = 0.08,
    initial_capital  = INITIAL_CAPITAL,
    stop_loss_atr    = 2.0,
    max_open_trades  = 10,
)

# All strategies to run
STRATEGIES = {
    "MA Cross (EMA 10/30)":          MACrossStrategy(fast=10,  slow=30,  ma_type="ema",  trend_filter=False, volume_confirm=False),
    "MA Cross (EMA 20/50)":          MACrossStrategy(fast=20,  slow=50,  ma_type="ema",  trend_filter=False, volume_confirm=False),
    "Momentum (12-1)":               MomentumStrategy(lookback=126, skip=10, rsi_entry_cap=80),
    "Mean Reversion (BB+Z)":         MeanReversionStrategy(bb_period=20, zscore_entry=-1.8, rsi_oversold=40),
    "MACD+RSI":                      MACDRSIStrategy(rsi_low=30, trend_filter=False),
    "Donchian Breakout (20/10)":     DonchianBreakoutStrategy(entry_period=20, exit_period=10, vol_confirm=False),
}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _load_data(
    source: str = "synthetic",
    filepath: str = None,
    symbol: str = "RELIANCE.NS",
    start: str = "2021-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    if source == "synthetic":
        return generate_synthetic(
            n_bars=756, start_price=1000.0,
            annual_return=0.28, annual_vol=0.20,
            regime_switches=True, seed=99
        )
    if source == "csv" and filepath:
        return load_csv(filepath)
    if source == "yfinance":
        return load_yfinance(symbol=symbol, start=start, end=end)
    raise ValueError(f"Unknown source: {source}. Use 'synthetic', 'csv', or 'yfinance'.")


def _print_summary_table(all_results: list[dict]):
    header = (
        f"\n{'─'*115}\n"
        f"{'Strategy':<28} │ {'Return%':>8} │ {'CAGR%':>7} │ "
        f"{'Sharpe':>7} │ {'Sortino':>7} │ {'MaxDD%':>7} │ "
        f"{'WinRate%':>8} │ {'PF':>5} │ {'Trades':>7} │ {'Expectancy':>10}\n"
        f"{'─'*115}"
    )
    print(header)
    for r in all_results:
        s = r["summary"]
        print(
            f"{r['name']:<28} │ "
            f"{s.get('total_return_pct',0):>8.2f} │ "
            f"{s.get('cagr_pct',0):>7.2f} │ "
            f"{s.get('sharpe_ratio',0):>7.3f} │ "
            f"{s.get('sortino_ratio',0):>7.3f} │ "
            f"{s.get('max_drawdown_pct',0):>7.2f} │ "
            f"{s.get('win_rate_pct',0):>8.2f} │ "
            f"{s.get('profit_factor',0):>5.2f} │ "
            f"{s.get('total_trades',0):>7d} │ "
            f"₹{s.get('expectancy_inr',0):>9.0f}"
        )
    print(f"{'─'*115}\n")


def _print_resilience_table(scorecard_df: pd.DataFrame):
    if scorecard_df.empty:
        return

    header = (
        f"\n{'─'*120}\n"
        f"{'Rank':>4} │ {'Strategy':<28} │ {'Resilience':>10} │ {'Sharpe':>7} │ "
        f"{'MaxDD%':>7} │ {'WF+%':>6} │ {'Months+%':>9} │ {'Regime%':>8}\n"
        f"{'─'*120}"
    )
    print(header)
    for _, row in scorecard_df.iterrows():
        print(
            f"{int(row['rank']):>4d} │ "
            f"{row['strategy']:<28} │ "
            f"{row['resilience_score']:>10.2f} │ "
            f"{row['sharpe_ratio']:>7.3f} │ "
            f"{row['max_drawdown_pct']:>7.2f} │ "
            f"{row['wf_positive_pct']:>6.1f} │ "
            f"{row['positive_month_pct']:>9.1f} │ "
            f"{row['regime_resilience_pct']:>8.1f}"
        )
    print(f"{'─'*120}\n")


def _safe_filename(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("[", "")
        .replace("]", "")
        .replace(",", "")
        .replace("<", "lt")
        .replace(">", "gt")
        .replace(":", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(
    data_source: str = "synthetic",
    filepath: str = None,
    symbol: str = "RELIANCE.NS",
    start: str = "2021-01-01",
    end: str = "2024-01-01",
    output_dir: str | Path = OUTPUT_DIR,
    initial_capital: float = INITIAL_CAPITAL,
    sizing_method: str = "kelly_fractional",
    risk_free_rate: float = RISK_FREE_RATE,
    strict: bool = False,
    optimize: bool = False,
    train_ratio: float = 0.70,
):
    config = PipelineConfig(
        data_source=data_source,
        filepath=filepath,
        symbol=symbol,
        start=start,
        end=end,
        output_dir=output_dir,
        initial_capital=initial_capital,
        sizing_method=sizing_method,
        risk_free_rate=risk_free_rate,
        strict=strict,
        optimize=optimize,
        train_ratio=train_ratio,
    ).validate()

    runtime_cost_cfg = build_runtime_cost_config(COST_CFG)
    runtime_risk_cfg = build_runtime_risk_config(RISK_CFG, config)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    logger, log_path = setup_run_logger(config.output_dir, run_id)
    run_started_at = utc_now_iso()
    run_clock = perf_counter()
    outputs_written = [log_path.name]
    audits = []
    failures = []
    all_results = []
    dataset_info = {}
    strategy_lab_payload = None
    strategy_lab_split = {}
    fatal_error = None

    logger.info(
        "Pipeline started | run_id=%s source=%s symbol=%s output_dir=%s optimize=%s train_ratio=%.2f",
        run_id, config.data_source, config.symbol, config.output_dir, config.optimize, config.train_ratio
    )

    try:
        if config.optimize:
            load_step = "[1/7]"
            strategy_step = "[3/7]"
            comparison_step = "[4/7]"
            leaderboard_step = "[5/7]"
            charts_step = "[6/7]"
            export_step = "[7/7]"
        else:
            load_step = "[1/6]"
            strategy_step = "[2/6]"
            comparison_step = "[3/6]"
            leaderboard_step = "[4/6]"
            charts_step = "[5/6]"
            export_step = "[6/6]"

        print("\n" + "═" * 60)
        print("  ALGORITHMIC BACKTESTING ENGINE")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("═" * 60)

        print(f"\n{load_step} Loading data (source={config.data_source}) ...")
        df = _load_data(
            source=config.data_source,
            filepath=config.filepath,
            symbol=config.symbol,
            start=config.start,
            end=config.end,
        )
        print(f"      Bars: {len(df)} | "
              f"From: {df.index[0].date()} | "
              f"To:   {df.index[-1].date()}")

        dataset_info = {
            "source": config.data_source,
            "symbol": config.symbol if config.data_source == "yfinance" else "synthetic_or_csv",
            "bars": len(df),
            "start": str(df.index[0].date()),
            "end": str(df.index[-1].date()),
            "selection_metric": "resilience_score",
            "research_mode": "preset_suite",
        }

        benchmark = df["close"].copy()
        regime_lens = RegimeLens()
        regimes = regime_lens.classify(df)
        wf_validator = WalkForwardValidator(
            train_size=252,
            test_size=63,
            step_size=21,
            expanding=True,
            warmup_bars=126,
            verbose=False,
        )

        strategies_to_run = [(name, strategy, None) for name, strategy in STRATEGIES.items()]

        if config.optimize:
            print(
                f"\n[2/7] Running strategy lab "
                f"(train={config.train_ratio:.0%}, test={1 - config.train_ratio:.0%}) ..."
            )
            lab = run_strategy_lab(
                df,
                base_cost_config=runtime_cost_cfg,
                base_risk_config=runtime_risk_cfg,
                risk_free_rate=config.risk_free_rate,
                train_ratio=config.train_ratio,
            )
            strategy_lab_split = lab["split"]
            dataset_info.update({
                "selection_metric": "strategy_lab_train_score",
                "research_mode": "train_test_optimization",
                **strategy_lab_split,
            })
            lab_candidate_df = lab["candidate_df"].drop(columns=["params"], errors="ignore")
            lab_summary_df = lab["summary_df"].drop(columns=["params"], errors="ignore")
            top_candidates = []
            if not lab_candidate_df.empty:
                top_candidates = (
                    lab_candidate_df[lab_candidate_df["status"] == "success"]
                    .sort_values(["train_score", "train_sharpe_ratio"], ascending=[False, False])
                    .head(8)
                    .to_dict("records")
                )
            strategy_lab_payload = {
                "enabled": True,
                "split": strategy_lab_split,
                "summary": lab["summary_records"],
                "candidates": lab["candidate_records"],
                "top_candidates": top_candidates,
            }

            lab_candidate_df.to_csv(config.output_dir / "strategy_lab_candidates.csv", index=False)
            lab_summary_df.to_csv(config.output_dir / "strategy_lab_summary.csv", index=False)
            (config.output_dir / "strategy_lab.json").write_text(
                json.dumps(strategy_lab_payload, indent=2, default=str),
                encoding="utf-8",
            )
            outputs_written.extend([
                "strategy_lab_candidates.csv",
                "strategy_lab_summary.csv",
                "strategy_lab.json",
            ])

            if not lab_summary_df.empty:
                plot_strategy_lab(
                    lab_summary_df,
                    savepath=str(config.output_dir / "strategy_lab.png"),
                )
                outputs_written.append("strategy_lab.png")

            print(
                f"      Evaluated {len(lab['candidate_records'])} candidates across "
                f"{len(lab['selected_strategies'])} strategy families."
            )
            for selection in lab["summary_records"]:
                print(
                    f"      {selection['family']}: {selection['selected_param_label']} | "
                    f"train {selection['train_score']:.2f} | test {selection.get('test_score', 0):.2f}"
                )

            strategies_to_run = [
                (item["name"], item["strategy"], item)
                for item in lab["selected_strategies"]
            ]

        print(f"\n{strategy_step} Running {len(strategies_to_run)} strategies with robustness checks ...")
        for name, strategy, optimization_meta in strategies_to_run:
            print(f"\n  ▶ {name}")
            result_payload, audit = evaluate_strategy(
                name=name,
                strategy=strategy,
                df=df,
                benchmark=benchmark,
                base_cost_config=runtime_cost_cfg,
                base_risk_config=runtime_risk_cfg,
                wf_validator=wf_validator,
                regime_lens=regime_lens,
                regimes=regimes,
                risk_free_rate=config.risk_free_rate,
                logger=logger,
            )
            audits.append(audit.to_dict())

            if result_payload is None:
                failures.append(audit.to_dict())
                print(f"    ✗ Failed: {audit.error_type}: {audit.error_message}")
                if config.strict:
                    raise RuntimeError(
                        f"Strict mode enabled. Aborting after strategy failure: {name}."
                    )
                continue

            print(
                f"    Return: {result_payload['summary']['total_return_pct']:+.2f}% | "
                f"Sharpe: {result_payload['summary']['sharpe_ratio']:.3f} | "
                f"MaxDD: {result_payload['summary']['max_drawdown_pct']:.2f}% | "
                f"WF+: {result_payload['walk_forward']['wf_positive_pct']:.1f}% | "
                f"Regime: {result_payload['regime_analysis']['resilience_score']:.1f}%"
            )
            result_payload["backend_duration_seconds"] = audit.duration_seconds
            if optimization_meta:
                result_payload["optimization"] = {
                    "family": optimization_meta["family"],
                    "params": optimization_meta["params"],
                    "param_label": optimization_meta["param_label"],
                    "train_score": optimization_meta["train_score"],
                    "test_score": optimization_meta["test_score"],
                    "candidate_count": optimization_meta["candidate_count"],
                    "test_status": optimization_meta["test_status"],
                    "test_error_type": optimization_meta["test_error_type"],
                    "test_error_message": optimization_meta["test_error_message"],
                    "test_metrics": optimization_meta["test_metrics"],
                    "split": strategy_lab_split,
                }
            all_results.append(result_payload)

        print(f"\n{comparison_step} Performance comparison:")
        _print_summary_table(all_results)

        if not all_results:
            raise RuntimeError("No strategies completed successfully.")

        scorecard = StrategyScorecard().build(all_results)
        print(f"{leaderboard_step} Resilience leaderboard:")
        _print_resilience_table(scorecard)

        best_name = scorecard.iloc[0]["strategy"]
        best = next(item for item in all_results if item["name"] == best_name)
        best["resilience_score"] = float(scorecard.iloc[0]["resilience_score"])
        print(f"  ★ Best by resilience score: {best['name']} ({best['resilience_score']:.2f})")

        print(f"\n{charts_step} Generating charts ...")
        plot_strategy_comparison(scorecard, savepath=str(config.output_dir / "strategy_comparison.png"))
        outputs_written.append("strategy_comparison.png")
        plot_dashboard(
            equity_curve=best["result"].equity_curve["equity"],
            trades_df=best["result"].trade_df,
            benchmark=benchmark,
            summary=best["summary"],
            strategy_name=best["name"],
            savepath=str(config.output_dir / "dashboard.png"),
        )
        outputs_written.append("dashboard.png")

        best_wf_df = best["walk_forward_df"]
        if not best_wf_df.empty:
            plot_walk_forward(best_wf_df, savepath=str(config.output_dir / "walk_forward.png"))
            best_wf_df.to_csv(config.output_dir / "walk_forward.csv", index=False)
            outputs_written.extend(["walk_forward.png", "walk_forward.csv"])

        if not best["result"].trade_df.empty and "return_pct" in best["result"].trade_df.columns:
            mc = MonteCarloAnalysis(n_simulations=1000, seed=42)
            best["monte_carlo"] = mc.run(
                trade_returns=best["result"].trade_df["return_pct"],
                initial_capital=config.initial_capital,
                n_trades=len(best["result"].trade_df),
            )
            plot_monte_carlo(
                best["monte_carlo"],
                initial_capital=config.initial_capital,
                savepath=str(config.output_dir / "monte_carlo.png"),
            )
            outputs_written.append("monte_carlo.png")

        print(f"\n{export_step} Exporting results ...")

        for r in all_results:
            safe_name = _safe_filename(r["name"])
            td = r["result"].trade_df
            if not td.empty:
                td.to_csv(config.output_dir / f"trades_{safe_name}.csv", index=False)
                filename = f"trades_{safe_name}.csv"
                outputs_written.append(filename)
                print(f"  → {filename} ({len(td)} trades)")

        eq_all = pd.DataFrame({
            r["name"]: r["result"].equity_curve["equity"]
            for r in all_results
        })
        eq_all.to_csv(config.output_dir / "equity_curves.csv")
        outputs_written.append("equity_curves.csv")
        print("  → equity_curves.csv")

        summary_rows = []
        for r in all_results:
            row = {"strategy": r["name"]}
            row.update(r["summary"])
            row.update(r["monthly_stats"])
            row.update(r["walk_forward"])
            row.update({
                "regime_resilience_pct": r["regime_analysis"].get("resilience_score", 0),
                "dominant_regime": r["regime_analysis"].get("dominant_regime", "N/A"),
                "backend_duration_seconds": r.get("backend_duration_seconds", 0),
            })
            if r.get("optimization"):
                row.update({
                    "strategy_family": r["optimization"].get("family"),
                    "optimized_param_label": r["optimization"].get("param_label"),
                    "strategy_lab_train_score": r["optimization"].get("train_score", 0),
                    "strategy_lab_test_score": r["optimization"].get("test_score", 0),
                })
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(config.output_dir / "performance_summary.csv", index=False)
        outputs_written.append("performance_summary.csv")
        print("  → performance_summary.csv")

        scorecard.to_csv(config.output_dir / "strategy_rankings.csv", index=False)
        outputs_written.append("strategy_rankings.csv")
        print("  → strategy_rankings.csv")

        regime_df = build_regime_frame(all_results)
        if not regime_df.empty:
            regime_df.to_csv(config.output_dir / "regime_summary.csv", index=False)
            outputs_written.append("regime_summary.csv")
            print("  → regime_summary.csv")

        pd.DataFrame(audits).to_csv(config.output_dir / "strategy_health.csv", index=False)
        outputs_written.append("strategy_health.csv")
        print("  → strategy_health.csv")

        report_paths = export_research_report(
            output_dir=config.output_dir,
            ranking_df=scorecard,
            all_results=all_results,
            dataset_info=dataset_info,
            optimization_report=strategy_lab_payload,
        )
        outputs_written.extend([Path(report_paths["markdown"]).name, Path(report_paths["json"]).name])
        print(f"  → {Path(report_paths['markdown']).name}")
        print(f"  → {Path(report_paths['json']).name}")

        _print_final_report(best, df, config.output_dir, config.initial_capital, len(failures))
        logger.info(
            "Pipeline completed | run_id=%s successful_strategies=%s failed_strategies=%s",
            run_id, len(all_results), len(failures)
        )
        return all_results

    except Exception as exc:
        fatal_error = {"type": type(exc).__name__, "message": str(exc)}
        logger.error("Pipeline failed | run_id=%s error=%s", run_id, exc)
        logger.error("%s", exc, exc_info=True)
        raise

    finally:
        manifest_status = "failed"
        if all_results:
            manifest_status = "partial_success" if failures else "success"
        manifest = {
            "run_id": run_id,
            "status": manifest_status,
            "started_at": run_started_at,
            "finished_at": utc_now_iso(),
            "duration_seconds": round(perf_counter() - run_clock, 4),
            "config": config.to_dict(),
            "dataset": dataset_info,
            "outputs": sorted(set(outputs_written + ["run_manifest.json"])),
            "successful_strategies": [item["name"] for item in all_results],
            "failed_strategies": failures,
            "strategy_audit": audits,
        }
        if fatal_error:
            manifest["fatal_error"] = fatal_error
        manifest_path = write_run_manifest(config.output_dir, manifest)
        logger.info("Run manifest written | path=%s", manifest_path)


def _print_final_report(
    best: dict,
    df: pd.DataFrame,
    output_dir: Path,
    initial_capital: float,
    failed_strategies: int = 0,
):
    s = best["summary"]
    n = best["name"]
    trades = best["result"].trade_df
    monthly = best.get("monthly_stats", {})
    walk_forward = best.get("walk_forward", {})
    regime = best.get("regime_analysis", {})
    monte_carlo = best.get("monte_carlo", {})
    optimization = best.get("optimization", {})

    print("\n" + "═" * 60)
    print(f"  FINAL REPORT — {n}")
    print("═" * 60)
    print(f"\n  Capital:           ₹{initial_capital/1e5:.1f} Lakh")
    print(f"  Period:            {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Data bars:         {len(df)}")
    print(f"  Resilience Score:  {best.get('resilience_score', 0):.2f}")
    print(f"  Failed Strategies: {failed_strategies}")
    print(f"\n  ── RETURNS ──")
    print(f"  Total Return:      {s['total_return_pct']:+.2f}%")
    print(f"  CAGR:              {s['cagr_pct']:+.2f}%")
    print(f"  Annual Vol:        {s['annualized_vol_pct']:.2f}%")
    print(f"\n  ── RISK-ADJUSTED ──")
    print(f"  Sharpe Ratio:      {s['sharpe_ratio']:.4f}")
    print(f"  Sortino Ratio:     {s['sortino_ratio']:.4f}")
    print(f"  Calmar Ratio:      {s['calmar_ratio']:.4f}")
    print(f"  Omega Ratio:       {s['omega_ratio']:.4f}")
    print(f"\n  ── DRAWDOWN ──")
    print(f"  Max Drawdown:      {s['max_drawdown_pct']:.2f}%")
    print(f"  Max DD Duration:   {s['max_dd_duration_days']} days")
    print(f"  # Drawdown Periods:{s['num_drawdowns']}")
    print(f"\n  ── TRADES ──")
    print(f"  Total Trades:      {s['total_trades']}")
    print(f"  Win Rate:          {s['win_rate_pct']:.2f}%")
    print(f"  Profit Factor:     {s['profit_factor']:.4f}")
    print(f"  Avg Win/Loss:      {s['avg_win_loss_ratio']:.4f}")
    print(f"  Avg Hold Days:     {s['avg_hold_days']:.1f}")
    print(f"  Expectancy:        ₹{s['expectancy_inr']:,.0f} per trade")
    print(f"  Avg R-Multiple:    {s['avg_r_multiple']:.4f}")
    print(f"  Total TXN Costs:   ₹{s['total_cost_inr']:,.0f}")
    print(f"\n  ── ROBUSTNESS ──")
    print(f"  Positive Months:   {monthly.get('positive_month_pct', 0):.2f}%")
    print(f"  WF Positive Folds: {walk_forward.get('wf_positive_pct', 0):.2f}%")
    print(f"  WF Avg Sharpe:     {walk_forward.get('wf_avg_sharpe', 0):.3f}")
    print(f"  Regime Resilience: {regime.get('resilience_score', 0):.2f}%")
    print(f"  Dominant Regime:   {regime.get('dominant_regime', 'N/A')}")
    if optimization:
        print(f"\n  ── STRATEGY LAB ──")
        print(f"  Family:            {optimization.get('family', 'N/A')}")
        print(f"  Params:            {optimization.get('param_label', 'N/A')}")
        print(f"  Train Score:       {optimization.get('train_score', 0):.2f}")
        print(f"  Test Score:        {optimization.get('test_score', 0):.2f}")
    print(f"\n  ── BENCHMARK ──")
    print(f"  Alpha (ann.):      {s['alpha_pct']:+.2f}%")
    print(f"  Beta:              {s['beta']:.4f}")
    print(f"  Info Ratio:        {s['information_ratio']:.4f}")
    print(f"\n  ── TAIL RISK ──")
    print(f"  VaR (95%, 1d):     {s['var_95_pct']:.4f}%")
    print(f"  CVaR (95%, 1d):    {s['cvar_95_pct']:.4f}%")
    print(f"  Skewness:          {s['skewness']:.4f}")
    print(f"  Excess Kurtosis:   {s['excess_kurtosis']:.4f}")
    if monte_carlo:
        print(f"\n  ── MONTE CARLO ──")
        print(f"  P(Profit):         {monte_carlo.get('prob_profit', 0):.2f}%")
        print(f"  Median Final:      ₹{monte_carlo.get('median_final', 0):,.0f}")
        print(f"  5th Pct Final:     ₹{monte_carlo.get('p5_final', 0):,.0f}")
        print(f"  Median Max DD:     {monte_carlo.get('median_max_dd', 0):.2f}%")

    if not trades.empty:
        print(f"\n  ── LAST 5 TRADES ──")
        cols = ["trade_id","entry_date","exit_date","entry_fill","exit_fill",
                "quantity","net_pnl","return_pct","exit_reason"]
        print(trades[cols].tail(5).to_string(index=False))

    print(f"\n  Output saved to: {output_dir.resolve()}/")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Algorithmic Backtesting Engine")
    parser.add_argument("--source",   default="synthetic",
                        choices=["synthetic", "csv", "yfinance"],
                        help="Data source")
    parser.add_argument("--file",     default=None,
                        help="Path to CSV file (if source=csv)")
    parser.add_argument("--symbol",   default="RELIANCE.NS",
                        help="Yahoo Finance ticker (if source=yfinance)")
    parser.add_argument("--start",    default="2021-01-01")
    parser.add_argument("--end",      default="2024-01-01")
    parser.add_argument("--capital",  default=1_000_000, type=float)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--sizing",   default="kelly_fractional",
                        choices=["kelly_fractional","fixed_pct","atr_based","equal"])
    parser.add_argument("--strict", action="store_true",
                        help="Abort the pipeline on the first strategy failure.")
    parser.add_argument("--optimize", action="store_true",
                        help="Run the strategy lab and promote tuned configurations.")
    parser.add_argument("--train-ratio", default=0.70, type=float,
                        help="Training split ratio used when --optimize is enabled.")
    args = parser.parse_args()

    run_pipeline(
        data_source=args.source,
        filepath=args.file,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        initial_capital=args.capital,
        sizing_method=args.sizing,
        strict=args.strict,
        optimize=args.optimize,
        train_ratio=args.train_ratio,
    )
