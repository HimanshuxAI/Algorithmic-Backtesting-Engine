"""
analytics.py — Performance metrics, walk-forward validation, Monte Carlo
All metrics computed from equity curve + trade log. No external dependencies.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  METRICS ENGINE
# ─────────────────────────────────────────────
class PerformanceMetrics:
    """
    Compute the full suite of quant performance metrics from
    an equity curve and trade DataFrame.
    """

    def __init__(
        self,
        equity_curve: pd.Series,     # portfolio value over time
        trades_df:    pd.DataFrame,
        benchmark:    Optional[pd.Series] = None,
        rf_annual:    float = 0.065,  # India 10yr ~6.5%
        periods_per_year: int = 252,
    ):
        self.equity     = equity_curve.dropna()
        self.trades     = trades_df
        self.benchmark  = benchmark
        self.rf         = rf_annual
        self.ppy        = periods_per_year
        self.returns    = self.equity.pct_change().dropna()

    # ── RETURN METRICS ──────────────────────
    def total_return(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        return (self.equity.iloc[-1] / self.equity.iloc[0]) - 1

    def cagr(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        n_years = len(self.equity) / self.ppy
        if n_years <= 0: return 0.0
        return (1 + self.total_return()) ** (1 / n_years) - 1

    def annualized_vol(self) -> float:
        if self.returns.empty:
            return 0.0
        return self.returns.std() * np.sqrt(self.ppy)

    def sharpe_ratio(self) -> float:
        if self.returns.empty:
            return 0.0
        rf_daily    = (1 + self.rf) ** (1 / self.ppy) - 1
        excess_ret  = self.returns - rf_daily
        if excess_ret.std() == 0: return 0.0
        return (excess_ret.mean() / excess_ret.std()) * np.sqrt(self.ppy)

    def sortino_ratio(self) -> float:
        if self.returns.empty:
            return 0.0
        rf_daily   = (1 + self.rf) ** (1 / self.ppy) - 1
        excess_ret = self.returns - rf_daily
        downside   = excess_ret[excess_ret < 0].std()
        if downside == 0: return 0.0
        return (excess_ret.mean() / downside) * np.sqrt(self.ppy)

    def calmar_ratio(self) -> float:
        mdd = self.max_drawdown()
        if mdd == 0: return 0.0
        return self.cagr() / abs(mdd)

    def omega_ratio(self, threshold: float = 0.0) -> float:
        gains  = (self.returns[self.returns > threshold] - threshold).sum()
        losses = (threshold - self.returns[self.returns <= threshold]).sum()
        return gains / losses if losses > 0 else np.inf

    # ── DRAWDOWN METRICS ────────────────────
    def drawdown_series(self) -> pd.Series:
        if self.equity.empty:
            return pd.Series(dtype=float)
        roll_max = self.equity.cummax()
        return (self.equity - roll_max) / roll_max

    def max_drawdown(self) -> float:
        dd = self.drawdown_series()
        return float(dd.min()) if not dd.empty else 0.0

    def drawdown_duration(self) -> dict:
        dd = self.drawdown_series()
        if dd.empty:
            return {
                "max_dd_duration_days": 0,
                "avg_dd_duration_days": 0,
                "num_drawdowns": 0,
            }
        in_dd  = dd < 0
        prev_in_dd = in_dd.shift(1, fill_value=False)
        starts = in_dd & ~prev_in_dd
        ends   = ~in_dd & prev_in_dd

        durations = []
        start_dates = dd.index[starts].tolist()
        end_dates   = dd.index[ends].tolist()

        for s in start_dates:
            later_ends = [e for e in end_dates if e > s]
            end = later_ends[0] if later_ends else dd.index[-1]
            durations.append((end - s).days)

        return {
            "max_dd_duration_days": max(durations) if durations else 0,
            "avg_dd_duration_days": np.mean(durations) if durations else 0,
            "num_drawdowns":        len(durations),
        }

    # ── TRADE METRICS ───────────────────────
    def win_rate(self) -> float:
        if self.trades.empty: return 0.0
        return (self.trades["net_pnl"] > 0).mean()

    def profit_factor(self) -> float:
        if self.trades.empty: return 0.0
        wins   = self.trades.loc[self.trades["net_pnl"] > 0, "net_pnl"].sum()
        losses = self.trades.loc[self.trades["net_pnl"] <= 0, "net_pnl"].abs().sum()
        return wins / losses if losses > 0 else np.inf

    def avg_win_loss_ratio(self) -> float:
        if self.trades.empty: return 0.0
        avg_win  = self.trades.loc[self.trades["net_pnl"] > 0,  "net_pnl"].mean()
        avg_loss = self.trades.loc[self.trades["net_pnl"] <= 0, "net_pnl"].abs().mean()
        if pd.isna(avg_win) or pd.isna(avg_loss) or avg_loss == 0: return 0.0
        return avg_win / avg_loss

    def avg_hold_days(self) -> float:
        if self.trades.empty: return 0.0
        return self.trades["hold_days"].mean()

    def expectancy(self) -> float:
        """Expected ₹ return per trade."""
        if self.trades.empty: return 0.0
        return self.trades["net_pnl"].mean()

    def avg_r_multiple(self) -> float:
        if self.trades.empty or "r_multiple" not in self.trades.columns: return 0.0
        return self.trades["r_multiple"].mean()

    def total_slippage_cost(self) -> float:
        if self.trades.empty: return 0.0
        return self.trades["total_cost"].sum()

    # ── BENCHMARK METRICS ───────────────────
    def alpha_beta(self) -> tuple[float, float]:
        if self.benchmark is None: return 0.0, 0.0
        bm_ret = self.benchmark.pct_change().dropna()
        # Align
        joined = pd.concat([self.returns, bm_ret], axis=1, join="inner")
        joined.columns = ["strat", "bench"]
        joined.dropna(inplace=True)
        if len(joined) < 10: return 0.0, 0.0

        slope, intercept, r, p, se = stats.linregress(joined["bench"], joined["strat"])
        beta  = slope
        alpha = (intercept * self.ppy)
        return alpha, beta

    def information_ratio(self) -> float:
        if self.benchmark is None: return 0.0
        bm_ret = self.benchmark.pct_change().dropna()
        joined = pd.concat([self.returns, bm_ret], axis=1, join="inner").dropna()
        if joined.empty: return 0.0
        joined.columns = ["strat", "bench"]
        active = joined["strat"] - joined["bench"]
        if active.std() == 0: return 0.0
        return (active.mean() / active.std()) * np.sqrt(self.ppy)

    # ── TAIL RISK ───────────────────────────
    def var_cvar(self, confidence: float = 0.95) -> tuple[float, float]:
        if self.returns.empty:
            return 0.0, 0.0
        var   = np.percentile(self.returns, (1 - confidence) * 100)
        cvar  = self.returns[self.returns <= var].mean()
        return var, cvar

    def skewness(self) -> float:
        if self.returns.empty:
            return 0.0
        return float(self.returns.skew())

    def kurtosis(self) -> float:
        if self.returns.empty:
            return 0.0
        return float(self.returns.kurtosis())

    # ── FULL SUMMARY ────────────────────────
    def summary(self) -> dict:
        alpha, beta = self.alpha_beta()
        var, cvar   = self.var_cvar()
        dd_info     = self.drawdown_duration()

        return {
            # Returns
            "total_return_pct":    round(self.total_return() * 100, 4),
            "cagr_pct":            round(self.cagr() * 100, 4),
            "annualized_vol_pct":  round(self.annualized_vol() * 100, 4),
            # Risk-adjusted
            "sharpe_ratio":        round(self.sharpe_ratio(), 4),
            "sortino_ratio":       round(self.sortino_ratio(), 4),
            "calmar_ratio":        round(self.calmar_ratio(), 4),
            "omega_ratio":         round(self.omega_ratio(), 4),
            # Drawdown
            "max_drawdown_pct":    round(self.max_drawdown() * 100, 4),
            "max_dd_duration_days": dd_info["max_dd_duration_days"],
            "num_drawdowns":       dd_info["num_drawdowns"],
            # Trade stats
            "total_trades":        len(self.trades),
            "win_rate_pct":        round(self.win_rate() * 100, 4),
            "profit_factor":       round(self.profit_factor(), 4),
            "avg_win_loss_ratio":  round(self.avg_win_loss_ratio(), 4),
            "avg_hold_days":       round(self.avg_hold_days(), 2),
            "expectancy_inr":      round(self.expectancy(), 2),
            "avg_r_multiple":      round(self.avg_r_multiple(), 4),
            "total_cost_inr":      round(self.total_slippage_cost(), 2),
            # Benchmark
            "alpha_pct":           round(alpha * 100, 4),
            "beta":                round(beta, 4),
            "information_ratio":   round(self.information_ratio(), 4),
            # Tail risk
            "var_95_pct":          round(var * 100, 4),
            "cvar_95_pct":         round(cvar * 100, 4),
            "skewness":            round(self.skewness(), 4),
            "excess_kurtosis":     round(self.kurtosis(), 4),
        }


# ─────────────────────────────────────────────
#  WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────
class WalkForwardValidator:
    """
    Expanding-window walk-forward analysis.
    Train on in-sample, test on out-of-sample, step forward.
    Prevents overfitting by validating on unseen data each fold.
    """

    def __init__(
        self,
        train_size:  int = 252,   # bars in training window
        test_size:   int = 63,    # bars in test window (~1 quarter)
        step_size:   int = 21,    # bars to step forward each fold
        expanding:   bool = True, # True=expanding, False=rolling window
        warmup_bars: int = 126,
        verbose:     bool = True,
    ):
        self.train_size = train_size
        self.test_size  = test_size
        self.step_size  = step_size
        self.expanding  = expanding
        self.warmup_bars = warmup_bars
        self.verbose = verbose

    def get_folds(self, n: int) -> list[tuple]:
        folds = []
        start = 0
        while True:
            train_end = start + self.train_size
            test_end  = train_end + self.test_size
            if test_end > n:
                break
            train_start = 0 if self.expanding else start
            folds.append((train_start, train_end, train_end, test_end))
            start += self.step_size
        return folds

    def run(
        self,
        df:          pd.DataFrame,
        strategy:    "Strategy",
        cost_config  = None,
        risk_config  = None,
        verbose:     bool | None = None,
    ) -> pd.DataFrame:
        from engine import BacktestEngine

        folds  = self.get_folds(len(df))
        results = []
        should_log = self.verbose if verbose is None else verbose

        if should_log:
            print(f"[wf] Running {len(folds)} walk-forward folds "
                  f"(train={self.train_size}d | test={self.test_size}d) ...")

        for i, (tr_start, tr_end, ts_start, ts_end) in enumerate(folds):
            ctx_start = max(0, ts_start - self.warmup_bars)
            context_df = df.iloc[ctx_start:ts_end]
            test_df = df.iloc[ts_start:ts_end]
            if len(test_df) < 10:
                continue

            engine = BacktestEngine(
                strategy    = strategy,
                cost_config = cost_config,
                risk_config = risk_config,
            )
            result  = engine.run(context_df)
            eq_srs  = result.equity_curve["equity"].loc[test_df.index]
            trade_df = result.trade_df.copy()
            if not trade_df.empty:
                trade_df = trade_df[
                    pd.to_datetime(trade_df["entry_date"]) >= pd.Timestamp(test_df.index[0])
                ]

            if len(eq_srs) < 2:
                continue

            pm = PerformanceMetrics(eq_srs, trade_df)
            results.append({
                "fold":              i + 1,
                "test_start":        test_df.index[0].date(),
                "test_end":          test_df.index[-1].date(),
                "total_return_pct":  round(pm.total_return() * 100, 2),
                "sharpe":            round(pm.sharpe_ratio(), 3),
                "max_dd_pct":        round(pm.max_drawdown() * 100, 2),
                "win_rate_pct":      round(pm.win_rate() * 100, 2),
                "num_trades":        len(trade_df),
                "profit_factor":     round(pm.profit_factor(), 3),
            })

        wf_df = pd.DataFrame(results)
        if should_log and not wf_df.empty:
            print(f"\n{'='*60}")
            print("WALK-FORWARD SUMMARY")
            print(f"{'='*60}")
            print(f"  Folds:             {len(wf_df)}")
            print(f"  Avg Return/fold:   {wf_df['total_return_pct'].mean():.2f}%")
            print(f"  Positive folds:    {(wf_df['total_return_pct'] > 0).sum()}/{len(wf_df)}")
            print(f"  Avg Sharpe:        {wf_df['sharpe'].mean():.3f}")
            print(f"  Avg Max DD:        {wf_df['max_dd_pct'].mean():.2f}%")
            print(f"{'='*60}\n")

        return wf_df


# ─────────────────────────────────────────────
#  MONTE CARLO SIMULATION
# ─────────────────────────────────────────────
class MonteCarloAnalysis:
    """
    Bootstrap trade returns to estimate distribution of outcomes.
    Answers: "what's the range of possible equity curves?"
    """

    def __init__(self, n_simulations: int = 1000, seed: int = 42):
        self.n_sims = n_simulations
        self.seed   = seed

    def run(
        self,
        trade_returns:    pd.Series,    # per-trade return %
        initial_capital:  float = 1_000_000,
        n_trades:         int   = None,
    ) -> dict:
        rng = np.random.default_rng(self.seed)
        n   = n_trades or len(trade_returns)

        if len(trade_returns) < 5:
            print("[mc] Not enough trades for Monte Carlo. Need ≥5.")
            return {}

        final_values = []
        max_dds      = []
        equity_paths = np.empty((self.n_sims, n))

        returns_arr  = trade_returns.values

        for idx in range(self.n_sims):
            sampled = rng.choice(returns_arr, size=n, replace=True) / 100
            equity  = initial_capital * np.cumprod(1 + sampled)
            peak    = np.maximum.accumulate(equity)
            dd      = (equity - peak) / peak
            equity_paths[idx] = equity
            final_values.append(equity[-1])
            max_dds.append(dd.min())

        fv   = np.array(final_values)
        mdd  = np.array(max_dds)
        percentile_paths = {
            "p10": np.percentile(equity_paths, 10, axis=0),
            "p50": np.percentile(equity_paths, 50, axis=0),
            "p90": np.percentile(equity_paths, 90, axis=0),
        }
        sample_idx = rng.choice(self.n_sims, size=min(30, self.n_sims), replace=False)

        result = {
            "n_simulations":   self.n_sims,
            "median_final":    round(np.median(fv), 2),
            "p5_final":        round(np.percentile(fv, 5), 2),
            "p95_final":       round(np.percentile(fv, 95), 2),
            "prob_profit":     round((fv > initial_capital).mean() * 100, 2),
            "median_max_dd":   round(np.median(mdd) * 100, 2),
            "worst_case_dd":   round(np.percentile(mdd, 5) * 100, 2),
            "all_final_values": fv,
            "all_max_dds":     mdd,
            "sample_paths":    equity_paths[sample_idx],
            "percentile_paths": percentile_paths,
        }

        print(f"\n{'='*50}")
        print("MONTE CARLO RESULTS")
        print(f"{'='*50}")
        print(f"  Simulations:       {self.n_sims:,}")
        print(f"  Initial Capital:   ₹{initial_capital:,.0f}")
        print(f"  Median Final:      ₹{result['median_final']:,.0f}")
        print(f"  5th Pct Final:     ₹{result['p5_final']:,.0f}")
        print(f"  95th Pct Final:    ₹{result['p95_final']:,.0f}")
        print(f"  P(profit):         {result['prob_profit']:.1f}%")
        print(f"  Median Max DD:     {result['median_max_dd']:.2f}%")
        print(f"  Worst-case DD(5%): {result['worst_case_dd']:.2f}%")
        print(f"{'='*50}\n")

        return result


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data import generate_synthetic
    from strategies import MACrossStrategy
    from engine import BacktestEngine, RiskConfig

    df       = generate_synthetic(756)
    strategy = MACrossStrategy(fast=20, slow=50)
    engine   = BacktestEngine(strategy, risk_config=RiskConfig(sizing_method="kelly_fractional"))
    result   = engine.run(df)

    eq  = result.equity_curve["equity"]
    pm  = PerformanceMetrics(eq, result.trade_df)
    s   = pm.summary()

    print("=== PERFORMANCE SUMMARY ===")
    for k, v in s.items():
        print(f"  {k:30s}: {v}")
