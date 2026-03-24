import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import PerformanceMetrics, WalkForwardValidator
from data import generate_synthetic
from engine import BacktestEngine
from execution import CostConfig, RiskConfig
from reporting import RegimeLens, StrategyScorecard, compute_monthly_stats, summarize_walk_forward
from strategies import MACrossStrategy


class BacktestSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic(n_bars=260, seed=7, regime_switches=True)
        cls.strategy = MACrossStrategy(fast=10, slow=30, trend_filter=False, volume_confirm=False)
        cls.risk = RiskConfig(
            sizing_method="fixed_pct",
            initial_capital=1_000_000,
            max_position_pct=0.08,
        )
        cls.engine = BacktestEngine(
            strategy=cls.strategy,
            cost_config=CostConfig(),
            risk_config=cls.risk,
            verbose=False,
        )

    def test_engine_produces_equity_curve(self):
        result = self.engine.run(self.df)
        self.assertIn("equity", result.equity_curve.columns)
        self.assertGreater(len(result.equity_curve), 50)

    def test_reporting_stack_builds_scorecard(self):
        result = self.engine.run(self.df)
        metrics = PerformanceMetrics(result.equity_curve["equity"], result.trade_df, benchmark=self.df["close"])
        monthly = compute_monthly_stats(result.equity_curve["equity"])
        walk_forward_df = WalkForwardValidator(
            train_size=120,
            test_size=40,
            step_size=20,
            verbose=False,
        ).run(self.df, self.strategy, cost_config=CostConfig(), risk_config=self.risk, verbose=False)
        regime_lens = RegimeLens()
        regime = regime_lens.evaluate(result.equity_curve["equity"], regime_lens.classify(self.df))

        scorecard = StrategyScorecard().build([{
            "name": "Smoke MA",
            "summary": metrics.summary(),
            "monthly_stats": monthly,
            "walk_forward": summarize_walk_forward(walk_forward_df),
            "regime_analysis": regime,
        }])

        self.assertFalse(scorecard.empty)
        self.assertIn("resilience_score", scorecard.columns)


if __name__ == "__main__":
    unittest.main()
