import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analytics import WalkForwardValidator
from backend import PipelineConfig, evaluate_strategy
from data import generate_synthetic
from execution import CostConfig, RiskConfig
from reporting import RegimeLens
from strategies import MACrossStrategy, Strategy


class BrokenStrategy(Strategy):
    name = "BrokenStrategy"

    def generate_signals(self, df: pd.DataFrame):
        return pd.Series([True, False]), pd.Series([False])


class BackendHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic(n_bars=220, seed=11)
        cls.benchmark = cls.df["close"]
        cls.cost = CostConfig()
        cls.risk = RiskConfig(sizing_method="fixed_pct", initial_capital=500_000)
        cls.wf = WalkForwardValidator(train_size=120, test_size=40, step_size=20, verbose=False)
        cls.regime_lens = RegimeLens()
        cls.regimes = cls.regime_lens.classify(cls.df)

    def test_pipeline_config_rejects_missing_csv_path(self):
        with self.assertRaises(ValueError):
            PipelineConfig(data_source="csv", filepath=None).validate()

    def test_backend_handles_strategy_failure_without_crashing(self):
        result, audit = evaluate_strategy(
            name="Broken",
            strategy=BrokenStrategy(),
            df=self.df,
            benchmark=self.benchmark,
            base_cost_config=self.cost,
            base_risk_config=self.risk,
            wf_validator=self.wf,
            regime_lens=self.regime_lens,
            regimes=self.regimes,
            risk_free_rate=0.065,
        )
        self.assertIsNone(result)
        self.assertEqual(audit.status, "failed")
        self.assertIn("length", audit.error_message.lower())

    def test_backend_evaluates_healthy_strategy(self):
        result, audit = evaluate_strategy(
            name="Healthy",
            strategy=MACrossStrategy(fast=10, slow=30, trend_filter=False, volume_confirm=False),
            df=self.df,
            benchmark=self.benchmark,
            base_cost_config=self.cost,
            base_risk_config=self.risk,
            wf_validator=self.wf,
            regime_lens=self.regime_lens,
            regimes=self.regimes,
            risk_free_rate=0.065,
        )
        self.assertIsNotNone(result)
        self.assertEqual(audit.status, "success")
        self.assertIn("summary", result)


if __name__ == "__main__":
    unittest.main()
