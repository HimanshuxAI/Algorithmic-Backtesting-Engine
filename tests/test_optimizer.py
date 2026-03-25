import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import generate_synthetic
from execution import CostConfig, RiskConfig
from optimizer import StrategyLabSpec, run_strategy_lab
from strategies import MACDRSIStrategy, MACrossStrategy


class StrategyLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic(n_bars=260, seed=19, regime_switches=True)
        cls.cost = CostConfig()
        cls.risk = RiskConfig(sizing_method="fixed_pct", initial_capital=750_000)
        cls.specs = [
            StrategyLabSpec(
                family_key="ma_cross",
                family_name="MA Cross",
                factory=MACrossStrategy,
                grid=[
                    {"fast": 10, "slow": 30, "ma_type": "ema", "trend_filter": False, "volume_confirm": False},
                    {"fast": 20, "slow": 50, "ma_type": "ema", "trend_filter": False, "volume_confirm": False},
                ],
                label_builder=lambda params: f"EMA {params['fast']}/{params['slow']}",
            ),
            StrategyLabSpec(
                family_key="macd_rsi",
                family_name="MACD+RSI",
                factory=MACDRSIStrategy,
                grid=[
                    {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "rsi_low": 30, "rsi_high": 72, "trend_filter": False},
                    {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "rsi_low": 40, "rsi_high": 75, "trend_filter": True},
                ],
                label_builder=lambda params: f"RSI {params['rsi_low']}-{params['rsi_high']}",
            ),
        ]

    def test_strategy_lab_returns_ranked_candidates_and_selected_configs(self):
        result = run_strategy_lab(
            self.df,
            base_cost_config=self.cost,
            base_risk_config=self.risk,
            train_ratio=0.70,
            family_specs=self.specs,
        )

        self.assertEqual(len(result["candidate_df"]), 4)
        self.assertEqual(len(result["summary_df"]), 2)
        self.assertEqual(len(result["selected_strategies"]), 2)
        self.assertEqual(int(result["candidate_df"]["selected"].sum()), 2)
        self.assertEqual(result["split"]["train_bars"] + result["split"]["test_bars"], len(self.df))
        self.assertTrue(all("train_score" in row for row in result["summary_records"]))
        self.assertTrue(all("name" in row for row in result["selected_strategies"]))


if __name__ == "__main__":
    unittest.main()
