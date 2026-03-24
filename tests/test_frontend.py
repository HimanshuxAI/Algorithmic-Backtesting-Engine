import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import create_app


class FrontendSmokeTests(unittest.TestCase):
    def test_index_renders_with_output_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "run_manifest.json").write_text(json.dumps({
                "run_id": "demo",
                "status": "success",
                "duration_seconds": 1.23,
                "config": {"output_dir": str(output_dir), "strict": False},
                "successful_strategies": ["Mean Reversion (BB+Z)"],
                "failed_strategies": [],
            }), encoding="utf-8")
            (output_dir / "research_report.json").write_text(json.dumps({
                "dataset": {"source": "synthetic", "bars": 100, "start": "2024-01-01", "end": "2024-06-01"},
                "leaderboard": [{"rank": 1, "strategy": "Mean Reversion (BB+Z)", "resilience_score": 42.5, "total_return_pct": 3.1, "sharpe_ratio": 1.2, "max_drawdown_pct": -2.0}],
                "best_strategy": {"name": "Mean Reversion (BB+Z)", "summary": {"total_return_pct": 3.1, "max_drawdown_pct": -2.0}, "monte_carlo": {"prob_profit": 64.0}},
            }), encoding="utf-8")
            (output_dir / "strategy_health.csv").write_text(
                "name,status,started_at,finished_at,duration_seconds,trades,total_return_pct,sharpe_ratio,resilience_score,error_type,error_message\n"
                "Mean Reversion (BB+Z),success,2024-01-01,2024-01-01,0.15,8,3.1,1.2,42.5,,\n",
                encoding="utf-8",
            )

            app = create_app(output_dir)
            client = app.test_client()
            response = client.get("/")

            self.assertEqual(response.status_code, 200)
            page = response.get_data(as_text=True)
            self.assertIn("Quant Backtest Dashboard", page)
            self.assertIn("Mean Reversion (BB+Z)", page)

    def test_demo_run_triggers_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(tmpdir)
            client = app.test_client()

            with patch("app.run_pipeline") as mock_run:
                response = client.post("/run", data={"mode": "demo"})

            self.assertEqual(response.status_code, 302)
            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["data_source"], "synthetic")

    def test_manual_run_triggers_pipeline_with_form_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(tmpdir)
            client = app.test_client()

            with patch("app.run_pipeline") as mock_run:
                response = client.post("/run", data={
                    "mode": "manual",
                    "source": "yfinance",
                    "file": "",
                    "symbol": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-06-01",
                    "capital": "250000",
                    "sizing": "fixed_pct",
                    "strict": "on",
                })

            self.assertEqual(response.status_code, 302)
            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            self.assertEqual(kwargs["data_source"], "yfinance")
            self.assertEqual(kwargs["symbol"], "AAPL")
            self.assertEqual(kwargs["initial_capital"], 250000.0)
            self.assertEqual(kwargs["sizing_method"], "fixed_pct")
            self.assertTrue(kwargs["strict"])


if __name__ == "__main__":
    unittest.main()
