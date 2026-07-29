import tempfile
import unittest
from pathlib import Path

import pandas as pd

from capital_scale_analysis import infer_run_identity, rank_base_results, trade_metrics
from factor_rank_backtest_v2h import FeatureSnapshotCache, PriceDateStore


class CapitalScaleAnalysisTest(unittest.TestCase):
    def test_price_date_store_builds_and_evicts_daily_dictionaries_lazily(self):
        prices = pd.DataFrame(
            [
                {"trade_date": "2026-01-05", "code": "000001", "close": 10.0},
                {"trade_date": "2026-01-05", "code": "000002", "close": 20.0},
                {"trade_date": "2026-01-06", "code": "000001", "close": 11.0},
            ]
        )
        store = PriceDateStore(prices, max_cached_dates=1)
        self.assertEqual(store.get("2026-01-05")["000002"]["close"], 20.0)
        self.assertEqual(store.get("2026-01-06")["000001"]["close"], 11.0)
        self.assertEqual(list(store.cache), ["2026-01-06"])
        self.assertEqual(store.get("missing", {}), {})

    def test_feature_snapshot_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = FeatureSnapshotCache(Path(directory) / "features.sqlite")
            expected = pd.DataFrame(
                [{"code": "000001", "low_beta_score": 1.25}]
            )
            cache.put("fingerprint", "2026-01-05", expected)
            actual = cache.get("fingerprint", "2026-01-05")
            self.assertEqual(cache.count("fingerprint"), 1)
            pd.testing.assert_frame_equal(actual, expected)
            cache.close()

    def test_run_identity_uses_capital_scenario_and_variant_directories(self):
        root = Path("outputs/capital_scale")
        summary = root / "capital_00500000" / "base" / "stock25" / "run_summary.json"
        self.assertEqual(infer_run_identity(summary, root), ("base", "stock25"))

    def test_trade_metrics_detect_minimum_commission_and_capacity_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trades.csv"
            pd.DataFrame(
                {
                    "gross_amount": [1000.0, 20000.0],
                    "avg_amount_for_cap": [100000.0, 200000.0],
                }
            ).to_csv(path, index=False, encoding="utf-8-sig")
            metrics = trade_metrics(
                path,
                {
                    "broker_commission_rate": 0.0003,
                    "broker_minimum_commission": 5.0,
                },
            )
        self.assertEqual(metrics["minimum_commission_order_count"], 1)
        self.assertAlmostEqual(metrics["minimum_commission_order_share"], 0.5)
        self.assertAlmostEqual(metrics["estimated_broker_commission"], 11.0)
        self.assertAlmostEqual(metrics["participation_max"], 0.1)

    def test_robust_rank_is_calculated_separately_for_each_capital(self):
        frame = pd.DataFrame(
            [
                {
                    "initial_cash": 100000.0,
                    "variant": "stock20",
                    "annualized_return": 0.10,
                    "sharpe_no_risk_free": 1.0,
                    "max_drawdown": -0.10,
                    "holdout_return": 0.02,
                    "holdout_sharpe_no_risk_free": 0.8,
                },
                {
                    "initial_cash": 100000.0,
                    "variant": "stock40",
                    "annualized_return": 0.08,
                    "sharpe_no_risk_free": 0.8,
                    "max_drawdown": -0.12,
                    "holdout_return": 0.01,
                    "holdout_sharpe_no_risk_free": 0.6,
                },
                {
                    "initial_cash": 1000000.0,
                    "variant": "stock20",
                    "annualized_return": 0.07,
                    "sharpe_no_risk_free": 0.7,
                    "max_drawdown": -0.13,
                    "holdout_return": 0.00,
                    "holdout_sharpe_no_risk_free": 0.0,
                },
                {
                    "initial_cash": 1000000.0,
                    "variant": "stock40",
                    "annualized_return": 0.09,
                    "sharpe_no_risk_free": 0.9,
                    "max_drawdown": -0.09,
                    "holdout_return": 0.03,
                    "holdout_sharpe_no_risk_free": 1.0,
                },
            ]
        )
        ranked, winners = rank_base_results(frame)
        self.assertEqual(len(ranked), 4)
        self.assertEqual(
            winners.set_index("initial_cash")["variant"].to_dict(),
            {100000.0: "stock20", 1000000.0: "stock40"},
        )


if __name__ == "__main__":
    unittest.main()
