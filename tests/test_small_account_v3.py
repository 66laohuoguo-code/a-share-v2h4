import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import build_residual_momentum_cache
from risk_aware_portfolio import WeeklyAlphaFeatureStore
from small_account_v3 import (
    commission_efficient_trade_floor,
    optimize_discrete_target_shares,
    select_cost_aware_codes,
)


class CostAwareSelectionTest(unittest.TestCase):
    def make_features(self):
        return pd.DataFrame(
            [
                {
                    "code": "000001",
                    "score_v2": 0.50,
                    "close": 10.0,
                    "industry_1": "A",
                },
                {
                    "code": "000002",
                    "score_v2": 0.45,
                    "close": 10.0,
                    "industry_1": "B",
                },
            ]
        )

    def test_minimum_commission_creates_economic_order_floor(self):
        self.assertAlmostEqual(
            commission_efficient_trade_floor(5.0, 0.002), 2500.0
        )

    def test_small_score_edge_does_not_replace_an_incumbent(self):
        selected, _, meta = select_cost_aware_codes(
            self.make_features(),
            {"000002": 100},
            target_equity_weight=0.80,
            portfolio_value=20000.0,
            target_count=1,
            minimum_holdings=1,
            buy_rank=2,
            sell_rank=10,
            maximum_industry_weight=1.0,
            slippage_bps=5.0,
            decision_date="2024-01-05",
            expected_return_per_score=0.005,
            hurdle_buffer_bps=10.0,
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
        )
        self.assertEqual(selected, ["000002"])
        self.assertEqual(meta["economic_replacements_blocked"], 1)

    def test_large_causal_edge_can_replace_an_incumbent(self):
        features = self.make_features()
        features.loc[features["code"] == "000001", "score_v2"] = 2.50
        selected, _, meta = select_cost_aware_codes(
            features,
            {"000002": 100},
            target_equity_weight=0.80,
            portfolio_value=20000.0,
            target_count=1,
            minimum_holdings=1,
            buy_rank=2,
            sell_rank=10,
            maximum_industry_weight=1.0,
            slippage_bps=5.0,
            decision_date="2024-01-05",
            expected_return_per_score=0.01,
            hurdle_buffer_bps=10.0,
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
        )
        self.assertEqual(selected, ["000001"])
        self.assertEqual(meta["economic_replacements_approved"], 1)

    def test_always_policy_replaces_without_an_expected_return_estimate(self):
        selected, _, meta = select_cost_aware_codes(
            self.make_features(),
            {"000002": 100},
            target_equity_weight=0.80,
            portfolio_value=20000.0,
            target_count=1,
            minimum_holdings=1,
            buy_rank=2,
            sell_rank=10,
            maximum_industry_weight=1.0,
            slippage_bps=5.0,
            decision_date="2024-01-05",
            expected_return_per_score=0.0,
            hurdle_buffer_bps=10.0,
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
            replacement_policy="always",
        )
        self.assertEqual(selected, ["000001"])
        self.assertEqual(meta["economic_replacements_approved"], 1)
        self.assertEqual(meta["economic_replacement_policy"], "always")

    def test_none_policy_keeps_the_incumbent(self):
        features = self.make_features()
        features.loc[features["code"] == "000001", "score_v2"] = 2.50
        selected, _, meta = select_cost_aware_codes(
            features,
            {"000002": 100},
            target_equity_weight=0.80,
            portfolio_value=20000.0,
            target_count=1,
            minimum_holdings=1,
            buy_rank=2,
            sell_rank=10,
            maximum_industry_weight=1.0,
            slippage_bps=5.0,
            decision_date="2024-01-05",
            expected_return_per_score=0.03,
            hurdle_buffer_bps=0.0,
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
            replacement_policy="none",
        )
        self.assertEqual(selected, ["000002"])
        self.assertEqual(meta["economic_replacement_policy"], "none")


class DiscreteOptimizerTest(unittest.TestCase):
    def test_optimizer_uses_legal_lots_and_keeps_cash_nonnegative(self):
        targets = {
            "000001": 0.20,
            "000002": 0.20,
            "000003": 0.20,
            "000004": 0.20,
        }
        prices = {
            "000001": 10.0,
            "000002": 20.0,
            "000003": 5.0,
            "000004": 8.0,
        }
        shares, meta = optimize_discrete_target_shares(
            targets,
            20000.0,
            prices,
            {},
            20000.0,
            "2024-01-08",
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
            minimum_final_holdings=4,
            maximum_stock_weight=0.35,
        )
        self.assertEqual(meta["integer_optimizer_status"], "applied")
        self.assertGreaterEqual(meta["integer_optimizer_projected_cash"], 0.0)
        self.assertEqual(len(shares), 4)
        self.assertTrue(all(value >= 100 and value % 100 == 0 for value in shares.values()))

    def test_optimizer_reserves_cash_for_target_without_execution_price(self):
        shares, meta = optimize_discrete_target_shares(
            {"000001": 0.40, "601288": 0.40},
            20000.0,
            {"000001": 10.0},
            {},
            20000.0,
            "2024-01-08",
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
            minimum_final_holdings=2,
            maximum_stock_weight=0.50,
        )
        self.assertEqual(meta["integer_optimizer_status"], "applied")
        self.assertEqual(meta["integer_optimizer_unavailable_targets"], ["601288"])
        self.assertAlmostEqual(
            meta["integer_optimizer_reserved_unavailable_weight"], 0.40
        )
        self.assertNotIn("601288", shares)
        self.assertGreaterEqual(meta["integer_optimizer_projected_cash"], 0.0)

    def test_optimizer_freezes_unpriced_existing_holding(self):
        shares, meta = optimize_discrete_target_shares(
            {"000001": 0.50},
            3000.0,
            {"000001": 10.0},
            {"601288": 100},
            2000.0,
            "2024-01-08",
            broker_commission_rate=0.0003,
            broker_minimum_commission=5.0,
            minimum_final_holdings=1,
            maximum_stock_weight=0.60,
        )
        self.assertEqual(meta["integer_optimizer_status"], "applied")
        self.assertGreaterEqual(meta["integer_optimizer_projected_cash"], 0.0)
        self.assertEqual(meta["integer_optimizer_projected_orders"], 1)
        self.assertNotIn("601288", shares)


class ResidualMomentumCacheTest(unittest.TestCase):
    def test_cache_excludes_the_most_recent_skip_weeks(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "risk.sqlite"
            conn = sqlite3.connect(database)
            conn.execute(
                """
                CREATE TABLE weekly_specific_return (
                    model_date TEXT, code TEXT, specific_return REAL
                )
                """
            )
            dates = pd.date_range("2022-01-07", periods=60, freq="W-FRI")
            conn.executemany(
                "INSERT INTO weekly_specific_return VALUES (?, '000001', 0.01)",
                ((date.date().isoformat(),) for date in dates),
            )
            conn.commit()
            conn.close()

            build_residual_momentum_cache.build(
                SimpleNamespace(
                    database=database,
                    lookback_weeks=52,
                    skip_weeks=4,
                    minimum_observations=26,
                    batch_size=1000,
                    overwrite=False,
                )
            )
            conn = sqlite3.connect(database)
            row = conn.execute(
                """
                SELECT model_date, observations, source_end_date
                FROM weekly_residual_momentum
                ORDER BY model_date DESC LIMIT 1
                """
            ).fetchone()
            self.assertEqual(row[0], dates[-1].date().isoformat())
            self.assertEqual(row[1], 48)
            self.assertLessEqual(row[2], dates[-5].date().isoformat())

            conn.execute(
                """
                CREATE TABLE weekly_exposure (
                    model_date TEXT, code TEXT, EARNINGS_YIELD REAL,
                    industry_group TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO weekly_exposure VALUES (?, '000001', 1.25, 'A')",
                (dates[-1].date().isoformat(),),
            )
            conn.commit()
            conn.close()

            store = WeeklyAlphaFeatureStore(database, 52, 4)
            try:
                features, meta = store.augment(
                    pd.DataFrame([{"code": "000001"}]),
                    dates[-1].date().isoformat(),
                )
                self.assertEqual(meta["alpha_feature_status"], "applied")
                self.assertAlmostEqual(features.iloc[0]["risk_earnings_yield_raw"], 1.25)
                self.assertGreater(features.iloc[0]["residual_momentum_raw"], 0.0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
