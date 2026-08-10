import unittest

import pandas as pd

import factor_rank_backtest_v2h
import weekly_rebalance_v2h
from opening_auction import (
    CausalOpeningGapEstimator,
    OpeningGapEstimate,
    opening_auction_limit_price,
    opening_auction_order_is_marketable,
)


class OpeningAuctionEstimatorTest(unittest.TestCase):
    def test_limit_price_is_a_protective_boundary(self):
        estimate = OpeningGapEstimate(
            observations=252,
            expected_gap=0.001,
            lower_gap=-0.02,
            upper_gap=0.03,
            fill_probability=0.90,
            source="test",
        )

        buy_limit = opening_auction_limit_price(10.0, "BUY", estimate, 2.0)
        sell_limit = opening_auction_limit_price(10.0, "SELL", estimate, 2.0)

        self.assertEqual(buy_limit, 10.31)
        self.assertEqual(sell_limit, 9.79)
        self.assertTrue(
            opening_auction_order_is_marketable(10.20, buy_limit, "BUY")
        )
        self.assertFalse(
            opening_auction_order_is_marketable(10.40, buy_limit, "BUY")
        )
        self.assertTrue(
            opening_auction_order_is_marketable(9.90, sell_limit, "SELL")
        )
        self.assertFalse(
            opening_auction_order_is_marketable(9.70, sell_limit, "SELL")
        )

    def test_seed_respects_the_as_of_date(self):
        frame = pd.DataFrame(
            [
                {
                    "code": "000001",
                    "trade_date": "2026-07-01",
                    "prev_close": 10.0,
                    "open": 10.0,
                },
                {
                    "code": "000001",
                    "trade_date": "2026-07-02",
                    "prev_close": 10.0,
                    "open": 10.1,
                },
                {
                    "code": "000001",
                    "trade_date": "2026-07-03",
                    "prev_close": 10.0,
                    "open": 15.0,
                },
            ]
        )
        estimator = CausalOpeningGapEstimator(
            lookback_days=20,
            min_observations=10,
            max_absolute_gap=0.60,
        )
        estimator.seed_from_frame(frame, "2026-07-02")

        estimate = estimator.estimate("000001")

        self.assertEqual(estimate.observations, 2)
        self.assertLess(estimate.upper_gap, 0.10)

    def test_auction_backtest_does_not_assume_a_fill_above_the_limit(self):
        args = factor_rank_backtest_v2h.parse_args(
            [
                "--execution-model",
                "opening_auction_limit",
                "--min-trade-value",
                "0",
                "--min-trade-weight",
                "0",
                "--entry-exit-min-trade-value",
                "0",
                "--entry-exit-min-trade-weight",
                "0",
                "--max-participation-rate",
                "0",
                "--disable-limit-trade-filter",
            ]
        )
        estimator = CausalOpeningGapEstimator(
            lookback_days=20,
            min_observations=10,
            fill_probability=0.90,
        )
        for _ in range(20):
            estimator.update(
                {"000001": {"prev_close": 10.0, "open": 10.0}}
            )
        prices = {
            "000001": {
                "name": "Sample",
                "prev_close": 10.0,
                "open": 10.50,
            }
        }

        cash, executed, diagnostics = factor_rank_backtest_v2h.execute_trades_v2(
            "2026-07-06",
            "2026-07-03",
            {},
            10_000.0,
            {"000001": 0.50},
            prices,
            10_000.0,
            {},
            args,
            {"000001": 10.0},
            10_000.0,
            estimator,
        )

        self.assertEqual(cash, 10_000.0)
        self.assertEqual(executed, [])
        self.assertEqual(diagnostics["auction_order_attempts"], 1)
        self.assertEqual(diagnostics["auction_unmarketable_orders"], 1)

    def test_auction_share_count_uses_friday_information_not_monday_open(self):
        args = factor_rank_backtest_v2h.parse_args(
            [
                "--execution-model",
                "opening_auction_limit",
                "--min-trade-value",
                "0",
                "--min-trade-weight",
                "0",
                "--entry-exit-min-trade-value",
                "0",
                "--entry-exit-min-trade-weight",
                "0",
                "--max-participation-rate",
                "0",
            ]
        )
        estimator = CausalOpeningGapEstimator(
            lookback_days=20,
            min_observations=10,
            fill_probability=0.90,
        )
        for _ in range(20):
            estimator.update(
                {"000001": {"prev_close": 10.0, "open": 10.0}}
            )

        cash, executed, diagnostics = factor_rank_backtest_v2h.execute_trades_v2(
            "2026-07-06",
            "2026-07-03",
            {},
            10_000.0,
            {"000001": 0.95},
            {
                "000001": {
                    "name": "Sample",
                    "prev_close": 10.0,
                    "open": 9.0,
                }
            },
            10_000.0,
            {},
            args,
            {"000001": 10.0},
            10_000.0,
            estimator,
        )

        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0]["shares"], 900)
        self.assertEqual(executed[0]["price"], 9.0)
        self.assertEqual(diagnostics["auction_executed_orders"], 1)
        self.assertLess(cash, 10_000.0)

    def test_weekly_output_separates_expected_price_from_protective_limit(self):
        args = factor_rank_backtest_v2h.parse_args(
            [
                "--execution-model",
                "opening_auction_limit",
                "--min-trade-value",
                "0",
                "--min-trade-weight",
                "0",
                "--entry-exit-min-trade-value",
                "0",
                "--entry-exit-min-trade-weight",
                "0",
                "--max-participation-rate",
                "0",
            ]
        )

        class FixedEstimator:
            @staticmethod
            def estimate(_code):
                return OpeningGapEstimate(
                    observations=252,
                    expected_gap=0.001,
                    lower_gap=-0.02,
                    upper_gap=0.03,
                    fill_probability=0.90,
                    source="test",
                )

        orders, _, projected_cash, _ = weekly_rebalance_v2h.build_order_plan(
            holdings={},
            cash=10_000.0,
            targets={"000001": 0.50},
            feature_frame=pd.DataFrame(
                [
                    {
                        "code": "000001",
                        "industry_1": "TEST",
                        "rank": 1,
                        "score_v2": 1.0,
                        "avg_amount_60": 1_000_000.0,
                    }
                ]
            ),
            prices={
                "000001": {
                    "name": "Sample",
                    "close": 10.0,
                    "industry_1": "TEST",
                }
            },
            total_value=10_000.0,
            as_of_date="2026-07-03",
            strategy_args=args,
            opening_gap_estimator=FixedEstimator(),
        )

        self.assertEqual(len(orders), 1)
        order = orders.iloc[0]
        self.assertEqual(order["indicative_price"], 10.01)
        self.assertEqual(order["estimated_execution_price"], 10.01)
        self.assertEqual(order["broker_order_limit_price"], 10.31)
        self.assertEqual(order["cash_reservation_price"], 10.31)
        self.assertEqual(order["limit_price_role"], "MAXIMUM_BUY_PRICE")
        self.assertFalse(bool(order["limit_price_is_expected_fill"]))
        self.assertTrue(bool(order["cancel_if_unfilled_after_opening_auction"]))
        self.assertEqual(order["gross_amount"], order["shares"] * 10.31)
        self.assertEqual(
            order["expected_gross_amount"],
            order["shares"] * 10.01,
        )
        self.assertLessEqual(order["expected_fee"], order["estimated_fee"])
        self.assertAlmostEqual(
            projected_cash,
            10_000.0 - order["gross_amount"] - order["estimated_fee"],
        )


if __name__ == "__main__":
    unittest.main()
