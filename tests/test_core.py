import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import ashare_utils
import database_status
import factor_rank_backtest
import factor_rank_backtest_v2h
import import_csmar_forward_quotation


class TradingRulesTest(unittest.TestCase):
    def test_historical_cost_schedule(self):
        self.assertAlmostEqual(
            ashare_utils.mandatory_trade_cost_rate("SELL", "2021-01-05", "000001"),
            0.0010887,
        )
        self.assertAlmostEqual(
            ashare_utils.mandatory_trade_cost_rate("SELL", "2024-01-05", "000001"),
            0.0005641,
        )

    def test_board_order_sizes(self):
        self.assertEqual(ashare_utils.buy_order_size_rules("000001"), (100, 100))
        self.assertEqual(ashare_utils.buy_order_size_rules("688001"), (200, 1))
        self.assertEqual(ashare_utils.buy_order_size_rules("830001"), (100, 1))

    def test_entry_exit_trade_floors_are_separate_from_adjustments(self):
        common = {
            "portfolio_value": 566_446.71,
            "min_trade_value": 20_000.0,
            "min_trade_weight": 0.0,
            "entry_exit_min_trade_value": 2_000.0,
            "entry_exit_min_trade_weight": 0.0,
        }
        entry_floor, entry_type = ashare_utils.trade_value_floor(
            current_shares=0, target_shares=1000, **common
        )
        exit_floor, exit_type = ashare_utils.trade_value_floor(
            current_shares=1000, target_shares=0, **common
        )
        adjust_floor, adjust_type = ashare_utils.trade_value_floor(
            current_shares=1000, target_shares=1200, **common
        )

        self.assertEqual((entry_floor, entry_type), (2_000.0, "ENTRY"))
        self.assertEqual((exit_floor, exit_type), (2_000.0, "EXIT"))
        self.assertEqual((adjust_floor, adjust_type), (20_000.0, "ADJUST"))

    def test_percentage_trade_floors_scale_with_portfolio_value(self):
        common = {
            "portfolio_value": 566_446.71,
            "min_trade_value": 0.0,
            "min_trade_weight": 0.02,
            "entry_exit_min_trade_value": 0.0,
            "entry_exit_min_trade_weight": 0.0035,
        }
        entry_floor, _ = ashare_utils.trade_value_floor(
            current_shares=0, target_shares=1000, **common
        )
        adjust_floor, _ = ashare_utils.trade_value_floor(
            current_shares=1000, target_shares=1200, **common
        )

        self.assertAlmostEqual(entry_floor, 1_982.563485)
        self.assertAlmostEqual(adjust_floor, 11_328.9342)

    def test_portfolio_lot_rounding_uses_residual_cash_for_an_extra_lot(self):
        shares = ashare_utils.round_portfolio_target_shares(
            {"000001": 0.035, "000002": 0.035},
            portfolio_value=10_000.0,
            prices={"000001": 3.0, "000002": 4.0},
        )
        self.assertEqual(shares["000001"], 100)
        self.assertEqual(shares["000002"], 100)
        self.assertEqual(shares["000001"] * 3.0 + shares["000002"] * 4.0, 700.0)

    def test_lot_aware_selection_skips_unaffordable_ranked_stock(self):
        features = pd.DataFrame(
            [
                {"code": "000001", "close": 100.0, "industry_1": "A"},
                {"code": "000002", "close": 5.0, "industry_1": "A"},
                {"code": "000003", "close": 6.0, "industry_1": "B"},
                {"code": "000004", "close": 7.0, "industry_1": "C"},
                {"code": "000005", "close": 8.0, "industry_1": "D"},
                {"code": "000006", "close": 9.0, "industry_1": "E"},
                {"code": "000007", "close": 10.0, "industry_1": "F"},
            ]
        )
        features["score_v2"] = list(reversed(range(len(features))))
        features["volatility_120"] = 0.20
        args = SimpleNamespace(
            target_count=5,
            sell_rank=10,
            buy_rank=10,
            max_industry_weight=0.40,
            slippage_bps=0.0,
            lot_aware_min_holdings=3,
            enable_lot_aware_selection=True,
            score_temperature=0.80,
            min_stock_volatility=0.08,
            inverse_vol_power=1.0,
            max_stock_weight=0.035,
            rebalance_band_weight=0.0025,
            lot_aware_stock_cap_multiplier=1.25,
            lot_aware_max_stock_weight=0.25,
            lot_aware_max_industry_weight=0.50,
        )

        selected, minimum_weights, meta = (
            factor_rank_backtest_v2h.select_lot_aware_codes(
                features,
                {},
                args,
                target_equity_weight=0.70,
                portfolio_value=10_000.0,
            )
        )

        self.assertNotIn("000001", selected)
        self.assertIn("000002", selected)
        self.assertEqual(len(selected), 5)
        self.assertLessEqual(float(minimum_weights.sum()), 0.70)
        self.assertGreater(meta["skipped_lot_too_expensive"], 0)

        targets, target_meta = factor_rank_backtest_v2h.build_targets_v2(
            features,
            {},
            {},
            args,
            target_equity_weight=0.70,
            portfolio_value=10_000.0,
        )
        self.assertAlmostEqual(sum(targets.values()), 0.70, places=6)
        self.assertLessEqual(target_meta["effective_max_stock_weight"], 0.25)
        for code, weight in targets.items():
            close = float(features.loc[features["code"].eq(code), "close"].iloc[0])
            minimum, _ = ashare_utils.buy_order_size_rules(code)
            self.assertGreaterEqual(weight * 10_000.0 + 1e-8, close * minimum)

    def test_corporate_action_accounting(self):
        holdings = {"003010": 1000}
        rows = {
            "003010": {
                "close": 11.13,
                "daily_return": 0.0046,
                "capital_return": -0.0144,
            }
        }
        cash, actions = factor_rank_backtest.apply_corporate_actions_before_open(
            holdings, rows, {"003010": 15.81}
        )
        self.assertEqual(holdings["003010"], 1400)
        self.assertAlmostEqual(cash, 300.39, places=2)
        self.assertEqual(len(actions), 1)


class DatabaseStatusTest(unittest.TestCase):
    def test_database_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.sqlite"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE stock_daily (code TEXT, trade_date TEXT)")
            conn.executemany(
                "INSERT INTO stock_daily VALUES (?, ?)",
                [("000001", "2026-01-05"), ("000002", "2026-01-06")],
            )
            conn.commit()
            conn.close()
            status = database_status.read_status(path)
            self.assertEqual(status["max_date"], "2026-01-06")
            self.assertEqual(status["rows"], 2)
            self.assertEqual(status["stocks"], 2)


class ForwardQuotationImportTest(unittest.TestCase):
    def test_incremental_anchor_is_latest_close_before_source_start(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE stock_daily (
                code TEXT,
                name TEXT,
                trade_date TEXT,
                close REAL,
                listed_state TEXT,
                currency TEXT,
                industry_1 TEXT,
                industry_2 TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO stock_daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("000001", "Sample", "2026-03-31", 10.0, "Norm", "CNY", "A", "A01"),
                ("000001", "Sample", "2026-07-03", 12.5, "Norm", "CNY", "A", "A01"),
                ("000001", "Sample", "2026-07-06", 13.0, "Norm", "CNY", "A", "A01"),
            ],
        )

        meta = import_csmar_forward_quotation.latest_database_meta_before(
            conn, "2026-07-06"
        )

        self.assertEqual(meta["000001"]["anchor_date"], "2026-07-03")
        self.assertEqual(meta["000001"]["last_close"], 12.5)
        conn.close()


if __name__ == "__main__":
    unittest.main()
