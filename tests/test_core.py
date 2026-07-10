import sqlite3
import tempfile
import unittest
from pathlib import Path

import ashare_utils
import database_status
import factor_rank_backtest


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


if __name__ == "__main__":
    unittest.main()
