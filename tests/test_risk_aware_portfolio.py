import tempfile
import unittest
from pathlib import Path
import sqlite3

import pandas as pd

import factor_rank_backtest_v2h
import risk_aware_portfolio


class WeeklyRiskModelStoreTest(unittest.TestCase):
    def test_model_date_lookup_never_uses_a_future_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "risk.sqlite"
            conn = sqlite3.connect(path)
            try:
                conn.executescript(
                    """
                    CREATE TABLE weekly_factor_covariance (
                        model_date TEXT, factor_1 TEXT, factor_2 TEXT,
                        covariance REAL
                    );
                    CREATE TABLE weekly_specific_risk (
                        model_date TEXT, code TEXT
                    );
                    CREATE TABLE weekly_exposure (
                        model_date TEXT, code TEXT
                    );
                    """
                )
                for model_date in ("2024-01-05", "2024-01-12"):
                    conn.execute(
                        "INSERT INTO weekly_factor_covariance VALUES (?, 'MARKET', 'MARKET', 0.04)",
                        (model_date,),
                    )
                    conn.execute(
                        "INSERT INTO weekly_specific_risk VALUES (?, '000001')",
                        (model_date,),
                    )
                    conn.execute(
                        "INSERT INTO weekly_exposure VALUES (?, '000001')",
                        (model_date,),
                    )
                conn.commit()
            finally:
                conn.close()

            store = risk_aware_portfolio.WeeklyRiskModelStore(path)
            try:
                self.assertIsNone(store.model_date_for("2024-01-04"))
                self.assertEqual(
                    store.model_date_for("2024-01-11"), "2024-01-05"
                )
                self.assertEqual(
                    store.model_date_for("2024-01-12"), "2024-01-12"
                )
            finally:
                store.close()


class CausalRiskCalibrationStoreTest(unittest.TestCase):
    def test_multiplier_lookup_never_uses_a_future_schedule_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.csv"
            pd.DataFrame(
                [
                    {
                        "as_of_date": "2024-01-05",
                        "multiplier": 1.10,
                        "forecast_weeks": 52,
                    },
                    {
                        "as_of_date": "2024-01-12",
                        "multiplier": 1.20,
                        "forecast_weeks": 53,
                    },
                ]
            ).to_csv(path, index=False)

            store = risk_aware_portfolio.CausalRiskCalibrationStore(path)
            value, metadata = store.multiplier_for(
                "2024-01-04", default=1.0
            )
            self.assertEqual(value, 1.0)
            self.assertIsNone(metadata["risk_calibration_as_of_date"])

            value, metadata = store.multiplier_for("2024-01-11")
            self.assertAlmostEqual(value, 1.10)
            self.assertEqual(
                metadata["risk_calibration_as_of_date"], "2024-01-05"
            )

            value, metadata = store.multiplier_for("2024-01-12")
            self.assertAlmostEqual(value, 1.20)
            self.assertEqual(
                metadata["risk_calibration_forecast_weeks"], 53
            )


class RiskAwareOptimizationTest(unittest.TestCase):
    def make_snapshot(self, specific):
        exposure = pd.DataFrame(
            {
                "code": ["000001", "000002"],
                "industry_group": ["10", "20"],
                "specific_variance": specific,
                "specific_volatility": [
                    value**0.5 for value in specific
                ],
                "exposure_source": ["model", "model"],
            }
        )
        covariance = pd.DataFrame(
            [[0.0]], index=["MARKET"], columns=["MARKET"]
        )
        return risk_aware_portfolio.RiskSnapshot(
            "2024-01-05", covariance, exposure
        )

    def test_variance_blend_moves_weight_toward_lower_risk_stock(self):
        baseline = pd.Series(
            [0.5, 0.5], index=["000001", "000002"], dtype=float
        )
        optimized, meta = (
            risk_aware_portfolio.optimize_risk_aware_weights(
                baseline,
                self.make_snapshot([0.36, 0.04]),
                stock_cap=1.0,
                maximum_industry_fraction=1.0,
                strength=0.50,
                target_volatility=1.0,
                calibration_multiplier=1.0,
                minimum_equity_scale=1.0,
            )
        )

        self.assertLess(optimized["000001"], 0.5)
        self.assertGreater(optimized["000002"], 0.5)
        self.assertLess(
            meta["risk_predicted_volatility_optimized"],
            meta["risk_predicted_volatility_before"],
        )
        self.assertEqual(meta["risk_overlay_status"], "applied")

    def test_volatility_cap_scales_equity_without_lookahead(self):
        baseline = pd.Series(
            [0.5, 0.5], index=["000001", "000002"], dtype=float
        )
        optimized, meta = (
            risk_aware_portfolio.optimize_risk_aware_weights(
                baseline,
                self.make_snapshot([0.04, 0.04]),
                stock_cap=1.0,
                maximum_industry_fraction=1.0,
                strength=0.0,
                target_volatility=0.10,
                calibration_multiplier=1.0,
                minimum_equity_scale=0.50,
            )
        )

        self.assertLess(float(optimized.sum()), 1.0)
        self.assertAlmostEqual(
            meta["risk_predicted_volatility_after"], 0.10, places=6
        )
        self.assertTrue(meta["risk_cap_met"])

    def test_deployed_small_account_config_enables_risk_overlay(self):
        args = factor_rank_backtest_v2h.parse_args(
            [
                "--strategy-config",
                "config/v22s_20k_entry_weight_monthly_06_official.json",
                "--risk-model-database",
                "risk.sqlite",
            ]
        )
        self.assertEqual(args.risk_overlay_mode, "variance_blend")
        self.assertAlmostEqual(args.risk_calibration_multiplier, 1.0)
        self.assertEqual(args.target_count, 12)

    def test_deployed_account_configs_preserve_their_base_constraints(self):
        small = factor_rank_backtest_v2h.parse_args(
            [
                "--strategy-config",
                "config/v22s_20k_entry_weight_monthly_06_official.json",
                "--risk-model-database",
                "risk.sqlite",
            ]
        )
        medium = factor_rank_backtest_v2h.parse_args(
            [
                "--strategy-config",
                "config/v22r3_weekly_560k_official.json",
                "--risk-model-database",
                "risk.sqlite",
            ]
        )

        self.assertEqual(small.target_count, 12)
        self.assertEqual(small.min_trade_weight, 0.075)
        self.assertEqual(small.risk_model_max_industry_weight, 0.40)
        self.assertEqual(medium.target_count, 20)
        self.assertEqual(medium.min_trade_weight, 0.03)
        self.assertEqual(medium.risk_model_max_industry_weight, 0.25)


if __name__ == "__main__":
    unittest.main()
