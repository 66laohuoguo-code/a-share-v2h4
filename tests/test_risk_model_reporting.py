import math
import unittest

import numpy as np
import pandas as pd

import portfolio_risk_report
import risk_model_reporting
import validate_risk_forecasts


class RiskModelPortfolioMathTest(unittest.TestCase):
    def test_portfolio_risk_includes_cash_through_subunit_equity_weights(self):
        exposures = pd.DataFrame(
            {
                "code": ["000001", "000002"],
                "industry_group": ["10", "20"],
                "BETA": [1.0, -1.0],
            }
        )
        covariance = pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.01]],
            index=["MARKET", "BETA"],
            columns=["MARKET", "BETA"],
        )
        weights = pd.Series(
            [0.25, 0.25], index=["000001", "000002"], dtype=float
        )
        specific = pd.Series(
            [0.09, 0.09], index=["000001", "000002"], dtype=float
        )

        result = risk_model_reporting.calculate_portfolio_risk(
            weights, exposures, covariance, specific
        )

        expected_common = 0.5**2 * 0.04
        expected_specific = 2 * 0.25**2 * 0.09
        self.assertAlmostEqual(result["common_variance"], expected_common)
        self.assertAlmostEqual(result["specific_variance"], expected_specific)
        self.assertAlmostEqual(
            result["annual_variance"],
            expected_common + expected_specific,
        )
        self.assertAlmostEqual(
            result["weekly_volatility"],
            math.sqrt(expected_common + expected_specific) / math.sqrt(52),
        )
        self.assertAlmostEqual(
            result["stock"]["total_variance_contribution"].sum(),
            result["annual_variance"],
        )

    def test_missing_model_exposure_uses_conservative_synthetic_row(self):
        holdings = pd.DataFrame({"code": ["000001", "999999"]})
        exposure = pd.DataFrame(
            {
                "code": ["000001", "000002"],
                "industry_group": ["10", "20"],
                "total_market_cap": [100.0, 200.0],
                "float_market_cap": [80.0, 160.0],
                "avg_amount_60": [10.0, 20.0],
                "specific_variance": [0.04, 0.16],
                "specific_volatility": [0.2, 0.4],
                "specific_observations": [30, 30],
                **{
                    factor: [0.1, -0.1]
                    for factor in risk_model_reporting.STYLE_FACTORS
                },
            }
        )
        covariance = pd.DataFrame(
            np.eye(2),
            index=["MARKET", "BETA"],
            columns=["MARKET", "BETA"],
        )

        completed = portfolio_risk_report.complete_holding_exposures(
            holdings, exposure, covariance
        ).set_index("code")

        self.assertEqual(
            completed.loc["999999", "exposure_source"],
            "synthetic_market_plus_specific_p90",
        )
        self.assertGreaterEqual(
            completed.loc["999999", "specific_variance"], 0.04
        )
        self.assertEqual(completed.loc["999999", "BETA"], 0.0)

    def test_cash_only_exposure_frame_is_valid_and_empty(self):
        exposure = pd.DataFrame(
            {
                "code": ["000001"],
                "industry_group": ["10"],
                "specific_variance": [0.04],
            }
        )
        for factor in risk_model_reporting.STYLE_FACTORS:
            exposure[factor] = 0.0
        covariance = pd.DataFrame(
            [[0.04]], index=["MARKET"], columns=["MARKET"]
        )

        completed = portfolio_risk_report.complete_holding_exposures(
            pd.DataFrame(columns=["code"]), exposure, covariance
        )
        result = risk_model_reporting.calculate_portfolio_risk(
            pd.Series(dtype=float),
            completed,
            covariance,
            pd.Series(dtype=float),
        )

        self.assertTrue(completed.empty)
        self.assertEqual(result["annual_variance"], 0.0)
        self.assertEqual(result["annual_volatility"], 0.0)


class RiskForecastCalibrationTest(unittest.TestCase):
    def test_test_portfolios_are_long_only_and_fully_invested(self):
        count = 100
        exposure = pd.DataFrame(
            {
                "code": [f"{index:06d}" for index in range(count)],
                "industry_group": ["10"] * 50 + ["20"] * 50,
                "total_market_cap": np.arange(1, count + 1, dtype=float),
                **{
                    factor: np.linspace(-2.0, 2.0, count)
                    for factor in risk_model_reporting.STYLE_FACTORS
                },
            }
        )
        portfolios = validate_risk_forecasts.build_test_portfolios(
            exposure, quantile=0.20, minimum_industry_stocks=30
        )

        names = {name for _, name, _ in portfolios}
        self.assertIn("MARKET_EW", names)
        self.assertIn("MARKET_CAP", names)
        self.assertIn("VALUE_LOW", names)
        self.assertIn("VALUE_HIGH", names)
        self.assertIn("INDUSTRY_10", names)
        for _, _, weights in portfolios:
            self.assertAlmostEqual(float(weights.sum()), 1.0)
            self.assertTrue((weights >= 0).all())

    def test_summary_recovers_known_variance_scale(self):
        rows = []
        for week in range(120):
            for portfolio_name, offset in (("A", 0.0), ("B", 0.01)):
                predicted_variance = 0.0004
                residual = 0.02 if week % 2 == 0 else -0.02
                rows.append(
                    {
                        "forecast_date": f"2024-{week // 4 + 1:02d}-{week % 4 + 1:02d}",
                        "realized_date": f"2024-{week // 4 + 1:02d}-{week % 4 + 2:02d}",
                        "portfolio_class": "test",
                        "portfolio_name": portfolio_name,
                        "stock_count": 20,
                        "realized_coverage": 1.0,
                        "predicted_annual_variance": predicted_variance * 52,
                        "predicted_annual_volatility": math.sqrt(
                            predicted_variance * 52
                        ),
                        "predicted_weekly_variance": predicted_variance,
                        "predicted_weekly_volatility": math.sqrt(
                            predicted_variance
                        ),
                        "realized_weekly_return": offset + residual,
                    }
                )
        observations = pd.DataFrame(rows)

        _, overall, summary = validate_risk_forecasts.summarize_forecasts(
            observations
        )

        self.assertAlmostEqual(
            overall[
                "calibration_ratio_realized_to_predicted_variance"
            ],
            1.0,
            places=6,
        )
        self.assertEqual(overall["status"], "calibrated")
        self.assertIn("checks", summary.columns)


if __name__ == "__main__":
    unittest.main()
