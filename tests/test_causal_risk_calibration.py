import unittest

import pandas as pd

import build_causal_risk_calibration


class CausalRiskCalibrationScheduleTest(unittest.TestCase):
    def test_schedule_uses_only_returns_realized_by_each_as_of_date(self):
        observations = pd.DataFrame(
            [
                {
                    "forecast_date": "2024-01-05",
                    "realized_date": "2024-01-12",
                    "portfolio_name": "A",
                    "predicted_weekly_variance": 0.01,
                    "realized_weekly_return": 0.10,
                },
                {
                    "forecast_date": "2024-01-12",
                    "realized_date": "2024-01-19",
                    "portfolio_name": "A",
                    "predicted_weekly_variance": 0.01,
                    "realized_weekly_return": -0.10,
                },
                {
                    "forecast_date": "2024-01-19",
                    "realized_date": "2024-01-26",
                    "portfolio_name": "A",
                    "predicted_weekly_variance": 0.01,
                    "realized_weekly_return": 0.30,
                },
            ]
        )

        schedule = build_causal_risk_calibration.build_expanding_schedule(
            observations,
            minimum_forecast_weeks=2,
            default_multiplier=1.0,
        ).set_index("as_of_date")

        self.assertEqual(
            schedule.loc["2024-01-12", "status"],
            "default_insufficient_history",
        )
        self.assertAlmostEqual(
            schedule.loc["2024-01-19", "multiplier"], 1.0
        )
        self.assertGreater(
            schedule.loc["2024-01-26", "multiplier"],
            schedule.loc["2024-01-19", "multiplier"],
        )

    def test_optional_lookback_keeps_only_known_recent_forecast_weeks(self):
        observations = pd.DataFrame(
            [
                {
                    "forecast_date": f"2024-01-{day:02d}",
                    "realized_date": f"2024-02-{day:02d}",
                    "portfolio_name": name,
                    "predicted_weekly_variance": 0.01,
                    "realized_weekly_return": value,
                }
                for day, name, value in (
                    (1, "A", 0.01),
                    (2, "A", -0.01),
                    (3, "A", 0.02),
                )
            ]
        )

        schedule = build_causal_risk_calibration.build_expanding_schedule(
            observations,
            minimum_forecast_weeks=2,
            lookback_weeks=2,
        )

        self.assertEqual(int(schedule.iloc[-1]["forecast_weeks"]), 2)


if __name__ == "__main__":
    unittest.main()
